import subprocess

import pytest

from pipeline.common import atomic_json, capture, config, paths, run
from pipeline import install


@pytest.mark.parametrize("protocol", ["HTTP/1.1", "HTTP/2"])
def test_http_configuration_is_local_and_preserves_existing_comfyui(tmp_path, monkeypatch, protocol):
    global_config = tmp_path / "global.gitconfig"
    original = "[http]\n\tversion = HTTP/2\n"
    global_config.write_text(original)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    c = config(overrides=[f"workspace_root={(tmp_path / 'managed').as_posix()}", f"git_http_version={protocol}"])
    p = paths(c)
    old = tmp_path / "old-comfyui"
    run(["git", "init", old], capture_output=True)
    run(["git", "-C", old, "config", "--local", "http.version", "HTTP/2"])
    old_config = (old / ".git/config").read_bytes()
    run(["git", "init", p["comfy"]], capture_output=True)
    atomic_json(p["state"] / "checkout-owner.json", {"path": str(p["comfy"]), "commit": c["comfyui_ref"]})
    install.configure_git(c)
    assert capture(["git", "-C", p["comfy"], "config", "--local", "--get", "http.version"]) == protocol
    assert global_config.read_text() == original
    assert (old / ".git/config").read_bytes() == old_config
    assert capture(["git", "-C", p["comfy"], "status", "--porcelain"]) == ""
    # Older custom YAML files also get the HTTP/1.1 default without changing their contents.
    c.pop("git_http_version")
    install.configure_git(c)
    assert capture(["git", "-C", p["comfy"], "config", "--local", "--get", "http.version"]) == "HTTP/1.1"


def test_first_install_and_failed_fetch_retry_set_http11_before_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "unused.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    p = paths(c)
    protocols = []
    def command(args, **kwargs):
        if "fetch" in args:
            protocols.append(capture(["git", "-C", p["comfy"], "config", "--local", "--get", "http.version"]))
            raise subprocess.CalledProcessError(128, args)
        return run(args, **kwargs)
    monkeypatch.setattr(install, "run", command)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            install.install(c, require_cuda=False)
    assert protocols == ["HTTP/1.1", "HTTP/1.1"]
    assert not (p["state"] / "installed.json").exists()


def test_unmanaged_repository_git_config_is_preserved(tmp_path):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    p = paths(c)
    run(["git", "init", p["comfy"]], capture_output=True)
    original = (p["comfy"] / ".git/config").read_bytes()
    with pytest.raises(RuntimeError, match="outside this managed installation"):
        install.configure_git(c)
    assert (p["comfy"] / ".git/config").read_bytes() == original


def test_invalid_git_transport_rejected():
    with pytest.raises(ValueError, match="git_http_version"):
        config(overrides=["git_http_version=HTTP/3"])
