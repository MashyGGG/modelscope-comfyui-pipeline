import sys
from types import ModuleType

import pytest

from pipeline.common import config, sha256
from pipeline import models


def test_sdk_token_login_tuple_and_exact_single_file_arguments(tmp_path, monkeypatch):
    # Exercise our SDK integration without importing SDK, network or actual model weights.
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    source = tmp_path / "cached.bin"
    source.write_bytes(b"fixture-only")
    item = {"key": "test", "repo": "owner/model", "revision": "a"*40, "file": "split/model.bin",
            "directory": "vae", "size": source.stat().st_size, "sha256": sha256(source)}
    marker_cookie = object()
    download_calls = []
    class HubApi:
        def login(self, token):
            assert token == "test-token-never-logged"
            return "git-token", marker_cookie
    def model_file_download(repo, **kwargs):
        download_calls.append((repo, kwargs))
        assert kwargs["cookies"] is marker_cookie
        return str(source)
    for name in ("modelscope", "modelscope.hub", "modelscope.hub.api", "modelscope.hub.file_download"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["modelscope.hub.api"].HubApi = HubApi
    sys.modules["modelscope.hub.file_download"].model_file_download = model_file_download
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "test-token-never-logged")
    monkeypatch.setattr(models, "remote_metadata", lambda _: None)
    # Windows symlink creation may require a privilege; copying here only doubles deployment.
    def deploy(cached, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cached.read_bytes())
    monkeypatch.setattr(models, "link_model", deploy)
    models.download(c, [item])
    assert len(download_calls) == 1
    assert download_calls[0][1]["file_path"] == "split/model.bin"
    assert download_calls[0][1]["revision"] == "a"*40
    assert models.valid(models.target(c, item), item)


def test_failed_download_has_bounded_retries_and_no_secret_output(tmp_path, monkeypatch, capsys):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}", "download_attempts=2"])
    item = {"key": "test", "repo": "test/repo", "revision": "a"*40,
            "directory": "vae", "file": "model.bin", "size": 4, "sha256": "0"*64}
    calls = []
    def remote(_):
        calls.append(1)
        raise RuntimeError("secret-token-in-upstream-error")
    monkeypatch.setattr(models, "remote_metadata", remote)
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="cache preserved"):
        models.download(c, [item])
    assert len(calls) == 2
    assert "secret-token" not in capsys.readouterr().out
