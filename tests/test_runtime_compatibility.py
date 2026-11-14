import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pipeline import common, install, preflight, server
from pipeline.common import atomic_json, capture, config, load_data, paths, run


@pytest.mark.parametrize("driver,status", [
    ("525.60.12", "FAIL"), ("525.60.13", "WARN"),
    ("550.54.15", "WARN"), ("560.35.03", "PASS"), ("580.65.06", "PASS"),
])
def test_driver_policy_separates_compatibility_floor_and_recommended_version(driver, status):
    result, detail = preflight.driver_status(config(), driver)
    assert result == status
    if status == "WARN":
        assert "actual CUDA operations and full model inference must pass" in detail


def test_python_versions_select_different_environments_and_service_identities(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    monkeypatch.setattr(common, "runtime_tag", lambda: "py3.11")
    cpu, cpu_key = paths(c), server.server_key(c)
    monkeypatch.setattr(common, "runtime_tag", lambda: "py3.12")
    gpu, gpu_key = paths(c), server.server_key(c)
    assert cpu["venv"] != gpu["venv"] and cpu["install_marker"] != gpu["install_marker"]
    assert cpu["python"] != gpu["python"] and cpu_key != gpu_key
    assert cpu["comfy"] == gpu["comfy"] and cpu["cache"] == gpu["cache"]
    assert cpu["output"] == gpu["output"]


def test_cuda_unavailable_cannot_be_reported_as_pass(monkeypatch):
    fake = SimpleNamespace(__version__="2.8.0+cu126", version=SimpleNamespace(cuda="12.6"),
                           cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="CPU inference is disabled"):
        preflight.cuda()


def source_and_legacy_fixture(tmp_path):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    p = paths(c)
    run(["git", "init", p["comfy"]], capture_output=True)
    run(["git", "-C", p["comfy"], "-c", "user.name=OfflineTests", "-c", "user.email=tests@example.invalid",
         "commit", "--allow-empty", "-m", "Offline fixture"], capture_output=True)
    c["comfyui_ref"] = capture(["git", "-C", p["comfy"], "rev-parse", "HEAD"])
    atomic_json(p["state"] / "checkout-owner.json", {"path": str(p["comfy"]), "commit": c["comfyui_ref"]})
    legacy = p["base"] / "venv/bin/python"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy-python-do-not-execute")
    atomic_json(p["state"] / "installed.json", {"fingerprint": "legacy-python-3.11"})
    p["cache"].mkdir()
    (p["cache"] / "existing-model.bin").write_bytes(b"preserve-existing-model")
    return c, p, legacy


@pytest.mark.parametrize("cuda_fails", [False, True])
def test_gpu_rebuild_preserves_legacy_environment_source_and_models(tmp_path, monkeypatch, cuda_fails):
    c, p, legacy = source_and_legacy_fixture(tmp_path)
    original_marker = (p["state"] / "installed.json").read_bytes()
    commands, validations = [], []
    def command(args, **kwargs):
        args = [str(x) for x in args]
        commands.append(args)
        if args[0] == "git":
            assert "fetch" not in args  # Existing pinned source must be reused without network.
            return run(args, **kwargs)
        if args[1:3] == ["-m", "venv"]:
            assert Path(args[3]) == p["venv"] and p["venv"] != legacy.parent.parent
            p["python"].parent.mkdir(parents=True)
            p["python"].write_text("new-version-specific-python")
        return subprocess.CompletedProcess(args, 0)
    def captured(args):
        if str(args[0]) == str(p["python"]):
            if "-c" in args:
                return json.dumps({"version": list(sys.version_info[:2]), "prefix": str(p["venv"])})
            if "freeze" in args:
                return "torch==2.8.0+cu126"
            pytest.fail(f"Unexpected interpreter probe: {args}")
        return capture(args)
    def validate(c, require_cuda):
        validations.append(require_cuda)
        if cuda_fails:
            raise RuntimeError("actual CUDA operation failed")
    monkeypatch.setattr(install, "run", command)
    monkeypatch.setattr(install, "capture", captured)
    monkeypatch.setattr(install, "validate_runtime", validate)
    if cuda_fails:
        with pytest.raises(RuntimeError, match="actual CUDA operation failed"):
            install.install(c, require_cuda=True)
        assert not p["install_marker"].exists()
        assert not (p["state"] / "active-environment.json").exists()
    else:
        install.install(c, require_cuda=True)
        state = load_data(p["install_marker"])
        assert state["venv"] == str(p["venv"]) and state["last_validation"] == "GPU_CUDA"
        assert state["python_version"] == list(sys.version_info[:2])
        assert p["freeze"].is_file()
        assert load_data(p["state"] / "active-environment.json")["python"] == str(p["python"])
        count = len([cmd for cmd in commands if "pip" in cmd])
        install.install(c, require_cuda=True)
        assert len([cmd for cmd in commands if "pip" in cmd]) == count
    assert validations and all(validations)
    assert legacy.read_text() == "legacy-python-do-not-execute"
    assert (p["state"] / "installed.json").read_bytes() == original_marker
    assert (p["cache"] / "existing-model.bin").read_bytes() == b"preserve-existing-model"


def test_python_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    p = paths(c)
    p["python"].parent.mkdir(parents=True)
    p["python"].touch()
    monkeypatch.setattr(install, "capture", lambda _: json.dumps({"version": [3, 9], "prefix": str(p["venv"])}))
    with pytest.raises(RuntimeError, match="identity differs"):
        install.verify_python(c)


def test_server_from_another_python_can_be_stopped_but_not_reused(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    monkeypatch.setattr(common, "runtime_tag", lambda: "py3.11")
    p = paths(c)
    identity = {"start_ticks": "original-process", "cmdline": "managed-command", "uid": 0}
    state = {"pid": 1234, "identity": identity, "config_hash": server.server_key(c),
             "command": [str(p["python"]), str(p["comfy"] / "main.py")]}
    atomic_json(p["state"] / "server.json", state)
    monkeypatch.setattr(server, "identity", lambda _: identity)
    monkeypatch.setattr(common, "runtime_tag", lambda: "py3.12")
    with pytest.raises(RuntimeError, match="mismatch"):
        server.owned(c)
    assert server.owned(c, stopping=True)["pid"] == 1234
