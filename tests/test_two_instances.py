import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from pipeline.common import ROOT, config, load_data, paths
from pipeline import environment, install, models, transfer


def fixtures(base):
    c = config(overrides=[f"workspace_root={base.as_posix()}"])
    items = []
    for key, directory in (("sdxl", "checkpoints"), ("wan", "diffusion_models"), ("umt5", "text_encoders"), ("vae", "vae")):
        data = ("offline-fixture-" + key).encode()
        items.append({"key": key, "directory": directory, "file": key + ".bin", "size": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(), "repo": "test/repo", "revision": "a"*40})
        dest = models.target(c, items[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return c, items


def test_bundle_transfers_only_verified_weights_to_an_independent_root(tmp_path):
    cpu, items = fixtures(tmp_path / "cpu")
    (paths(cpu)["base"] / "credentials.txt").write_text("not-portable")
    bundle = tmp_path / "models.tar"
    transfer.export_models(cpu, items, bundle)
    with tarfile.open(bundle) as archive:
        assert set(archive.getnames()) == {"manifest.json", *(transfer.member_name(m) for m in items)}
        assert all(member.isfile() for member in archive)
    gpu = config(overrides=[f"workspace_root={(tmp_path / 'gpu').as_posix()}"])
    transfer.import_models(gpu, items, bundle)
    # Import is valid before ComfyUI installation, and must not occupy the future checkout directory.
    assert not paths(gpu)["comfy"].exists()
    models.verify_available(gpu, items)
    models.download(gpu, items, skip=True)  # Only deploy from imported cache; no network/SDK needed.
    models.verify_all(gpu, items)
    assert not (paths(gpu)["state"] / "installed.json").exists()
    assert not (paths(gpu)["base"] / "credentials.txt").exists()
    # Reimport and repeated deployment reuse the complete files.
    transfer.import_models(gpu, items, bundle)
    models.download(gpu, items, skip=True)
    with pytest.raises(RuntimeError, match="already exists"):
        transfer.export_models(cpu, items, bundle)


@pytest.mark.parametrize("damage", ["hash", "manifest", "path", "symlink"])
def test_bundle_damage_and_unsafe_entries_are_rejected(tmp_path, damage):
    cpu, items = fixtures(tmp_path / "cpu")
    bundle = tmp_path / "damaged.tar"
    manifest = {"format_version": 1, "models": copy.deepcopy(items)}
    if damage == "manifest":
        manifest["models"][0]["revision"] = "b"*40
    with tarfile.open(bundle, "w") as archive:
        content = json.dumps(manifest).encode()
        header = tarfile.TarInfo("manifest.json")
        header.size = len(content)
        archive.addfile(header, io.BytesIO(content))
        for index, item in enumerate(items):
            data = models.target(cpu, item).read_bytes()
            name = transfer.member_name(item)
            if damage == "hash" and index == 0:
                data = b"x" * len(data)
            if damage == "path" and index == 0:
                name = "../../outside"
            header = tarfile.TarInfo(name)
            header.size = len(data)
            if damage == "symlink" and index == 0:
                header.type, header.linkname, header.size = tarfile.SYMTYPE, "/etc/passwd", 0
                archive.addfile(header)
            else:
                archive.addfile(header, io.BytesIO(data))
    gpu = config(overrides=[f"workspace_root={(tmp_path / 'gpu').as_posix()}"])
    with pytest.raises(RuntimeError):
        transfer.import_models(gpu, items, bundle)
    assert not models.target(gpu, items[0]).exists()
    assert not list(paths(gpu)["cache"].rglob("*.partial-*"))


def test_cpu_diagnosis_skips_gpu_and_port_requirements(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    monkeypatch.setattr(environment.shutil, "which", lambda tool: None if tool == "nvidia-smi" else "/bin/" + tool)
    import pipeline.preflight as preflight
    monkeypatch.setattr(preflight, "gpu_info", lambda _: pytest.fail("CPU path queried GPU"))
    monkeypatch.setattr(preflight, "port_open", lambda _: pytest.fail("CPU path checked inference port"))
    result = environment.diagnose(c, require_gpu=False, probe_torch=False)
    checks = {x["name"]: x for x in result["checks"]}
    for name in ("nvidia-smi", "GPU", "NVIDIA driver", "GPU memory", "ComfyUI port"):
        assert checks[name]["status"] == "SKIP"
    assert result["mode"] == "cpu-prepare"


def test_cpu_installer_import_gate_never_requests_cuda_computation(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    calls = []
    monkeypatch.setattr(install, "run", lambda command, **kw: calls.append((command, kw)))
    install.validate_runtime(c, require_cuda=False)
    assert len(calls) == 1 and calls[0][0][1] == "-c"
    assert "cuda.is_available" not in calls[0][0][2] and "device='cuda'" not in calls[0][0][2]
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    calls.clear()
    install.validate_runtime(c, require_cuda=True)
    assert "--cuda" in calls[0][0]


def test_cpu_cli_prepares_without_starting_service_or_gpu_sampling(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cli
    import pipeline.preflight as preflight
    from pipeline.logging import Operation
    args = cli.parser().parse_args(["--cpu-prepare"])
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    calls = []
    monkeypatch.setattr(preflight, "host", lambda c, **kw: calls.append(("host", kw)))
    monkeypatch.setattr(install, "install", lambda c, **kw: calls.append(("install", kw)))
    class Child:
        stdout = ["[PASS] offline runtime stage\n"]
        def __init__(self, command, **kw): calls.append(("stage", command))
        def wait(self): return 0
    monkeypatch.setattr(cli.subprocess, "Popen", Child)
    with Operation(tmp_path / "logs", "test-cpu") as operation:
        cli.operate(args, c, models.specifications(), paths(c), operation)
    assert calls[0][1]["require_gpu"] is False
    assert calls[1][1]["require_cuda"] is False
    stage_commands = [value for kind, value in calls if kind == "stage"]
    assert [command[command.index("--stage") + 1] for command in stage_commands] == ["download", "prepare"]
    assert all("--cuda" not in command for command in stage_commands)
    state = load_data(tmp_path / "state/cpu-preparation.json")
    assert state["status"] == "PREPARED" and state["gpu_e2e"] == "NOT RUN"
    assert "E2E VALIDATION PASSED" not in operation.log.read_text()


def test_gpu_validation_requires_models_before_installing(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cli
    import pipeline.preflight as preflight
    from pipeline.logging import Operation
    args = cli.parser().parse_args(["--gpu-validate"])
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    monkeypatch.setattr(preflight, "host", lambda *a, **kw: None)
    monkeypatch.setattr(install, "install", lambda *a, **kw: pytest.fail("should fail before install"))
    with pytest.raises(RuntimeError, match="Prepared model missing"):
        with Operation(tmp_path / "logs", "test-gpu") as operation:
            cli.operate(args, c, models.specifications(), paths(c), operation)
    assert load_data(operation.summary)["status"] == "FAILED"


def test_gpu_cli_enforces_cuda_and_disables_model_downloads(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cli
    import pipeline.preflight as preflight
    monkeypatch.setattr(cli, "sys", SimpleNamespace(platform="linux", executable=sys.executable, argv=["cli.py"]))
    calls = []
    monkeypatch.setattr(preflight, "host", lambda c, **kw: calls.append(("host", kw)))
    monkeypatch.setattr(models, "verify_available", lambda *a: calls.append(("models", None)))
    monkeypatch.setattr(install, "install", lambda c, **kw: calls.append(("install", kw)))
    class Child:
        stdout = ["[PASS] offline stage fixture\n"]
        def __init__(self, command, **kw): calls.append(("stage", command))
        def wait(self): return 0
    monkeypatch.setattr(cli.subprocess, "Popen", Child)
    cli.main(["--gpu-validate", "--set", f"workspace_root={tmp_path.as_posix()}"])
    assert calls[0][1]["require_gpu"] is True
    assert ("install", {"require_cuda": True}) in calls
    commands = [v for k, v in calls if k == "stage"]
    assert [cmd[cmd.index("--stage")+1] for cmd in commands] == ["preflight", "download", "prepare", "start", "e2e"]
    assert "--cuda" in commands[0]
    assert all("--skip-download" in cmd for cmd in commands)
    assert not (tmp_path / "state/cpu-preparation.json").exists()
