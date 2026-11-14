import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pipeline.common import ROOT, config, load_data, run
from pipeline.logging import Operation
from pipeline import environment


def test_all_output_child_stderr_and_failure_are_logged_once(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "test-sensitive-token")
    original_log = os.environ.get("PIPELINE_LOG_PATH")
    with pytest.raises(RuntimeError):
        with Operation(tmp_path / "logs", "test", ["arg-test-sensitive-token"], tmp_path / "state") as operation:
            with operation.phase("Install test"):
                print("parent-output-unique")
                print("stderr-output-unique", file=sys.stderr)
                run([sys.executable, "-c", "import sys; print('child-output-unique'); print('child-stderr-unique',file=sys.stderr)"])
                raise RuntimeError("failure test-sensitive-token")
    text = operation.log.read_text(encoding="utf-8")
    # Command text contains literal fixture strings too; count standalone emitted lines.
    assert text.count(" parent-output-unique\n") == 1
    assert text.count(" child-output-unique\n") == 1
    assert "stderr-output-unique" in text and "child-stderr-unique" in text
    assert "Traceback" in text and "[FAIL] Install test" in text
    assert "test-sensitive-token" not in text
    summary = load_data(operation.summary)
    assert summary["status"] == "FAILED" and summary["phases"][0]["status"] == "FAILED"
    assert "test-sensitive-token" not in operation.summary.read_text()
    assert summary["seconds"] >= 0
    assert load_data(tmp_path / "state/latest-operation.json")["log"] == str(operation.log)
    assert os.environ.get("PIPELINE_LOG_PATH") == original_log


def test_stop_has_its_own_log_even_without_installation(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cli
    from pipeline import server
    # Run the real stop control flow with only the platform and target process doubled.
    monkeypatch.setattr(cli, "sys", SimpleNamespace(platform="linux", executable=sys.executable, argv=["cli.py"]))
    monkeypatch.setattr(server, "stop", lambda c: print("[PASS] No managed process (offline test)"))
    cli.main(["--stop", "--set", f"workspace_root={tmp_path.as_posix()}"])
    logs = list((tmp_path / "logs").glob("stop-*.log"))
    assert len(logs) == 1
    assert "Stop managed ComfyUI" in logs[0].read_text()
    assert load_data(logs[0].with_suffix(".json"))["status"] == "PASSED"


def test_diagnosis_collects_later_checks_after_failures(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    monkeypatch.setattr(environment.platform, "system", lambda: "UnsupportedOS")
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    result = environment.diagnose(c, probe_torch=False)
    checks = {item["name"]: item for item in result["checks"]}
    assert checks["Operating system"]["status"] == "FAIL"
    assert checks["nvidia-smi"]["status"] == "FAIL"
    assert "Workspace disk" in checks and "ComfyUI port" in checks
    assert result["status"] == "FAILED"


def test_standalone_check_writes_json_and_log_before_install(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_environment.py"), "--log-dir", str(tmp_path),
                             "--set", f"workspace_root={(tmp_path / 'runtime').as_posix()}"],
                            capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    # On a Linux GPU test host the environment may pass; on local Windows it must fail.
    assert result.returncode in (0, 1)
    log = list(tmp_path.glob("environment-*.log"))
    assert len(log) == 1
    assert "[CHECK]" in log[0].read_text(encoding="utf-8")
    report = load_data(next(tmp_path.glob("environment-*-checks.json")))
    assert len(report["checks"]) >= 10
    assert not (tmp_path / "runtime/venv").exists()


def test_strict_cuda_check_cannot_pass_using_only_system_python(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    monkeypatch.setattr(environment.shutil, "which", lambda _: None)
    result = environment.diagnose(c, require_cuda=True)
    probe = next(x for x in result["checks"] if x["name"] == "PyTorch CUDA computation")
    assert probe["status"] == "FAIL" and "Managed runtime is missing" in probe["detail"]


def test_configuration_errors_are_persisted(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cli
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        cli.main(["--set", "video.frames=48"])
    logs = list((tmp_path / "logs").glob("configuration-*.log"))
    assert len(logs) == 1
    assert "video.frames" in logs[0].read_text()
    assert load_data(logs[0].with_suffix(".json"))["status"] == "FAILED"
