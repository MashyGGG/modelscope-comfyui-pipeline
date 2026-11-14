import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline.common import ROOT, atomic_json, checked_path, config
from pipeline import preflight


def test_dry_run_has_no_side_effects(tmp_path):
    base = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/cli.py"), "--dry-run"],
                            env={**os.environ, "WORKSPACE_ROOT": str(base)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "GPU E2E: NOT RUN" in result.stdout
    assert "E2E VALIDATION PASSED" not in result.stdout
    assert not base.exists()


@pytest.mark.parametrize("setting", ["video.frames=48", "video.width=833", "task_timeout=0", "video.fps=0", "image.seed=-1"])
def test_invalid_settings_rejected(setting):
    with pytest.raises(ValueError):
        config(overrides=[setting])


def test_no_gpu_fails_explicitly(monkeypatch):
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: None if tool == "nvidia-smi" else "/bin/git")
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        preflight.host(config())


def test_path_escape_and_symlink_rejected(tmp_path):
    with pytest.raises(ValueError):
        checked_path(tmp_path, "../escape")
    with pytest.raises(ValueError):
        checked_path(tmp_path, str(tmp_path.parent / "escape"))


def test_environment_and_cli_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFYUI_PORT", "8200")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    c = config(overrides=["port=8300"])
    assert c["port"] == 8300
    assert Path(c["comfyui_dir"]) == tmp_path / "ComfyUI"
    resolved = tmp_path / "resolved.json"
    atomic_json(resolved, c)
    assert config(resolved, use_env=False)["port"] == 8300
