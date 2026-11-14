"""Offline batch state-machine tests; synthetic media is not GPU evidence."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import av
from PIL import Image
import pytest

from pipeline import batch, models, orchestrator, workflows
from pipeline.common import ROOT, atomic_json, config, fingerprint, load_data, paths, sha256, task_configs


def batch_config(tmp_path, tasks=None):
    source = tmp_path / "jobs.json"
    atomic_json(source, {
        "image": {"width": 64, "height": 32},
        "video": {"width": 64, "height": 32, "frames": 5},
        "tasks": tasks if tasks is not None else [
            {"name": name, "image": {"prompt": f"Offline {name}", "seed": seed},
             "video": {"prompt": f"Offline motion {name}"}}
            for name, seed in (("lake", 101), ("coast", 102), ("forest", 103))
        ],
    })
    return config(source, [f"workspace_root={(tmp_path / 'runtime').as_posix()}"], use_env=False)


def test_partial_batch_config_and_defaults_are_independent(tmp_path):
    c = batch_config(tmp_path)
    original = deepcopy(c)
    plan = task_configs(c)
    assert len(plan) == 3
    assert plan[0][1]["image"]["seed"] == 101
    assert plan[0][1]["video"]["seed"] == 20261115
    assert plan[0][1]["image"]["steps"] == 24
    plan[0][1]["image"]["prompt"] = "changed"
    assert plan[1][1]["image"]["prompt"] == "Offline coast"
    assert c == original and "tasks" not in plan[0][1]
    assert paths(c) == paths(plan[0][1])  # Reuse the same managed service/models/environment.


@pytest.mark.parametrize("tasks", [
    [], None, "invalid", ["invalid"],
    [{"name": "same"}, {"name": "same"}],
    [{"name": "../escape"}], [{"name": "x", "workspace_root": "/tmp/other"}],
    [{"name": "x", "image": {"promt": "typo"}}],
    [{"name": "x", "video": []}],
    [{"name": "valid"}, {"name": "invalid", "video": {"frames": 48}}],
])
def test_invalid_batch_is_rejected_before_execution(tmp_path, tasks):
    source = tmp_path / "invalid.json"
    atomic_json(source, {"tasks": tasks})
    with pytest.raises(ValueError):
        config(source, use_env=False)


def test_legacy_single_config_snapshot_remains_compatible(tmp_path):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"], use_env=False)
    assert "tasks" not in c
    snapshot = tmp_path / "single.json"
    atomic_json(snapshot, c)
    assert config(snapshot, use_env=False) == c
    assert fingerprint(config(snapshot, use_env=False)) == fingerprint(c)


def test_batch_dry_run_has_no_runtime_writes(tmp_path):
    runtime = tmp_path / "must-not-exist"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/cli.py"), "--dry-run", "--config",
        str(ROOT / "config/batch.example.yaml"), "--set", f"workspace_root={runtime.as_posix()}",
    ], capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stderr
    assert "Batch: 2 tasks" in result.stdout
    assert "Task lake:" in result.stdout and "Task lighthouse:" in result.stdout
    assert "GPU E2E: NOT RUN" in result.stdout
    assert not runtime.exists()


def test_batch_workflow_previews_use_each_tasks_prompts(tmp_path):
    c = batch_config(tmp_path)
    workflows.prepare(c, models.specifications())
    for name, effective in task_configs(c):
        api = load_data(paths(c)["base"] / "workflows/api" / name / "01_sdxl_txt2img_api.json")
        ui = load_data(paths(c)["comfy"] / "user/default/workflows/modelscope-pipeline" / name / "01_sdxl_txt2img.json")
        assert api["2"]["inputs"]["text"] == effective["image"]["prompt"]
        assert ui == workflows.to_ui(api)


@pytest.fixture
def offline_executor(monkeypatch):
    calls = []
    behavior = {"fail_seed": None, "interrupt_after_pass": False}

    def execute(c, items, resume=None, *, run_id=None):
        run_id = resume or run_id
        calls.append({"seed": c["image"]["seed"], "resume": resume, "run_id": run_id})
        output = paths(c)["output"]
        state_file = output / "reports" / run_id / "state.json"
        if resume:
            state = load_data(state_file)
            image = Path(state["image"]["artifact"]["path"])
            assert sha256(image) == state["image"]["artifact"]["sha256"]
        else:
            image = output / "images" / run_id / "offline.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 32), (c["image"]["seed"], 60, 90)).save(image)
            state = {"run_id": run_id, "config": c, "config_hash": fingerprint(c),
                     "models_hash": fingerprint(items), "status": "RUNNING",
                     "image": {"status": "PASSED", "artifact": batch.media.png(image, c["image"])}}
        if c["image"]["seed"] == behavior["fail_seed"]:
            state["status"] = "FAILED"
            atomic_json(state_file, state)
            raise RuntimeError("OFFLINE simulated video failure")
        video = output / "videos" / run_id / "offline.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(video), "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=16)
            stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
            for i in range(5):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64, 32), (i * 20, 60, 90)))
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        state.update(status="PASSED", video={"status": "PASSED", "artifact": batch.media.video(video, c["video"])})
        atomic_json(state_file, state)
        if behavior["interrupt_after_pass"]:
            raise KeyboardInterrupt("OFFLINE interrupt after child checkpoint")
        return state

    monkeypatch.setattr(orchestrator, "execute_single", execute)
    return calls, behavior


def test_batch_runs_in_order_with_independent_outputs_and_logs(tmp_path, offline_executor):
    calls, _ = offline_executor
    c, items = batch_config(tmp_path), models.specifications()
    state = orchestrator.execute(c, items)
    assert state["status"] == "PASSED"
    assert [call["seed"] for call in calls] == [101, 102, 103]
    assert len({call["run_id"] for call in calls}) == 3
    for entry in state["tasks"]:
        assert entry["status"] == "PASSED" and entry["result"]["video"]["fully_decoded"]
        attempt = entry["attempts"][0]
        assert load_data(attempt["summary"])["status"] == "PASSED"
        assert "[START] Batch task" in Path(attempt["log"]).read_text(encoding="utf-8")
        assert entry["run_id"] in entry["result"]["image"]["path"]
    assert load_data(paths(c)["output"] / "batches/batch_report.json")["batch_id"] == state["batch_id"]
    with pytest.raises(RuntimeError, match="already passed"):
        orchestrator.execute(c, items, state["batch_id"])


def test_batch_failure_and_resume_skip_completed_and_continue_remaining(tmp_path, offline_executor):
    calls, behavior = offline_executor
    c, items = batch_config(tmp_path), models.specifications()
    behavior["fail_seed"] = 102
    with pytest.raises(RuntimeError, match="simulated video failure"):
        orchestrator.execute(c, items)
    failed = load_data(paths(c)["output"] / "batches/batch_report.json")
    assert [t["status"] for t in failed["tasks"]] == ["PASSED", "FAILED", "PENDING"]
    first_checkpoint = paths(c)["output"] / "reports" / failed["tasks"][0]["run_id"] / "state.json"
    first_bytes = first_checkpoint.read_bytes()
    failed_attempt = failed["tasks"][1]["attempts"][0]
    assert load_data(failed_attempt["summary"])["status"] == "FAILED"
    behavior["fail_seed"] = None
    resumed = orchestrator.execute(c, items, failed["batch_id"])
    assert resumed["status"] == "PASSED"
    assert [call["seed"] for call in calls] == [101, 102, 102, 103]
    assert calls[2]["resume"] == failed["tasks"][1]["run_id"]
    assert calls[3]["resume"] is None
    assert first_checkpoint.read_bytes() == first_bytes
    assert "[SKIP] Task lake" in Path(resumed["tasks"][0]["attempts"][-1]["log"]).read_text(encoding="utf-8")
    assert list(batch.batch_directory(c, failed["batch_id"]).glob("attempt-*/batch_report.json"))


def test_batch_recovers_child_passed_before_batch_checkpoint(tmp_path, offline_executor):
    calls, behavior = offline_executor
    c, items = batch_config(tmp_path), models.specifications()
    behavior["interrupt_after_pass"] = True
    with pytest.raises(KeyboardInterrupt):
        orchestrator.execute(c, items)
    failed = load_data(paths(c)["output"] / "batches/batch_report.json")
    behavior["interrupt_after_pass"] = False
    resumed = orchestrator.execute(c, items, failed["batch_id"])
    assert resumed["status"] == "PASSED"
    assert [call["seed"] for call in calls] == [101, 102, 103]


@pytest.mark.parametrize("damage", ["image", "video", "missing_state", "config", "models"])
def test_batch_resume_refuses_changed_evidence_or_settings(tmp_path, offline_executor, damage):
    calls, behavior = offline_executor
    c, items = batch_config(tmp_path), models.specifications()
    behavior["fail_seed"] = 102
    with pytest.raises(RuntimeError):
        orchestrator.execute(c, items)
    failed = load_data(paths(c)["output"] / "batches/batch_report.json")
    first = failed["tasks"][0]
    if damage in ("image", "video"):
        Path(first["result"][damage]["path"]).write_bytes(b"corrupted fixture")
    elif damage == "missing_state":
        (paths(c)["output"] / "reports" / first["run_id"] / "state.json").unlink()
    elif damage == "config":
        c["tasks"][0]["image"]["prompt"] = "changed prompt"
    else:
        items[0]["revision"] = "b" * 40
    behavior["fail_seed"] = None
    with pytest.raises((ValueError, RuntimeError)):
        orchestrator.execute(c, items, failed["batch_id"])
    assert len(calls) == 2  # Never regenerate a completed task or start later work silently.


def test_cli_runtime_dispatches_batch_and_regenerates_report(tmp_path, offline_executor):
    sys.path.insert(0, str(ROOT / "scripts"))
    import cli
    c, items = batch_config(tmp_path), models.specifications()
    # This runtime dispatch uses an explicit offline executor; no CUDA/weights are tested.
    from unittest.mock import patch
    with patch.object(models, "verify_all", return_value=None):
        cli.runtime(cli.parser().parse_args(["--stage", "e2e"]), c, items)
    state = load_data(paths(c)["output"] / "batches/batch_report.json")
    report = batch.batch_directory(c, state["batch_id"]) / "batch_report.md"
    report.unlink()
    cli.runtime(cli.parser().parse_args(["--stage", "report", "--resume", state["batch_id"]]), c, items)
    assert report.exists() and state["batch_id"] in report.read_text(encoding="utf-8")
