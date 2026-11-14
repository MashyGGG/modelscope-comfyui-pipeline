"""Sequential GPU tasks with independent evidence and resumable batch state."""
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import time
import uuid

from .common import atomic_json, checked_path, fingerprint, load_data, paths, sha256, task_configs
from .logging import Operation
from . import media


def now():
    return datetime.now(timezone.utc).isoformat()


def batch_directory(c, batch_id):
    if not re.fullmatch(r"batch-[A-Za-z0-9_-]+", batch_id):
        raise ValueError("Invalid batch ID")
    return checked_path(paths(c)["output"] / "batches", batch_id)


def task_run_id(batch_id, index, name):
    return f"{batch_id}-{index + 1:03d}-{name}"


def load_batch(c, items, batch_id, *, allow_passed=False):
    directory = batch_directory(c, batch_id)
    state = load_data(directory / "state.json")
    if (state.get("batch_id") != batch_id or state.get("format_version") != 1 or
            state.get("config_hash") != fingerprint(c) or state.get("models_hash") != fingerprint(items)):
        raise RuntimeError("Batch resume configuration/models differ from original batch")
    expected = [(name, task_run_id(batch_id, i, name)) for i, (name, _) in enumerate(task_configs(c))]
    if [(entry.get("name"), entry.get("run_id")) for entry in state["tasks"]] != expected:
        raise RuntimeError("Batch task identities differ from original plan")
    if state["status"] == "PASSED" and not allow_passed:
        raise RuntimeError("Batch already passed; omit --resume to start a new batch")
    return state, directory


def batch_report(state, directory, output):
    directory, output = Path(directory), Path(output)
    lines = ["# Batch execution report", "", f"- Batch ID: `{state['batch_id']}`",
             f"- Status: **{state['status']}**", f"- Updated (UTC): {state['updated_at']}",
             f"- Passed: {sum(t['status'] == 'PASSED' for t in state['tasks'])}/{len(state['tasks'])}",
             "", "| Task | Status | Run ID | Image | Video | Report |", "|---|---|---|---|---|---|"]
    for entry in state["tasks"]:
        result = entry.get("result", {})
        lines.append(f"| {entry['name']} | {entry['status']} | `{entry['run_id']}` | "
                     f"`{result.get('image', {}).get('path', '')}` | `{result.get('video', {}).get('path', '')}` | "
                     f"`{entry.get('report', '')}` |")
    if state.get("error"):
        lines += ["", "Failure:", "```text", state["error"], "```"]
    lines += ["", "Each task retains its own logs, state and E2E report. "
              "Offline fixtures do not count as GPU acceptance.", ""]
    content = "\n".join(lines)
    atomic_json(directory / "batch_report.json", state)
    (directory / "batch_report.md").write_text(content, encoding="utf-8")
    latest = output / "batches"
    latest.mkdir(parents=True, exist_ok=True)
    atomic_json(latest / "batch_report.json", state)
    temporary = latest / "batch_report.md.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(latest / "batch_report.md")


def verified_child(c, items, run_id):
    state = load_data(paths(c)["output"] / "reports" / run_id / "state.json")
    if (state.get("run_id") != run_id or state.get("config_hash") != fingerprint(c) or
            state.get("models_hash") != fingerprint(items)):
        raise RuntimeError(f"Task state configuration/models differ: {run_id}")
    return state


def passed_artifacts(c, state):
    """Recheck completed files before recording success or skipping a task."""
    if state.get("status") != "PASSED":
        raise RuntimeError("Task has no completed E2E result")
    result = {}
    for name, check in (("image", media.png), ("video", media.video)):
        stage = state.get(name, {})
        if stage.get("status") != "PASSED":
            raise RuntimeError(f"Task {name} has not passed")
        artifact = stage["artifact"]
        relative = Path(artifact["path"]).relative_to(paths(c)["output"])
        path = checked_path(paths(c)["output"], relative)
        if sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Completed task {name} changed: {path}")
        result[name] = check(path, c[name])
    return result


def execute_batch(c, items, resume=None):
    from .orchestrator import execute_single
    plan = task_configs(c)
    p = paths(c)
    if resume:
        state, directory = load_batch(c, items, resume)
        attempt = directory / f"attempt-{time.time_ns()}"
        attempt.mkdir()
        for evidence in directory.iterdir():
            if evidence.is_file():
                shutil.copy2(evidence, attempt / evidence.name)
    else:
        batch_id = "batch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        directory = batch_directory(c, batch_id)
        directory.mkdir(parents=True, exist_ok=False)
        state = {"format_version": 1, "batch_id": batch_id, "status": "RUNNING", "started_at": now(),
                 "config": c, "config_hash": fingerprint(c), "models": items, "models_hash": fingerprint(items),
                 "tasks": [{"name": name, "run_id": task_run_id(batch_id, i, name), "status": "PENDING",
                            "attempts": []} for i, (name, _) in enumerate(plan)]}
    batch_id = state["batch_id"]
    state.update(status="RUNNING")
    state.pop("error", None)

    def save():
        state["updated_at"] = now()
        atomic_json(directory / "state.json", state)
        batch_report(state, directory, p["output"])

    save()
    print(f"[INFO] Batch id={batch_id}; tasks={len(plan)}; report={directory / 'batch_report.md'}", flush=True)
    active = None
    started = time.monotonic()
    try:
        for index, (name, effective) in enumerate(plan):
            active = state["tasks"][index]
            run_id = active["run_id"]
            task_state_path = p["output"] / "reports" / run_id / "state.json"
            active["report"] = str(task_state_path.with_name("e2e_report.md"))
            operation = Operation(p["logs"], f"task-{name}", ["batch-task", batch_id, name],
                                  console_timestamps=False)
            active["attempts"].append({"operation_id": operation.id, "log": str(operation.log),
                                       "summary": str(operation.summary)})
            save()
            with operation:
                with operation.phase(f"Batch task {index + 1}/{len(plan)}: {name}; run_id={run_id}"):
                    child = verified_child(effective, items, run_id) if task_state_path.exists() else None
                    if child and child["status"] == "PASSED":
                        active["result"] = passed_artifacts(effective, child)
                        print(f"[SKIP] Task {name}: completed image/video hashes and media revalidated", flush=True)
                    else:
                        if not child and (active["status"] == "PASSED" or active.get("result")):
                            raise RuntimeError(f"Completed task state missing: {run_id}")
                        active.update(status="RUNNING")
                        active.pop("error", None)
                        save()
                        atomic_json(directory / "configs" / f"{name}.json", effective)
                        if child:
                            execute_single(effective, items, resume=run_id)
                        else:
                            execute_single(effective, items, run_id=run_id)
                        child = verified_child(effective, items, run_id)
                        active["result"] = passed_artifacts(effective, child)
                    active.update(status="PASSED", report=str(task_state_path.with_name("e2e_report.md")))
                    save()
            print(f"[PASS] Batch task {index + 1}/{len(plan)}: {name}", flush=True)
        state.update(status="PASSED")
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        if active:
            active.update(status="FAILED", error=error)
        state.update(status="FAILED", error=error)
        print(f"[FAIL] Batch {batch_id}: {error}", flush=True)
        print(f"[INFO] Keep the same --config and add --resume {batch_id}; "
              "completed tasks are revalidated and skipped", flush=True)
        raise
    finally:
        state["last_attempt_seconds"] = round(time.monotonic() - started, 3)
        save()
    print(f"BATCH VALIDATION PASSED\nBatch: {batch_id}\nReport: {directory / 'batch_report.md'}", flush=True)
    return state
