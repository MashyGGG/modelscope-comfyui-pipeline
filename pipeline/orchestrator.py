from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from .client import ComfyClient
from .common import atomic_json, capture, checked_path, fingerprint, load_data, paths, sha256
from . import media, preflight, server, workflows
from .report import report


def now():
    return datetime.now(timezone.utc).isoformat()


class MemoryMonitor:
    def __init__(self, pid):
        self.pid, self.peak, self.samples = pid, None, 0
        self.end = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.end.is_set():
            try:
                text = capture(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], quiet=True)
                for line in text.splitlines():
                    pid, value = line.split(",")
                    if int(pid.strip()) == self.pid and value.strip().isdigit():
                        self.peak = max(self.peak or 0, int(value.strip()))
                        self.samples += 1
            except Exception:
                pass
            self.end.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.end.set()
        self.thread.join(timeout=35)


def resume_state(c, items, run_id, run_dir):
    state = load_data(run_dir / "state.json")
    if state["config_hash"] != fingerprint(c) or state["models_hash"] != fingerprint(items):
        raise RuntimeError("Resume configuration/models differ from original run")
    if state["status"] == "PASSED":
        raise RuntimeError("Run already passed; start a new run to perform fresh GPU inference")
    return state


def execute(c, items, resume=None):
    p = paths(c)
    run_id = resume or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    run_dir = p["output"] / "reports" / run_id
    print(f"[INFO] E2E run_id={run_id}; evidence directory={run_dir}", flush=True)
    if resume:
        state = resume_state(c, items, run_id, run_dir)
        # Retain earlier failure evidence on each resumed attempt.
        attempt_dir = run_dir / f"attempt-{time.time_ns()}"
        attempt_dir.mkdir()
        for evidence in run_dir.iterdir():
            if evidence.is_file():
                shutil.copy2(evidence, attempt_dir / evidence.name)
        report(state, attempt_dir, p["output"])
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {"run_id": run_id, "started_at": now(), "status": "RUNNING", "config": c,
                 "config_hash": fingerprint(c), "models": items, "models_hash": fingerprint(items)}
    client, monitor = None, None
    active_prompt = None
    def save():
        state["updated_at"] = now()
        atomic_json(run_dir / "state.json", state)
    save()
    try:
        state["status"] = "RUNNING"
        state.pop("error", None)
        state["environment"] = preflight.host(c, check_space=False)
        managed = server.owned(c)
        if not managed or not server.healthy(c):
            raise RuntimeError("A healthy managed server is required")
        state["server"] = managed
        state["cuda_runtime"] = preflight.cuda()
        client = ComfyClient(c)
        client.require_idle()
        info = client.info()
        atomic_json(run_dir / "object_info.json", info)
        atomic_json(run_dir / "model-manifest.json", items)
        monitor = MemoryMonitor(managed["pid"])
        with monitor:
            a, _ = workflows.build(c, items, run_id)
            image_stage = state.get("image", {})
            if image_stage.get("status") == "PASSED":
                artifact = image_stage["artifact"]
                image_path = checked_path(p["output"], Path(artifact["path"]).relative_to(p["output"]))
                if sha256(image_path) != artifact["sha256"]:
                    raise RuntimeError("Saved image changed; cannot resume")
                media.png(image_path, c["image"])
                print("[PASS] Text-to-Image (validated from original run)", flush=True)
            else:
                workflows.validate(a, info)
                atomic_json(run_dir / "image-prompt.json", a)
                client.require_idle()
                started = time.monotonic()
                active_prompt = client.submit(a)
                state["image"] = {"status": "RUNNING", "prompt_id": active_prompt}
                save()
                history = client.wait(active_prompt)
                active_prompt = None
                atomic_json(run_dir / "image-history.json", history)
                image_path = media.history_file(history, "7", p["output"])
                state["image"].update(status="PASSED", seconds=time.monotonic()-started,
                                      artifact=media.png(image_path, c["image"]))
                save()
                print("[PASS] Text-to-Image", flush=True)
            print(f"[INFO] PNG validated: {image_path}; sha256={state['image']['artifact']['sha256']}", flush=True)
            print("[START] Unload SDXL and free GPU memory", flush=True)
            client.free()
            print("[PASS] Model unload requested; uploading the generated PNG", flush=True)
            upload = client.upload(image_path, f"pipeline/{run_id}")
            uploaded_relative = str(Path(upload.get("subfolder", "")) / upload["name"]).replace("\\", "/")
            uploaded_path = checked_path(p["comfy"] / "input", uploaded_relative)
            if sha256(uploaded_path) != state["image"]["artifact"]["sha256"]:
                raise RuntimeError("Uploaded PNG SHA256 does not match generated PNG")
            state["input_image"] = {"path": str(uploaded_path), "sha256": sha256(uploaded_path)}
            print(f"[PASS] Uploaded PNG integrity: {uploaded_path}; sha256={state['input_image']['sha256']}", flush=True)
            _, b = workflows.build(c, items, run_id, uploaded_relative)
            workflows.validate(b, client.info(), input_dir=p["comfy"] / "input")
            atomic_json(run_dir / "video-prompt.json", b)
            client.require_idle()
            started = time.monotonic()
            active_prompt = client.submit(b)
            state["video"] = {"status": "RUNNING", "prompt_id": active_prompt}
            save()
            history = client.wait(active_prompt)
            active_prompt = None
            atomic_json(run_dir / "video-history.json", history)
            video_path = media.history_file(history, "12", p["output"])
            print(f"[START] MP4 full decode and metadata validation: {video_path}", flush=True)
            state["video"].update(status="PASSED", seconds=time.monotonic()-started,
                                  artifact=media.video(video_path, c["video"]))
            print("[PASS] Image-to-Video\n[PASS] MP4 Validation", flush=True)
        state["status"] = "PASSED"
        save()
        report(state, run_dir, p["output"])
        print(f"\nE2E VALIDATION PASSED\nImage: {image_path}\nVideo: {video_path}\nReport: {run_dir / 'e2e_report.md'}\nComfyUI: http://127.0.0.1:{c['port']}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "FAILED"
        state["error"] = str(exc) or type(exc).__name__
        if active_prompt and client:
            try:
                state["cancellation"] = client.cancel(active_prompt)
                partial = client.request("GET", f"/history/{active_prompt}")
                atomic_json(run_dir / "failed-history.json", partial)
            except Exception as cancel_error:
                state["cancellation"] = f"Cancellation failed: {cancel_error}"
        for stage in ("image", "video"):
            if state.get(stage, {}).get("status") == "RUNNING":
                state[stage]["status"] = "FAILED"
        save()
        report(state, run_dir, p["output"])
        print(f"[FAIL] E2E: {state['error']}\nReport: {run_dir / 'e2e_report.md'}", flush=True)
        print(f"Resume: bash oneclick.sh --resume {run_id}", flush=True)
        if "memory" in state["error"].lower() or "oom" in state["error"].lower():
            print("[HINT] New run: --set video.width=640 --set video.height=352 --set video.frames=25 --set vae_tiled=true", flush=True)
        raise
    finally:
        if monitor:
            state["gpu_memory_observed_peak_mib"] = monitor.peak
            state["gpu_memory_samples"] = monitor.samples
        save()
        report(state, run_dir, p["output"])
