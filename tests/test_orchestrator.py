from pathlib import Path
import time

import av
from PIL import Image
import pytest

from pipeline.client import ComfyClient
from pipeline.common import atomic_json, config, fingerprint, load_data, sha256
from pipeline import media, models, orchestrator, server


def encode_test_video(path, frames=5, fps=16):
    """Synthetic media fixture. This is never GPU inference or GPU acceptance evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 32), (i * 20, 60, 90)))
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def test_video_full_decode_and_parameter_rejection(tmp_path):
    path = tmp_path / "fixture.mp4"
    encode_test_video(path)
    expected = {"width": 64, "height": 32, "fps": 16, "frames": 5}
    result = media.video(path, expected)
    assert result["fully_decoded"] and result["frames"] == 5
    assert result["duration_seconds"] == pytest.approx(5/16)
    for key, wrong in (("frames", 9), ("fps", 24), ("width", 128)):
        with pytest.raises(RuntimeError):
            media.video(path, {**expected, key: wrong})
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises((RuntimeError, av.error.FFmpegError)):
        media.video(path, expected)


def test_save_video_history_uses_images_and_rejects_escape(tmp_path):
    (tmp_path / "real.mp4").write_bytes(b"fixture")
    entry = {"outputs": {"12": {"images": [{"filename": "real.mp4", "subfolder": "", "type": "output"}]}}}
    assert media.history_file(entry, "12", tmp_path) == tmp_path / "real.mp4"
    entry["outputs"]["12"]["images"][0]["filename"] = "../outside.mp4"
    with pytest.raises(RuntimeError):
        media.history_file(entry, "12", tmp_path)


def test_client_execution_error_and_cache_rejected(monkeypatch):
    client = ComfyClient(config())
    for status in (
        {"status_str": "error", "messages": [["execution_error", {"exception_message": "OOM"}]]},
        {"completed": True, "messages": [["execution_cached", {"nodes": ["5"]}]]},
    ):
        monkeypatch.setattr(client, "request", lambda *a, **kw: {"p": {"status": status}})
        with pytest.raises(RuntimeError):
            client.wait("p")


def test_timeout_is_bounded(monkeypatch):
    client = ComfyClient(config(overrides=["task_timeout=0.02", "poll_interval=0.005"]))
    monkeypatch.setattr(client, "request", lambda *a, **kw: {})
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client.wait("missing")
    assert time.monotonic() - started < 1


def test_cancel_does_not_interrupt_other_prompt(monkeypatch):
    client = ComfyClient(config())
    calls = []
    def request(method, route, **kw):
        calls.append((route, kw))
        return {"queue_running": [[0, "someone-else"]], "queue_pending": [[1, "ours"]]}
    monkeypatch.setattr(client, "request", request)
    client.cancel("ours")
    assert not any(route == "/interrupt" for route, _ in calls)
    assert ("/queue", {"json": {"delete": ["ours"]}}) in calls


def test_invalid_model_is_preserved_and_skip_cannot_hide_it(tmp_path):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    m = {"key": "test", "directory": "vae", "file": "model.bin", "size": 4, "sha256": "0" * 64}
    path = models.target(c, m)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bad!")
    for skip in (False, True):
        with pytest.raises(RuntimeError):
            models.download(c, [m], skip=skip)
    assert path.read_bytes() == b"bad!"


def test_existing_model_needs_no_network(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    m = {"key": "test", "directory": "vae", "file": "model.bin", "size": 4}
    path = models.target(c, m)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"good")
    m["sha256"] = sha256(path)
    monkeypatch.setattr(models, "remote_metadata", lambda *_: pytest.fail("unexpected network"))
    models.download(c, [m])
    assert load_data(tmp_path / "state/model-manifest.json")[0]["verified_sha256"] == m["sha256"]


def test_stop_refuses_recycled_pid(tmp_path, monkeypatch):
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}"])
    atomic_json(tmp_path / "state/server.json", {"pid": 123, "identity": {"start_ticks": "old"}, "config_hash": server.server_key(c)})
    monkeypatch.setattr(server, "identity", lambda _: {"start_ticks": "new"})
    monkeypatch.setattr(server.os, "kill", lambda *_: pytest.fail("must not signal unrelated process"))
    with pytest.raises(RuntimeError, match="identity"):
        server.stop(c)


@pytest.mark.parametrize("valid_upload", [False, True])
def test_upload_validation_and_image_resume(tmp_path, monkeypatch, valid_upload):
    """Exercise failure persistence and stage reuse with explicit offline doubles."""
    c = config(overrides=[f"workspace_root={tmp_path.as_posix()}", "image.width=64", "image.height=32"])
    items = models.specifications()
    monkeypatch.setattr(orchestrator.preflight, "host", lambda *a, **kw: {"gpu": {"name": "OFFLINE DOUBLE"}})
    monkeypatch.setattr(orchestrator.preflight, "cuda", lambda: {"torch": "OFFLINE DOUBLE"})
    monkeypatch.setattr(orchestrator.server, "owned", lambda _: {"pid": 123})
    monkeypatch.setattr(orchestrator.server, "healthy", lambda _: True)
    real_validate = orchestrator.workflows.validate
    def validate_image_input(graph, info=None, **kwargs):
        # Keep real file validation at the orchestration boundary; other nodes
        # are outside this offline test's scope. Never simulate GPU success.
        if info is None:
            real_validate(graph)
        elif graph["7"]["class_type"] == "LoadImage":
            real_validate({"7": graph["7"]}, info, **kwargs)
    monkeypatch.setattr(orchestrator.workflows, "validate", validate_image_input)
    class NoMonitor:
        peak, samples = None, 0
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(orchestrator, "MemoryMonitor", NoMonitor)
    counts = {"image": 0, "video": 0}
    class FakeClient:
        def __init__(self, c): self.graph = None
        def require_idle(self): pass
        def info(self):
            return {"LoadImage": {"input": {"required": {
                "image": [[], {"image_upload": True}],
            }}, "output": ["IMAGE", "MASK"]}}
        def free(self): pass
        def submit(self, graph):
            if graph["7"]["class_type"] == "LoadImage":
                counts["video"] += 1
                raise RuntimeError("OFFLINE video submission reached")
            counts["image"] += 1
            self.graph = graph
            return "offline-prompt"
        def wait(self, prompt):
            prefix = self.graph["7"]["inputs"]["filename_prefix"]
            out = tmp_path / "output" / (prefix + ".png")
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 32)).save(out)
            return {"status": {"completed": True}, "outputs": {"7": {"images": [
                {"filename": out.name, "subfolder": str(out.parent.relative_to(tmp_path / "output")), "type": "output"}]}}}
        def upload(self, image, subfolder):
            target = tmp_path / "ComfyUI/input" / subfolder / "wrong.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image.read_bytes() if valid_upload else b"different")
            return {"type": "input", "subfolder": subfolder, "name": "wrong.png"}
    monkeypatch.setattr(orchestrator, "ComfyClient", FakeClient)
    expected_error = "OFFLINE video submission reached" if valid_upload else "Uploaded PNG SHA256"
    with pytest.raises(RuntimeError, match=expected_error):
        orchestrator.execute(c, items)
    state = load_data(tmp_path / "output/reports/e2e_report.json")
    assert state["status"] == "FAILED" and state["image"]["status"] == "PASSED"
    assert "video" not in state
    with pytest.raises(RuntimeError, match=expected_error):
        orchestrator.execute(c, items, state["run_id"])
    assert counts == {"image": 1, "video": 2 if valid_upload else 0}
    assert list((tmp_path / "output/reports" / state["run_id"]).glob("attempt-*/e2e_report.json"))
    assert list((tmp_path / "output/reports" / state["run_id"]).glob("attempt-*/image-history.json"))
    altered = config(overrides=[f"workspace_root={tmp_path.as_posix()}", "image.width=64", "image.height=32", "video.seed=42"])
    with pytest.raises(RuntimeError, match="configuration/models"):
        orchestrator.execute(altered, items, state["run_id"])
