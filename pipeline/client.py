import time
import uuid


class ComfyClient:
    def __init__(self, c):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.base = f"http://127.0.0.1:{c['port']}"
        self.client_id = str(uuid.uuid4())
        self.timeout = c["task_timeout"]
        self.interval = c["poll_interval"]

    def request(self, method, route, **kwargs):
        response = self.session.request(method, self.base + route, timeout=(5, 30), **kwargs)
        if not response.ok:
            raise RuntimeError(f"ComfyUI {route}: HTTP {response.status_code}: {response.text[:3000]}")
        return response.json() if response.content.strip() else {}

    def info(self):
        return self.request("GET", "/object_info")

    def require_idle(self):
        queue = self.request("GET", "/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("ComfyUI has other queued/running work; retry when idle")

    def submit(self, graph):
        print(f"[INFO] Submitting /prompt: {len(graph)} nodes; client_id={self.client_id}", flush=True)
        data = self.request("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id})
        if data.get("error") or data.get("node_errors") or not data.get("prompt_id"):
            raise RuntimeError(f"Prompt validation failed: {data}")
        print(f"[INFO] Task accepted: prompt_id={data['prompt_id']}", flush=True)
        return data["prompt_id"]

    def cancel(self, prompt_id):
        queue = self.request("GET", "/queue")
        if any(item[1] == prompt_id for item in queue.get("queue_pending", [])):
            self.request("POST", "/queue", json={"delete": [prompt_id]})
        running = queue.get("queue_running", [])
        if len(running) == 1 and running[0][1] == prompt_id:
            self.request("POST", "/interrupt", json={"prompt_id": prompt_id})
            return "interrupt_requested"
        return "not_running_or_removed_from_queue"

    def wait(self, prompt_id):
        deadline = time.monotonic() + self.timeout
        next_progress = 0
        started = time.monotonic()
        while time.monotonic() < deadline:
            if time.monotonic() >= next_progress:
                print(f"[WAIT] prompt_id={prompt_id}; elapsed={time.monotonic()-started:.1f}s; timeout={self.timeout}s", flush=True)
                next_progress = time.monotonic() + 30
            response = self.request("GET", f"/history/{prompt_id}")
            if prompt_id in response:
                entry = response[prompt_id]
                status = entry.get("status", {})
                messages = status.get("messages", [])
                errors = [data for event, data in messages if event in ("execution_error", "execution_interrupted")]
                if status.get("status_str") == "error" or errors:
                    raise RuntimeError(f"Task failed: {errors or messages}")
                if status.get("completed"):
                    cached = [data for event, data in messages if event == "execution_cached" and data.get("nodes")]
                    if cached:
                        raise RuntimeError("Cached execution is not accepted as fresh GPU inference")
                    print(f"[PASS] Task completed: prompt_id={prompt_id}; elapsed={time.monotonic()-started:.1f}s", flush=True)
                    return entry
            time.sleep(min(self.interval, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"Task {prompt_id} exceeded {self.timeout}s")

    def upload(self, path, subfolder):
        with path.open("rb") as stream:
            data = self.request("POST", "/upload/image", files={"image": (path.name, stream, "image/png")},
                                data={"type": "input", "subfolder": subfolder, "overwrite": "false"})
        if data.get("type") != "input":
            raise RuntimeError("Upload did not return an input file")
        return data

    def free(self):
        self.require_idle()
        self.request("POST", "/free", json={"unload_models": True, "free_memory": True})
