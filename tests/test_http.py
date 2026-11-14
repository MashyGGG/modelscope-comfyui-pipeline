import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline.client import ComfyClient
from pipeline.common import config


def test_real_http_empty_post_response_and_targeted_interrupt():
    """Local API contract fixture, without a ComfyUI/GPU server."""
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            payload = {"queue_running": [[0, "ours"]], "queue_pending": []}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            calls.append((self.path, payload))
            self.send_response(200)
            self.end_headers()  # ComfyUI /free and /interrupt return no JSON body.
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        client = ComfyClient(config(overrides=[f"port={http.server_port}"]))
        assert client.request("POST", "/free", json={"unload_models": True}) == {}
        assert client.cancel("ours") == "interrupt_requested"
        assert ("/interrupt", {"prompt_id": "ours"}) in calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join()
