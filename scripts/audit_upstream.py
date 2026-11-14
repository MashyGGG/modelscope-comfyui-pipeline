#!/usr/bin/env python3
"""Read-only verification of immutable upstream references; no weight downloads."""
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import config
from pipeline.models import remote_metadata, specifications


def main():
    c = config()
    for item in specifications():
        remote_metadata(item)
        print(f"[PASS] {item['key']}: immutable revision, path, size and SHA256", flush=True)
    base = f"https://raw.githubusercontent.com/Comfy-Org/ComfyUI/{c['comfyui_ref']}/"
    checks = {
        "comfy_extras/nodes_wan.py": ["class Wan22ImageToVideoLatent"],
        "comfy_extras/nodes_video.py": ["class SaveVideo", "class CreateVideo", "ui.PreviewVideo"],
        "comfy/cli_args.py": ["--cache-none", "--output-directory", "--disable-smart-memory"],
        "comfy_api/latest/_ui.py": ['"images": self.values'],
        "server.py": ["prompt_id = json_data.get('prompt_id')"],
    }
    for path, terms in checks.items():
        with urllib.request.urlopen(base + path, timeout=30) as r:
            text = r.read().decode()
        if any(term not in text for term in terms):
            raise RuntimeError(f"Pinned upstream interface mismatch: {path}")
        print(f"[PASS] {path}", flush=True)
    print("GPU integration: NOT RUN")


if __name__ == "__main__":
    main()
