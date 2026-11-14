import os
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request
import json

from .common import ROOT, atomic_json, checked_path, load_data, paths, sha256


def specifications(path=None):
    items = load_data(path or ROOT / "config/models.yaml")["models"]
    if {x["key"] for x in items} != {"sdxl", "wan", "umt5", "vae"} or len(items) != 4:
        raise ValueError("Expected exactly SDXL, Wan, UMT5 and VAE")
    for m in items:
        if not re.fullmatch(r"[0-9a-f]{40}", m["revision"]) or not re.fullmatch(r"[0-9a-f]{64}", m["sha256"]):
            raise ValueError("Model revision and SHA256 must be immutable hashes")
        if m["size"] <= 0 or m["directory"] not in ("checkpoints", "diffusion_models", "text_encoders", "vae"):
            raise ValueError("Invalid model size/directory")
        if "sdk_revision" in m and (not isinstance(m["sdk_revision"], str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", m["sdk_revision"])):
            raise ValueError("sdk_revision must be an explicit branch/tag supported by ModelScope SDK")
        checked_path(ROOT, m["file"])
    return items


def valid(path, m):
    path = Path(path)
    return path.is_file() and path.stat().st_size == m["size"] and sha256(path) == m["sha256"]


def target(c, m):
    return paths(c)["comfy"] / "models" / m["directory"] / Path(m["file"]).name


def link_model(cached, dest):
    cached, dest = Path(cached).resolve(), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(os.path.relpath(cached, dest.parent))
    except OSError:
        if sys.platform != "win32":
            raise
        os.link(cached, dest)  # Supports local offline tests without Windows symlink privileges.


def portable_path(c, item):
    return paths(c)["cache"] / "portable" / item["sha256"] / Path(item["file"]).name


def verify_available(c, items):
    for item in items:
        print(f"[CHECK] Prepared model availability: {item['key']}", flush=True)
        if not valid(target(c, item), item) and not valid(portable_path(c, item), item):
            raise RuntimeError(f"Prepared model missing/incomplete: {item['key']}; share the prepared workspace or import a model bundle")


def verify_all(c, items):
    for m in items:
        if not valid(target(c, m), m):
            raise RuntimeError(f"Missing/incomplete model: {target(c, m)}")


def remote_metadata(m):
    query = urllib.parse.urlencode({"Revision": m["revision"], "Recursive": "true"})
    url = f"https://modelscope.cn/api/v1/models/{m['repo']}/repo/files?{query}"
    request = urllib.request.Request(url)
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    if data.get("Code") != 200:
        raise RuntimeError(f"ModelScope metadata request failed for {m['key']}")
    match = next((f for f in data["Data"]["Files"] if f["Path"] == m["file"]), None)
    if not match or match["Size"] != m["size"] or match.get("Sha256") != m["sha256"]:
        raise RuntimeError(f"Pinned remote metadata differs from configuration: {m['key']}")


def download(c, items, skip=False):
    p = paths(c)
    manifest = []
    for m in items:
        dest = target(c, m)
        print(f"[INFO] {m['key']} -> {dest} ({m['size']} bytes)", flush=True)
        if not dest.exists() and not dest.is_symlink() and valid(portable_path(c, m), m):
            link_model(portable_path(c, m), dest)
            print(f"[PASS] {m['key']} deployed from imported portable cache (no download)", flush=True)
        if valid(dest, m):
            print(f"[PASS] {m['key']} size/SHA256 (reused)", flush=True)
        elif skip:
            raise RuntimeError(f"--skip-download requires a complete validated file: {dest}")
        else:
            if dest.exists() or dest.is_symlink():
                raise RuntimeError(f"Existing invalid destination preserved; inspect or select a new directory: {dest}")
            cached = None
            sdk_revision = m.get("sdk_revision", m["revision"])
            print(f"[INFO] ModelScope repo={m['repo']}; pinned_revision={m['revision']}; "
                  f"sdk_revision={sdk_revision}; file={m['file']}", flush=True)
            for attempt in range(1, int(c["download_attempts"]) + 1):
                try:
                    remote_metadata(m)
                    if sdk_revision != m["revision"]:
                        # SDK 1.29.2 resolves branch/tag names, not arbitrary commit hashes.
                        # Both references must expose the exact pinned bytes; never silently change weights.
                        remote_metadata({**m, "revision": sdk_revision})
                        print(f"[PASS] {m['key']} pinned and SDK-reference metadata match size/SHA256", flush=True)
                    from modelscope.hub.file_download import model_file_download
                    from modelscope.hub.api import HubApi
                    cookies = None
                    token = os.environ.get("MODELSCOPE_API_TOKEN")
                    if token:
                        _, cookies = HubApi().login(token)
                    cached = Path(model_file_download(m["repo"], file_path=m["file"], revision=sdk_revision,
                                                      cache_dir=str(p["cache"]), cookies=cookies))
                    if not valid(cached, m):
                        raise RuntimeError(f"Downloaded file failed SHA256/size: {m['key']}")
                    break
                except Exception as exc:
                    # Suppress arbitrary SDK/network URLs; revision lookup errors need actionable detail.
                    print(f"[FAIL] {m['key']} attempt {attempt}: {type(exc).__name__}", flush=True)
                    if type(exc).__name__ == "NotExistError":
                        from .logging import sanitized
                        detail = re.sub(r"https?://\S+", "[URL REDACTED]", str(exc))
                        print(f"[DETAIL] {sanitized(detail[:1000])}", flush=True)
                        print("[HINT] Check sdk_revision (SDK branch/tag), repository and file path; pinned SHA256 remains mandatory", flush=True)
                    if attempt == int(c["download_attempts"]):
                        raise RuntimeError(f"Model download failed: {m['key']}; cache preserved for retry") from None
                    time.sleep(min(attempt * 2, 10))
            link_model(cached, dest)
            print(f"[PASS] {m['key']} size/SHA256", flush=True)
        manifest.append({**m, "target": str(dest), "verified_sha256": m["sha256"]})
    atomic_json(p["state"] / "model-manifest.json", manifest)
    print("[PASS] Model Downloads", flush=True)
