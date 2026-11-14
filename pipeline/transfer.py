"""Move verified model bytes between independent instances, excluding runtime state."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import uuid

from .common import atomic_json, fingerprint, paths
from .models import portable_path, target, valid, verify_all


def member_name(item):
    return f"models/{item['directory']}/{Path(item['file']).name}"


def export_models(c, items, output):
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"Bundle already exists; choose a new filename: {output}")
    print("[START] Validate all model bytes before export", flush=True)
    verify_all(c, items)
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(m["size"] for m in items) + 512*1024**2
    if shutil.disk_usage(output.parent).free < required:
        raise RuntimeError("Insufficient disk for the additional uncompressed model bundle")
    temporary = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    try:
        with tarfile.open(temporary, "w", dereference=True) as archive:
            data = json.dumps({"format_version": 1, "models": items}, ensure_ascii=False).encode("utf-8")
            header = tarfile.TarInfo("manifest.json")
            header.size = len(data)
            archive.addfile(header, io.BytesIO(data))
            for item in items:
                print(f"[INFO] Export {item['key']}: {item['size']} bytes", flush=True)
                archive.add(target(c, item).resolve(), arcname=member_name(item), recursive=False)
        # Exclusive link prevents a simultaneous external writer from being overwritten.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[PASS] Portable model bundle: {output}; excludes venv, credentials, PID and installation state", flush=True)


def import_models(c, items, source):
    source = Path(source).expanduser().absolute()
    print(f"[INFO] Import bundle: {source}", flush=True)
    expected = {member_name(m): m for m in items}
    with tarfile.open(source, "r:") as archive:
        # Iterate with a strict entry budget; never use extractall on a supplied archive.
        members = {}
        for member in archive:
            if len(members) >= len(expected) + 1 or member.name in members or not member.isfile():
                raise RuntimeError("Bundle has duplicate, extra or non-regular entries")
            if member.name not in expected and member.name != "manifest.json":
                raise RuntimeError(f"Unexpected archive path: {member.name}")
            if member.name == "manifest.json" and member.size > 1024*1024:
                raise RuntimeError("Bundle manifest too large")
            members[member.name] = member
        if set(members) != {"manifest.json", *expected}:
            raise RuntimeError("Bundle is incomplete")
        manifest = json.load(archive.extractfile(members["manifest.json"]))
        if manifest.get("format_version") != 1 or fingerprint(manifest.get("models")) != fingerprint(items):
            raise RuntimeError("Bundle manifest does not match the configured model revisions/hashes")
        for name, item in expected.items():
            if members[name].size != item["size"]:
                raise RuntimeError(f"Bundle file size differs: {item['key']}")
        manifest_out = []
        for name, item in expected.items():
            dest = target(c, item)
            if valid(dest, item):
                print(f"[PASS] {item['key']} existing destination verified; reused", flush=True)
            else:
                if dest.exists() or dest.is_symlink():
                    raise RuntimeError(f"Existing invalid model preserved: {dest}")
                cached = portable_path(c, item)
                if not valid(cached, item):
                    if cached.exists() or cached.is_symlink():
                        raise RuntimeError(f"Existing invalid portable cache preserved: {cached}")
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    if shutil.disk_usage(cached.parent).free < item["size"] + 64*1024**2:
                        raise RuntimeError(f"Insufficient disk to import {item['key']}")
                    temporary = cached.with_name(cached.name + ".partial-" + uuid.uuid4().hex)
                    try:
                        print(f"[INFO] Import {item['key']}; validate {item['size']} bytes and SHA256", flush=True)
                        digest, count = hashlib.sha256(), 0
                        with archive.extractfile(members[name]) as stream, temporary.open("xb") as out:
                            for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                                out.write(chunk)
                                digest.update(chunk)
                                count += len(chunk)
                        if count != item["size"] or digest.hexdigest() != item["sha256"]:
                            raise RuntimeError(f"Bundle SHA256 mismatch: {item['key']}")
                        os.link(temporary, cached)
                    finally:
                        temporary.unlink(missing_ok=True)
                print(f"[PASS] {item['key']} imported to portable cache: {cached}; deployment follows installation", flush=True)
            manifest_out.append({**item, "target": str(dest), "verified_sha256": item["sha256"], "deployed": valid(dest, item)})
    atomic_json(paths(c)["state"] / "model-manifest.json", manifest_out)
    print("[PASS] Model import complete; GPU inference NOT RUN", flush=True)
