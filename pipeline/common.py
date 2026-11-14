from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def load_data(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path escapes managed directory: {relative}")
    return path


def run(command, **kwargs):
    command = [str(x) for x in command]
    log_path = os.environ.get("PIPELINE_LOG_PATH")
    if log_path and not kwargs.get("capture_output") and "stdout" not in kwargs:
        started = time.monotonic()
        print(f"[COMMAND] {shlex.join(command)}", flush=True)
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait(timeout=30)
            raise
        print(f"[{'PASS' if code == 0 else 'FAIL'}] Command exit={code}, elapsed={time.monotonic()-started:.3f}s", flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    return subprocess.run(command, check=True, text=True, **kwargs)


def capture(command, quiet=False):
    logging = bool(os.environ.get("PIPELINE_LOG_PATH")) and not quiet
    if logging:
        print(f"[COMMAND] {shlex.join([str(x) for x in command])}", flush=True)
    try:
        value = run(command, capture_output=True, timeout=30).stdout.strip()
    except subprocess.CalledProcessError as exc:
        if logging:
            print(f"[FAIL] Command exit={exc.returncode}: {exc.stdout or ''} {exc.stderr or ''}", flush=True)
        raise
    if logging:
        print(f"[PASS] Command output: {value or '(empty)'}", flush=True)
    return value


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def config(path=None, overrides=(), use_env=True):
    c = load_data(ROOT / "config/defaults.yaml")
    if path:
        supplied = load_data(path)
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be an object; put batch jobs in a tasks array")
        for key, value in supplied.items():
            if key in ("image", "video") and isinstance(value, dict):
                c[key].update(value)
            else:
                c[key] = value
    env = {"WORKSPACE_ROOT": "workspace_root", "COMFYUI_DIR": "comfyui_dir",
           "MODEL_CACHE_DIR": "model_cache_dir", "COMFYUI_REF": "comfyui_ref",
           "COMFYUI_PORT": "port"}
    for name, key in env.items():
        if use_env and os.environ.get(name):
            c[key] = int(os.environ[name]) if key == "port" else os.environ[name]
    for item in overrides:
        key, value = item.split("=", 1)
        cursor = c
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"Unknown setting: {key}")
        try:
            cursor[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            cursor[parts[-1]] = value
    c["workspace_root"] = str(Path(c["workspace_root"]).expanduser().resolve())
    base = Path(c["workspace_root"])
    c.setdefault("comfyui_dir", str(base / "ComfyUI"))
    c.setdefault("model_cache_dir", str(base / "model-cache"))
    for key in ("comfyui_dir", "model_cache_dir"):
        c[key] = str(Path(c[key]).expanduser().resolve())
    if not re.fullmatch(r"[0-9a-f]{40}", c["comfyui_ref"]):
        raise ValueError("COMFYUI_REF must be an immutable 40-character commit")
    if c.get("git_http_version", "HTTP/1.1") not in ("HTTP/1.1", "HTTP/2"):
        raise ValueError("git_http_version must be HTTP/1.1 or HTTP/2")
    index = urlsplit(c.get("pypi_index_url", "https://mirrors.aliyun.com/pypi/simple"))
    if index.scheme != "https" or not index.hostname or index.username or index.password or index.query or index.fragment:
        raise ValueError("pypi_index_url must be an HTTPS index without embedded credentials or query parameters")
    for key, minimum, maximum in (("pip_timeout", 1, 600), ("pip_retries", 0, 10)):
        value = c.get(key, 30 if key == "pip_timeout" else 3)
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be an integer in {minimum}..{maximum}")
    validate_generation(c)
    if "tasks" in c:
        task_configs(c)  # Validate every effective task before installation/inference.
    if not 1 <= c["port"] <= 65535:
        raise ValueError("Invalid port")
    for key in ("startup_timeout", "task_timeout", "poll_interval", "download_attempts"):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return c


def validate_generation(c):
    for section, multiple in (("image", 8), ("video", 32)):
        s = c[section]
        if not isinstance(s, dict):
            raise ValueError(f"{section} must be an object")
        for key in ("prompt", "negative", "sampler", "scheduler"):
            if not isinstance(s[key], str) or (key != "negative" and not s[key].strip()):
                raise ValueError(f"Invalid {section}.{key}")
        for key in ("width", "height"):
            if not isinstance(s[key], int) or not 32 <= s[key] <= 8192 or s[key] % multiple:
                raise ValueError(f"{section}.{key} must be a multiple of {multiple}, 32..8192")
        if not isinstance(s["steps"], int) or not 1 <= s["steps"] <= 10000:
            raise ValueError(f"Invalid {section}.steps")
        if not isinstance(s["seed"], int) or not 0 <= s["seed"] < 2**64:
            raise ValueError(f"Invalid {section}.seed")
        if not 0 <= s["cfg"] <= 100:
            raise ValueError(f"Invalid {section}.cfg")
    if not isinstance(c["video"]["frames"], int) or c["video"]["frames"] < 5 or (c["video"]["frames"] - 1) % 4:
        raise ValueError("video.frames must be 4n+1 and >=5")
    if not 1 <= c["video"]["fps"] <= 120:
        raise ValueError("Invalid FPS")


def task_configs(c):
    """Resolve partial task settings without modifying the caller's configuration."""
    tasks = c.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty array")
    names, resolved = set(), []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or task.keys() - {"name", "image", "video"}:
            raise ValueError(f"Invalid tasks[{index}]: only name, image and video are supported")
        name = task.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError(f"Invalid tasks[{index}].name: use 1-64 ASCII letters/digits, _ or -")
        if name in names:
            raise ValueError(f"Duplicate task name: {name}")
        names.add(name)
        effective = deepcopy({key: value for key, value in c.items() if key != "tasks"})
        for section in ("image", "video"):
            overrides = task.get(section, {})
            if not isinstance(overrides, dict) or overrides.keys() - c[section].keys():
                raise ValueError(f"Invalid task {name}.{section} settings")
            effective[section].update(overrides)
        try:
            validate_generation(effective)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid task {name}: {exc}") from exc
        resolved.append((name, effective))
    return resolved


def runtime_tag():
    return f"py{sys.version_info[0]}.{sys.version_info[1]}"


def paths(c):
    base = Path(c["workspace_root"])
    tag = runtime_tag()
    venv = base / "venvs" / tag
    return {"base": base, "comfy": Path(c["comfyui_dir"]),
            "cache": Path(c["model_cache_dir"]), "venv": venv,
            "python": venv / "bin/python", "state": base / "state",
            "install_marker": base / "state/environments" / f"{tag}.json",
            "freeze": base / "state/environments" / f"{tag}-pip-freeze.txt",
            "logs": base / "logs", "output": base / "output"}


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another pipeline operation is running") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another pipeline operation is running") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
