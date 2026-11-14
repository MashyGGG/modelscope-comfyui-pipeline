import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import json

from .common import atomic_json, fingerprint, load_data, paths
from .preflight import port_open


def identity(pid):
    proc = Path("/proc") / str(pid)
    try:
        # Fields after the final ')' avoid spaces in process names.
        ticks = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        return {"start_ticks": ticks, "cmdline": proc.joinpath("cmdline").read_bytes().hex(),
                "uid": proc.stat().st_uid}
    except (OSError, IndexError):
        return None


def server_key(c):
    return fingerprint({**{k: c[k] for k in ("workspace_root", "comfyui_dir", "comfyui_ref", "port", "gpu_index")},
                        "python": str(paths(c)["python"])})


def owned(c, stopping=False):
    record = paths(c)["state"] / "server.json"
    if not record.exists():
        return None
    state = load_data(record)
    current = identity(state["pid"])
    if current is None:
        return None
    allowed = {server_key(c)}
    if stopping:
        base = {k: c[k] for k in ("workspace_root", "comfyui_dir", "comfyui_ref", "port", "gpu_index")}
        allowed.add(fingerprint(base))  # Legacy server record before Python-specific environments.
        if state.get("command"):
            allowed.add(fingerprint({**base, "python": state["command"][0]}))
    if state["config_hash"] not in allowed or current != state["identity"]:
        raise RuntimeError("Managed server identity/configuration mismatch; refusing to reuse or stop it")
    return state


def healthy(c):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{c['port']}/system_stats", timeout=3) as response:
            data = json.load(response)
        return bool(data.get("devices"))
    except (OSError, ValueError):
        return False


def start(c):
    p = paths(c)
    existing = owned(c)
    if existing:
        if healthy(c):
            print("[PASS] ComfyUI Server (reused; cache disabled)", flush=True)
            return existing
        raise RuntimeError("Managed server exists but is unhealthy; inspect logs, then --stop")
    if port_open(c["port"]):
        raise RuntimeError(f"Port {c['port']} belongs to another service; select another port")
    from .install import verify
    verify(c)
    p["logs"].mkdir(parents=True, exist_ok=True)
    p["output"].mkdir(parents=True, exist_ok=True)
    log_path = p["logs"] / f"comfyui-{time.time_ns()}.log"
    command = [str(p["python"]), "-u", str(p["comfy"] / "main.py"), "--listen", "127.0.0.1",
               "--port", str(c["port"]), "--output-directory", str(p["output"]),
               "--cache-none", "--disable-auto-launch", "--disable-smart-memory"]
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=p["comfy"], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": str(c["gpu_index"])})
    state = {"pid": child.pid, "identity": identity(child.pid), "config_hash": server_key(c),
             "log": str(log_path), "command": command}
    print(f"[INFO] ComfyUI PID={child.pid}; server log={log_path}; health timeout={c['startup_timeout']}s", flush=True)
    # exec() may not yet have replaced the parent's Python image; wait for the expected command line.
    deadline = time.monotonic() + c["startup_timeout"]
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with status {child.returncode}")
            current = identity(child.pid)
            if current and str(p["comfy"] / "main.py").encode().hex() in current["cmdline"]:
                state["identity"] = current
                atomic_json(p["state"] / "server.json", state)
                if healthy(c):
                    print("[PASS] ComfyUI Server", flush=True)
                    return state
            time.sleep(1)
        raise TimeoutError("ComfyUI startup timed out")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-100:]), flush=True)
        raise


def stop(c):
    state = owned(c, stopping=True)
    if not state:
        print("[PASS] No managed ComfyUI process is running", flush=True)
        return
    os.kill(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and identity(state["pid"]) == state["identity"]:
        # A zombie has no GPU context or running executable.
        if Path(f"/proc/{state['pid']}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.25)
    else:
        if identity(state["pid"]) == state["identity"]:
            raise RuntimeError("Managed process did not exit after SIGTERM; inspect it manually")
    print("[PASS] Managed ComfyUI stopped", flush=True)
