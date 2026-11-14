"""Standard-library diagnostics: collect all checks even when one fails."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

from .common import capture, paths


def diagnose(c, require_cuda=False, network=False, probe_torch=True, check_space=True, require_gpu=True):
    from .preflight import driver_status, gpu_info, port_open
    if require_cuda and not require_gpu:
        raise ValueError("CPU preparation cannot require CUDA validation")
    result = {"status": "RUNNING", "mode": "gpu" if require_gpu else "cpu-prepare",
              "checks": [], "python": sys.version, "system": platform.platform()}
    def check(name, action):
        started = time.monotonic()
        print(f"[CHECK] {name}", flush=True)
        try:
            status, detail, values = action()
        except Exception as exc:
            status, detail, values = "FAIL", f"{type(exc).__name__}: {exc}", {}
        record = {"name": name, "status": status, "detail": detail, "seconds": round(time.monotonic()-started, 3)}
        result["checks"].append(record)
        result.update(values)
        print(f"[{status}] {name}: {detail} ({record['seconds']:.3f}s)", flush=True)

    check("Operating system", lambda: (
        "PASS" if platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64") else "FAIL",
        f"{platform.system()}/{platform.machine()}; require Linux x86_64", {}))
    check("Python", lambda: ("PASS" if (3,10) <= sys.version_info[:2] <= (3,12) else "FAIL",
                              f"{sys.version.split()[0]} at {sys.executable}; require 3.10-3.12", {}))
    for tool in ("git", "bash", "nvidia-smi"):
        if tool == "nvidia-smi" and not require_gpu:
            check(tool, lambda: ("SKIP", "CPU preparation does not require NVIDIA tools", {}))
            continue
        check(tool, lambda tool=tool: ("PASS" if shutil.which(tool) else "FAIL", shutil.which(tool) or "not found", {}))
    check("Python venv support", lambda: (
        "PASS" if importlib.util.find_spec("venv") and importlib.util.find_spec("ensurepip") else "FAIL",
        "venv and ensurepip are required to create the managed environment", {}))
    def gpu():
        if not require_gpu:
            return "SKIP", "GPU hardware is not required for CPU preparation", {}
        if not shutil.which("nvidia-smi"):
            return "FAIL", "nvidia-smi unavailable; GPU/driver cannot be inspected", {}
        value = gpu_info(c)
        return "PASS", f"{value['name']}, {value['memory_total_mib']} MiB, driver={value['driver']}, uuid={value['uuid']}", {"gpu": value}
    check("GPU", gpu)
    def driver():
        if not require_gpu:
            return "SKIP", "Driver compatibility will be checked on the GPU instance", {}
        if "gpu" not in result:
            return "FAIL", "GPU detection failed; driver compatibility unknown", {}
        value = result["gpu"]["driver"]
        status, detail = driver_status(c, value)
        return status, detail, {}
    check("NVIDIA driver", driver)
    def vram():
        if not require_gpu:
            return "SKIP", "VRAM will be checked on the GPU instance", {}
        if "gpu" not in result:
            return "FAIL", "GPU detection failed; VRAM unknown", {}
        value = result["gpu"]["memory_total_mib"]
        return ("PASS" if value >= c["min_vram_mib"] else "WARN",
                f"{value} MiB; target A10 24GB; low VRAM may require lower resolution/frame count", {})
    check("GPU memory", vram)
    def memory():
        if not Path("/proc/meminfo").exists():
            return "FAIL", "/proc/meminfo unavailable on this platform", {}
        info = {line.split(":")[0]: int(line.split(":")[1].split()[0])*1024
                for line in Path("/proc/meminfo").read_text().splitlines()}
        available, total = info["MemAvailable"]/2**30, info["MemTotal"]/2**30
        return ("PASS" if available >= c["min_ram_gib"] else "WARN",
                f"available={available:.2f} GiB, total={total:.2f} GiB; recommended available >= {c['min_ram_gib']} GiB",
                {"ram_available_gib": available, "ram_total_gib": total})
    check("Host memory", memory)
    def disk():
        ancestor = paths(c)["base"]
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free/2**30
        return ("FAIL" if check_space and free < c["min_disk_gib"] else "PASS",
                f"{free:.2f} GiB free on {ancestor}; first-install threshold={c['min_disk_gib']} GiB; gate={'enabled' if check_space else 'disabled (already installed)'}",
                {"disk_free_gib": free})
    check("Workspace disk", disk)
    def writable():
        ancestor = paths(c)["base"]
        while not ancestor.exists():
            ancestor = ancestor.parent
        return ("PASS" if os.access(ancestor, os.W_OK | os.X_OK) else "FAIL",
                f"workspace={paths(c)['base']}; existing parent={ancestor}", {})
    check("Workspace permissions", writable)
    check("Managed Python environment", lambda: (
        "PASS", f"selected={paths(c)['venv']}; marker={paths(c)['install_marker']}; "
        "legacy venv and model/cache paths are preserved", {"managed_python": str(paths(c)["python"])}))
    def port():
        if not require_gpu:
            return "SKIP", "CPU preparation does not start a ComfyUI service", {}
        if not port_open(c["port"]):
            return "PASS", f"127.0.0.1:{c['port']} is free", {}
        from .server import owned
        managed = owned(c)
        return ("PASS" if managed else "FAIL",
                f"port {c['port']} is occupied by " + (f"managed PID {managed['pid']}" if managed else "another service"), {})
    check("ComfyUI port", port)
    if probe_torch:
        def torch_probe():
            managed = paths(c)["python"]
            if require_cuda and not managed.is_file():
                return "FAIL", "Managed runtime is missing; first run bash oneclick.sh --stage install", {}
            interpreter = str(managed if managed.is_file() else Path(sys.executable))
            code = """import importlib.util,json
if importlib.util.find_spec('torch') is None:
 print(json.dumps({'available':False,'detail':'torch not installed; expected before installation'}))
else:
 import torch
 ok=torch.cuda.is_available() if REQUIRE_GPU else False
 if ok:
  x=torch.ones((32,32),device='cuda'); assert (x@x)[0,0].item()==32; torch.cuda.synchronize()
 print(json.dumps({'available':ok,'torch':torch.__version__,'cuda':torch.version.cuda}))
""".replace("REQUIRE_GPU", repr(require_gpu))
            if require_cuda:
                from .common import ROOT
                code = ("import json,sys; sys.path.insert(0," + repr(str(ROOT)) + "); "
                        "from pipeline.preflight import cuda; print(json.dumps(cuda()))")
            proc = subprocess.run([interpreter, "-c", code], capture_output=True, text=True, timeout=60,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(c["gpu_index"])})
            if proc.returncode:
                return ("FAIL" if require_cuda else "WARN", f"CUDA probe failed: {(proc.stderr or proc.stdout)[-2000:]}", {})
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            supported = data.get("available") and data.get("torch", "").split("+")[0] == "2.8.0" and data.get("cuda") == "12.6"
            if not require_gpu:
                matched = data.get("torch", "").split("+")[0] == "2.8.0" and data.get("cuda") == "12.6"
                return ("PASS" if matched else "WARN", f"Import-only CPU preparation: {data}; GPU computation NOT RUN", {"cuda_probe": data})
            status = "PASS" if supported else ("FAIL" if require_cuda else "WARN")
            return status, f"python={interpreter}; {data}; managed runtime requires torch 2.8.0/cu126", {"cuda_probe": data}
        check("PyTorch CUDA computation" if require_gpu else "PyTorch package import", torch_probe)
    if network:
        for name, url in {
            "GitHub source": "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/v0.3.60/README.md",
            "ModelScope Hub": "https://modelscope.cn/api/v1/models/AI-ModelScope/stable-diffusion-xl-base-1.0/repo/files?Revision=master",
            "Ordinary PyPI index": c.get("pypi_index_url", "https://mirrors.aliyun.com/pypi/simple").rstrip("/") + "/pip/",
            "PyTorch cu126 index": "https://download.pytorch.org/whl/cu126/torch/",
        }.items():
            def probe(url=url):
                with urllib.request.urlopen(url, timeout=15) as response:
                    response.read(256)
                    return "PASS", f"HTTP {response.status} {url}; small probe, no weights downloaded", {}
            check(name, probe)
    else:
        print("[SKIP] Network probes disabled; add --network to check download endpoints", flush=True)
    failures = [x for x in result["checks"] if x["status"] == "FAIL"]
    result["status"] = "FAILED" if failures else "PASSED"
    print(f"[{'FAIL' if failures else 'PASS'}] Environment: {len(failures)} failed, "
          f"{sum(x['status']=='WARN' for x in result['checks'])} warnings", flush=True)
    return result


def require_passed(result):
    if result["status"] != "PASSED":
        errors = "; ".join(f"{x['name']}: {x['detail']}" for x in result["checks"] if x["status"] == "FAIL")
        raise RuntimeError("Environment checks failed: " + errors)
