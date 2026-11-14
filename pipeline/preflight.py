import os
from pathlib import Path
import platform
import shutil
import socket
import sys

from .common import capture, paths


def gpu_info(c):
    text = capture(["nvidia-smi", f"--id={c['gpu_index']}",
                    "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"])
    name, uid, memory, driver = [s.strip() for s in text.split(",")]
    return {"name": name, "uuid": uid, "memory_total_mib": int(memory), "driver": driver}


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def version(value):
    return tuple(int(x) for x in value.split("."))


def host(c, check_space=True, require_gpu=True):
    from .environment import diagnose, require_passed
    result = diagnose(c, probe_torch=False, check_space=check_space, require_gpu=require_gpu)
    require_passed(result)
    return result


def cuda():
    import torch
    if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.6":
        raise RuntimeError(f"Expected torch 2.8.0/cu126, got {torch.__version__}/{torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA unavailable; CPU inference is disabled")
    x = torch.ones((32, 32), device="cuda")
    if float((x @ x)[0, 0].item()) != 32:
        raise RuntimeError("CUDA computation failed")
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}
