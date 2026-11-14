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


def driver_status(c, value):
    minimum = c.get("min_driver", "525.60.13")
    recommended = c.get("recommended_driver", "560.35.03")
    if version(value) < version(minimum):
        return "FAIL", f"{value}; CUDA 12.x minimum compatibility driver={minimum}"
    if version(value) < version(recommended):
        return "WARN", (f"{value}; CUDA 12.x minor-version compatibility candidate; recommended >= {recommended}; "
                        "actual CUDA operations and full model inference must pass; PTX/newer-driver features may fail")
    return "PASS", f"{value}; meets recommended driver {recommended}; CUDA computation still required"


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
    import torch.nn.functional as functional
    operations = []
    x = torch.ones((32, 32), device="cuda", dtype=torch.float16)
    if float((x @ x)[0, 0].item()) != 32:
        raise RuntimeError("CUDA computation failed")
    operations.append("fp16_matmul")
    image = torch.ones((1, 1, 16, 16), device="cuda", dtype=torch.float16)
    kernel = torch.ones((1, 1, 3, 3), device="cuda", dtype=torch.float16)
    conv = functional.conv2d(image, kernel)
    if conv.device.type != "cuda" or not bool(torch.all(conv == 9).item()):
        raise RuntimeError("CUDA convolution failed")
    operations.append("fp16_conv2d")
    query = torch.ones((1, 2, 16, 32), device="cuda", dtype=torch.float16)
    attention = functional.scaled_dot_product_attention(query, query, query)
    if attention.device.type != "cuda" or not bool(torch.allclose(attention, query)):
        raise RuntimeError("CUDA attention failed")
    operations.append("fp16_scaled_dot_product_attention")
    if torch.cuda.is_bf16_supported():
        bf16 = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        if float((bf16 @ bf16)[0, 0].item()) != 32:
            raise RuntimeError("CUDA BF16 matrix multiplication failed")
        operations.append("bf16_matmul")
    torch.cuda.synchronize()
    return {"available": True, "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.version.split()[0], "checked_operations": operations,
            "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}
