import os
from pathlib import Path
import subprocess
import sys

from .common import ROOT, atomic_json, capture, fingerprint, load_data, paths, run, sha256


def installation_id(c):
    return fingerprint({"commit": c["comfyui_ref"], "python": sys.version_info[:2],
                        "requirements": sha256(ROOT / "requirements.txt"),
                        "constraints": sha256(ROOT / "constraints.txt"), "torch": "2.8.0+cu126"})


def verify(c):
    p = paths(c)
    marker = p["state"] / "installed.json"
    if not marker.exists() or load_data(marker)["fingerprint"] != installation_id(c):
        raise RuntimeError("Installation missing or changed; run the full oneclick command")
    if capture(["git", "-C", p["comfy"], "rev-parse", "HEAD"]) != c["comfyui_ref"]:
        raise RuntimeError("ComfyUI checkout differs from configured commit")
    if capture(["git", "-C", p["comfy"], "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("ComfyUI tracked files modified; refusing to use an unverified installation")
    if not p["python"].is_file():
        raise RuntimeError("Managed virtual environment is missing")


def validate_runtime(c, require_cuda):
    p = paths(c)
    if require_cuda:
        try:
            run([p["python"], ROOT / "scripts/preflight.py", "--cuda"],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(c["gpu_index"])})
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("GPU runtime validation failed; check the driver and Python/image compatibility. "
                               "For a moved/incompatible venv use a fresh WORKSPACE_ROOT and import the model bundle.") from exc
        return
    code = ("import torch; "
            "assert torch.__version__.split('+')[0]=='2.8.0', torch.__version__; "
            "assert torch.version.cuda=='12.6', torch.version.cuda; "
            "print('[PASS] CUDA-build packages imported on CPU; GPU computation NOT RUN')")
    run([p["python"], "-c", code], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def configure_git(c):
    p = paths(c)
    owner = p["state"] / "checkout-owner.json"
    if not owner.exists() or load_data(owner) != {"path": str(p["comfy"]), "commit": c["comfyui_ref"]}:
        raise RuntimeError("Refusing to configure Git in a ComfyUI directory outside this managed installation")
    protocol = c.get("git_http_version", "HTTP/1.1")
    run(["git", "-C", p["comfy"], "config", "--local", "http.version", protocol])
    print(f"[PASS] Git HTTP transport: {protocol}; scope=local; repository={p['comfy']}", flush=True)


def install(c, require_cuda=True):
    p = paths(c)
    marker = p["state"] / "installed.json"
    if marker.exists():
        verify(c)
        configure_git(c)
        validate_runtime(c, require_cuda)
        state = load_data(marker)
        state["last_validation"] = "GPU_CUDA" if require_cuda else "CPU_IMPORT_ONLY"
        atomic_json(marker, state)
        print("[PASS] ComfyUI Installation (reused)", flush=True)
        return
    for key in ("base", "state", "logs", "output"):
        p[key].mkdir(parents=True, exist_ok=True)
    if not p["comfy"].exists():
        p["comfy"].parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", p["comfy"]])
        run(["git", "-C", p["comfy"], "remote", "add", "origin", "https://github.com/Comfy-Org/ComfyUI.git"])
        atomic_json(p["state"] / "checkout-owner.json", {"path": str(p["comfy"]), "commit": c["comfyui_ref"]})
    owner = p["state"] / "checkout-owner.json"
    if not owner.exists() or load_data(owner) != {"path": str(p["comfy"]), "commit": c["comfyui_ref"]}:
        raise RuntimeError("COMFYUI_DIR already exists outside this managed installation; choose an empty directory")
    if capture(["git", "-C", p["comfy"], "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("Refusing to overwrite modified ComfyUI source")
    configure_git(c)
    run(["git", "-C", p["comfy"], "fetch", "--depth", "1", "origin", c["comfyui_ref"]])
    run(["git", "-C", p["comfy"], "checkout", "--detach", c["comfyui_ref"]])
    if not p["venv"].exists():
        run([sys.executable, "-m", "venv", p["venv"]])
    run([p["python"], "-m", "pip", "install", "pip==25.2"])
    run([p["python"], "-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0",
         "--index-url", "https://download.pytorch.org/whl/cu126"])
    run([p["python"], "-m", "pip", "install", "-c", ROOT / "constraints.txt", "-r", ROOT / "requirements.txt",
         "-r", p["comfy"] / "requirements.txt"])
    run([p["python"], "-m", "pip", "check"])
    validate_runtime(c, require_cuda)
    freeze = capture([p["python"], "-m", "pip", "freeze"])
    (p["state"] / "pip-freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    atomic_json(marker, {"fingerprint": installation_id(c), "commit": c["comfyui_ref"],
                         "last_validation": "GPU_CUDA" if require_cuda else "CPU_IMPORT_ONLY"})
    print("[PASS] ComfyUI Installation", flush=True)
