#!/usr/bin/env python3
"""Bootstrap with the notebook Python; runtime stages use the managed venv."""
import argparse
from datetime import datetime, timezone
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import ROOT, atomic_json, config, load_data, lock, paths
from pipeline.models import specifications


def parser():
    p = argparse.ArgumentParser(description="ModelScope SDXL → Wan2.2 GPU pipeline")
    p.add_argument("--config", type=Path)
    p.add_argument("--models", type=Path)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--only-start", action="store_true")
    mode.add_argument("--stop", action="store_true")
    mode.add_argument("--cpu-prepare", action="store_true", help="CPU instance: install, download models and prepare workflows; no CUDA gate or inference")
    mode.add_argument("--gpu-validate", action="store_true", help="GPU instance: require prepared/imported models, validate CUDA and run fresh E2E")
    mode.add_argument("--export-models", type=Path, metavar="BUNDLE.tar", help="Export verified weights and a manifest for another instance")
    mode.add_argument("--import-models", type=Path, metavar="BUNDLE.tar", help="Import weights into portable cache without requiring an installed environment")
    mode.add_argument("--stage", choices=["preflight", "install", "download", "prepare", "start", "e2e", "report"])
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--resume", help="Resume a failed run using exactly the original config/model settings")
    p.add_argument("--cuda", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    return p


def runtime(args, c, items):
    from pipeline import preflight, models, workflows, server
    stage = args.stage
    if stage == "preflight":
        print(preflight.cuda() if args.cuda else preflight.host(c), flush=True)
        print("[PASS] CUDA / PyTorch" if args.cuda else "[PASS] GPU Environment", flush=True)
    elif stage == "download":
        models.download(c, items, args.skip_download)
    elif stage == "prepare":
        workflows.prepare(c, items)
    elif stage == "start":
        models.verify_all(c, items)
        server.start(c)
        from pipeline.client import ComfyClient
        for graph in workflows.build(c, items):
            # LoadImage lists input files; placeholder is intentionally supplied at actual submission time.
            info = ComfyClient(c).info()
            info["LoadImage"]["input"]["required"]["image"][0].append("generated.png") if "LoadImage" in info else None
            workflows.validate(graph, info)
        print("[PASS] Native nodes and model availability", flush=True)
    elif stage == "e2e":
        models.verify_all(c, items)
        from pipeline.orchestrator import execute
        execute(c, items, args.resume)
    elif stage == "report":
        if not args.resume:
            raise ValueError("--stage report requires --resume RUN_ID")
        from pipeline.common import checked_path
        from pipeline.report import report
        p = paths(c)
        run_dir = checked_path(p["output"] / "reports", args.resume)
        report(load_data(run_dir / "state.json"), run_dir, p["output"])


def main(argv=None):
    args = parser().parse_args(argv)
    if args.resume and (args.cpu_prepare or args.export_models or args.import_models):
        parser().error("--resume is only supported for GPU task execution")
    if args.gpu_validate:
        args.skip_download = True
    try:
        c = config(args.config, args.set, use_env=not args.runtime)
        items = specifications(args.models)
    except Exception:
        if args.runtime or args.dry_run:
            raise
        from pipeline.logging import Operation
        directory = Path(os.environ.get("WORKSPACE_ROOT", "/mnt/workspace/modelscope-comfyui-pipeline-runtime")) / "logs"
        with Operation(directory, "configuration", sys.argv):
            raise
    p = paths(c)
    if args.runtime:
        runtime(args, c, items)
        return
    if args.dry_run:
        for graph in __import__("pipeline.workflows", fromlist=["build"]).build(c, items):
            __import__("pipeline.workflows", fromlist=["validate"]).validate(graph)
        print("DRY RUN — no installation, download, directory creation, server or GPU inference")
        print(f"Workspace: {p['base']}\nComfyUI: {p['comfy']}\nCommit: {c['comfyui_ref']}\nModels: {sum(m['size'] for m in items)} bytes")
        print(f"Video: {c['video']['width']}x{c['video']['height']}, {c['video']['frames']} frames, {c['video']['fps']} FPS")
        print("Plan: host preflight → install → CUDA gate → model download → workflows → start → SDXL → Wan → MP4 validation")
        print("GPU E2E: NOT RUN")
        return
    from pipeline.logging import Operation
    name = ("cpu-prepare" if args.cpu_prepare else "gpu-validate" if args.gpu_validate else
            "export-models" if args.export_models else "import-models" if args.import_models else
            "stop" if args.stop else "start" if args.only_start else args.stage or "oneclick")
    with Operation(p["logs"], name, [sys.executable, str(ROOT / "scripts/cli.py"), *(argv if argv is not None else sys.argv[1:])], p["state"]) as operation:
        if sys.platform != "linux":
            raise RuntimeError("Runtime requires Linux x86_64 NVIDIA Notebook. Use --dry-run or offline tests here.")
        with lock(p["state"] / "operation.lock"):
            operate(args, c, items, p, operation)


def operate(args, c, items, p, operation):
    if args.export_models or args.import_models:
        from pipeline.transfer import export_models, import_models
        with operation.phase("Export portable models" if args.export_models else "Import portable models"):
            if args.export_models:
                export_models(c, items, args.export_models)
            else:
                import_models(c, items, args.import_models)
        return
    if args.stop:
        from pipeline.server import stop
        with operation.phase("Stop managed ComfyUI"):
            stop(c)
        return
    session_path = p["state"] / f"config-{operation.id}.json"
    model_path = session_path.with_name(session_path.stem + "-models.json")
    atomic_json(session_path, c)
    atomic_json(model_path, {"models": items})
    print(f"[INFO] Effective configuration: {session_path}")
    print(f"[INFO] Model configuration: {model_path}")
    def stage(name, cuda=False):
        label = name + ("-cuda" if cuda else "")
        with operation.phase(label):
            command = [str(p["python"]), str(ROOT / "scripts/cli.py"), "--runtime", "--stage", name,
                       "--config", str(session_path), "--models", str(model_path)]
            if args.skip_download:
                command.append("--skip-download")
            if args.resume:
                command += ["--resume", args.resume]
            if cuda:
                command.append("--cuda")
            # Only the parent transcript writes this operation's log: no duplicated child lines.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(c["gpu_index"]), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", start_new_session=True, env=env)
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        child.wait(timeout=15)
                raise
            if code:
                raise RuntimeError(f"Phase {name} failed with exit code {code}; log: {operation.log}")
    from pipeline.install import install, verify
    from pipeline.preflight import host
    if args.stage == "report":
        with operation.phase("Verify installation"):
            verify(c)
        stage("report")
        return
    with operation.phase("Host preflight"):
        host_settings = {**c, "min_disk_gib": c.get("min_runtime_disk_gib", 16)} if args.gpu_validate else c
        host(host_settings, check_space=not (p["state"] / "installed.json").exists(), require_gpu=not args.cpu_prepare)
    if args.stage == "preflight":
        return
    if args.gpu_validate:
        from pipeline.models import verify_available
        with operation.phase("Verify prepared or imported models"):
            verify_available(c, items)
    if args.only_start or args.stage in ("download", "prepare", "start", "e2e"):
        with operation.phase("Verify installation"):
            verify(c)
    else:
        with operation.phase("Install ComfyUI and dependencies"):
            install(c, require_cuda=not args.cpu_prepare)
    if args.cpu_prepare:
        for name in ("download", "prepare"):
            stage(name)
        preparation = {"status": "PREPARED", "gpu_e2e": "NOT RUN", "operation_id": operation.id,
                       "models": items, "comfyui_ref": c["comfyui_ref"], "log": str(operation.log)}
        atomic_json(p["state"] / "cpu-preparation.json", preparation)
        print("CPU PREPARATION PASSED; GPU E2E: NOT RUN", flush=True)
        return
    stage("preflight", cuda=True)
    if args.stage == "install":
        return
    if args.only_start:
        stage("start")
    elif args.stage:
        stage(args.stage)
    else:
        for name in ("download", "prepare", "start", "e2e"):
            stage(name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[FAIL] Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        from pipeline.logging import sanitized
        if "--runtime" in sys.argv:
            traceback.print_exc()
        print(f"[FAIL] {sanitized(str(exc))}", file=sys.stderr)
        sys.exit(1)
