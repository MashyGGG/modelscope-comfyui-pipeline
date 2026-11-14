#!/usr/bin/env python3
"""Run before installation. Uses standard library; always saves diagnostic evidence."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import atomic_json, config, paths
from pipeline.environment import diagnose, require_passed
from pipeline.logging import Operation, sanitized


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone environment diagnosis; no installation or weight downloads")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--log-dir", type=Path, help="Override log directory if the workspace is not writable")
    parser.add_argument("--network", action="store_true", help="Probe GitHub/ModelScope/PyPI/PyTorch endpoints")
    parser.add_argument("--require-cuda", action="store_true", help="After installation require the managed torch/CUDA computation to pass")
    parser.add_argument("--cpu", action="store_true", help="CPU preparation checks; skip GPU, driver and service-port gates")
    args = parser.parse_args(argv)
    if args.cpu and args.require_cuda:
        parser.error("--cpu and --require-cuda cannot be combined")
    c = config(args.config, args.set)
    with Operation(args.log_dir or paths(c)["logs"], "environment", sys.argv,
                   None if args.log_dir else paths(c)["state"]) as operation:
        with operation.phase("Environment diagnosis"):
            settings = {**c, "min_disk_gib": c.get("min_runtime_disk_gib", 16)} if (paths(c)["state"] / "model-manifest.json").is_file() else c
            result = diagnose(settings, require_cuda=args.require_cuda, network=args.network,
                              check_space=not paths(c)["install_marker"].is_file(), require_gpu=not args.cpu)
            output = operation.log.with_name(operation.log.stem + "-checks.json")
            atomic_json(output, result)
            print(f"[INFO] Full environment results: {output}")
            require_passed(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"[FAIL] {sanitized(str(exc))}", file=sys.stderr)
        sys.exit(1)  # Full traceback has already been persisted by Operation.
