#!/usr/bin/env python3
"""No GPU, no downloads and no dependency on a ComfyUI installation."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common import ROOT, config, load_data, task_configs
from pipeline.models import specifications
from pipeline.workflows import build, validate


def main():
    sources = list((ROOT / "pipeline").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    config()
    for _, effective in task_configs(config(ROOT / "config/batch.example.yaml")):
        for graph in build(effective, specifications()):
            validate(graph)
    specifications()
    for path in (ROOT / "workflows").rglob("*.json"):
        data = load_data(path)
        if path.parent.name == "api":
            validate(data)
        else:
            if not data.get("nodes") or data.get("version") != 0.4:
                raise ValueError(f"Invalid UI workflow: {path}")
    bash = shutil.which("bash") if sys.platform != "win32" else "C:/Program Files/Git/bin/bash.exe"
    if not bash or not Path(bash).exists():
        raise RuntimeError("Bash unavailable; shell syntax checks NOT RUN")
    for path in [ROOT / "oneclick.sh", *(ROOT / "scripts").glob("*.sh")]:
        subprocess.run([bash, "-n", path.as_posix()], check=True)
    print(f"[PASS] Python syntax ({len(sources)} files), shell syntax, YAML/model configuration, UI/API JSON and API links")
    print("GPU integration: NOT RUN (this command only performs static checks)")


if __name__ == "__main__":
    main()
