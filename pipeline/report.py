from pathlib import Path

from .common import atomic_json


def report(state, run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    atomic_json(run_dir / "e2e_report.json", state)
    lines = ["# 端到端执行报告", "", f"- Run ID：`{state['run_id']}`", f"- 状态：**{state['status']}**",
             f"- 开始时间（UTC）：{state['started_at']}", f"- 最后更新时间（UTC）：{state['updated_at']}",
             f"- ComfyUI commit：`{state['config']['comfyui_ref']}`",
             f"- GPU：{state.get('environment', {}).get('gpu', {})}",
             f"- ComfyUI 进程显存采样观测峰值：{state.get('gpu_memory_observed_peak_mib')} MiB（非精确分配器峰值）", ""]
    for name in ("image", "video"):
        stage = state.get(name, {})
        lines += [f"## {name}", "", f"- 阶段状态：{stage.get('status', 'NOT RUN')}",
                  f"- Seed：{state['config'][name]['seed']}", f"- Prompt ID：{stage.get('prompt_id')}",
                  f"- 耗时：{stage.get('seconds')} 秒", f"- 产物：`{stage.get('artifact', {}).get('path')}`",
                  f"- SHA256：`{stage.get('artifact', {}).get('sha256')}`", ""]
        if name == "video":
            lines += [f"- 视频验收：{stage.get('artifact', {})}", ""]
    if state.get("error"):
        lines += ["## 失败原因", "", "```text", state["error"], "```", ""]
    lines += ["## 证据", "", "同目录保留实际提交的 API 工作流、history、状态及模型清单；离线测试不代表 GPU 集成通过。", ""]
    md = "\n".join(lines)
    (run_dir / "e2e_report.md").write_text(md, encoding="utf-8")
    latest = output / "reports"
    latest.mkdir(parents=True, exist_ok=True)
    atomic_json(latest / "e2e_report.json", state)
    # Latest aliases may change; each run's report is retained permanently.
    temp = latest / "e2e_report.md.tmp"
    temp.write_text(md, encoding="utf-8")
    temp.replace(latest / "e2e_report.md")
