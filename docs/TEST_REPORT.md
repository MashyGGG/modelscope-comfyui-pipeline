# 本地测试与 GPU 验收状态

记录日期：2026-11-14（用户时区 Asia/Shanghai）。

## 已实际执行

环境：Windows、Python 3.11.9、Git for Windows Bash；独立 `.venv-dev`。未在本机安装 GPU 推理依赖或下载模型权重。

| 检查 | 实际结果 |
|---|---|
| Python 语法、Shell `bash -n` | PASS |
| YAML 配置、模型清单、四份 UI/API JSON、API 图连接 | PASS |
| 离线 pytest | 44 passed |
| `bash oneclick.sh --dry-run` | PASS；未执行安装、下载或推理 |
| 固定 ModelScope revision 的四个真实文件路径、大小与 SHA256 | PASS，通过只读仓库 API 实查 |
| 固定 ComfyUI commit 的 Wan/Video 节点、缓存开关、history 字段、定向中断接口 | PASS，通过官方源码实查 |

原始离线结果：[local-checks.log](local-checks.log)、[offline-tests.xml](offline-tests.xml)；联网核实结果：[upstream-audit.log](upstream-audit.log)。

离线测试覆盖：无副作用 dry-run、配置优先级与非法参数、缺失 GPU、路径越界、UI/API 对齐和连接、必需节点缺失、模型完整性/复用/有限重试、SDK 登录元组与单文件参数、错误信息凭证隐藏、真实本地 HTTP 空响应处理、定向任务取消、history 文件定位、失败报告及图片阶段续跑、配置变更拒绝恢复、PID 复用拒绝停止，以及使用 PyAV 实际编码的合成 H.264 测试文件的完整解码与损坏拒绝。

合成 H.264 文件和 API doubles **仅用于离线测试**，不属于 SDXL/Wan 推理结果；未将它们保存为项目 E2E 成功报告。

本次补充：独立环境检测在无 ComfyUI/torch 安装时也能生成详细结果；遇到失败继续收集后续检查。测试覆盖日志时间戳、父/子进程输出与 stderr、失败阶段和异常堆栈、耗时及 JSON 摘要、Token 脱敏、停止操作日志、配置解析失败日志、缺失受管环境时严格 CUDA 检查失败，以及 dry-run 保持无写入。日志相关测试使用离线 doubles，不代表 Linux GPU 运行验证。

双实例模式新增测试：CPU 预检跳过 GPU/驱动/端口；CPU 安装门槛仅检查 CUDA 构建包导入，禁止调用 CUDA 运算；CPU 调度仅下载/部署，不能启动服务或采样；GPU 调度仍执行 CUDA 门槛并强制 skip-download；缺失预备模型时提前失败。使用微型离线权重 fixture 测试不同目录间的真实 tar 导出/导入、导入后部署、重复复用、现有包防覆盖，以及损坏哈希/清单不匹配/路径越界/符号链接的拒绝。CPU 准备状态与 GPU E2E 通过状态分开记录。

## 尚未执行

| 验收项 | 状态 |
|---|---|
| Linux A10 Notebook 完整依赖安装与 CUDA 运算 | NOT RUN |
| 约 25.08 GB 权重真实下载、断点续传与落盘 SHA256 | NOT RUN（元数据已核实） |
| 原生 ComfyUI 启动和运行时 `/object_info` 校验 | NOT RUN |
| SDXL 真实生成 1024×576 PNG | NOT RUN |
| Wan2.2 真实读取该 PNG 并生成 49 帧视频 | NOT RUN |
| 真实 GPU 输出 MP4 的完整解码验收 | NOT RUN |
| A10 显存、内存、耗时测量 | NOT RUN |
| 真实 CPU Notebook 环境安装和完整模型下载 | NOT RUN |
| 两个 Notebook 实例间约 25 GB 模型包传输及 GPU 复用 | NOT RUN（离线跨目录 fixture 已通过） |
| Python 3.10/3.12 和远程 GitHub Actions | NOT RUN（已提供 CI 配置） |

当前没有已连接的魔搭 A10 Notebook；本地也没有可用的 nvidia-smi。因此本报告不能作为 GPU E2E 通过证明。

## 在魔搭执行真实验收

在已经创建的 A10 GPU Notebook 内，项目源码放到 `/mnt/workspace/modelscope-comfyui-pipeline` 后执行：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
bash oneclick.sh
RUNTIME="${WORKSPACE_ROOT:-/mnt/workspace/modelscope-comfyui-pipeline-runtime}"
cat "$RUNTIME/output/reports/e2e_report.md"
"$RUNTIME/venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASSED" and r["image"]["status"]=="PASSED" and r["video"]["artifact"]["fully_decoded"]' "$RUNTIME/output/reports/e2e_report.json"
```

只有全部通过后，终端才会显示 `E2E VALIDATION PASSED`。请保留该次 `output/reports/<run_id>/`、`logs/`、`state/pip-freeze.txt`，以便补充实机兼容性和性能结果。测试失败时保留失败报告，不替换为预制产物。
