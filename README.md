# ModelScope ComfyUI Pipeline

在魔搭 GPU Notebook 上，一条命令安装独立 ComfyUI 环境，使用 SDXL 生成 PNG，再将这张真实图片交给 Wan2.2 TI2V 5B，生成并完整解码验收 MP4/H.264。

**当前状态：实现和离线测试已完成；魔搭 A10 GPU 集成验收尚未执行。** 模型约 25.08 GB，首次运行会下载权重和 PyTorch。只有真实两阶段推理及媒体验收均成功，才会显示 `E2E VALIDATION PASSED`。离线测试中的合成媒体仅为校验器测试输入。

## 选择验证方式

两种方式都在 clone 项目后、项目根目录执行，使用内置提示词完成真实 PNG→MP4 验收。

| 方式 | 实例及核心命令 | 执行过程 |
|---|---|---|
| 拆分验证 | CPU：`bash oneclick.sh --cpu-prepare`；GPU：`bash oneclick.sh --gpu-validate` | CPU 安装/下载/准备，GPU 检查 CUDA 并推理；独立存储需迁移模型 |
| 合并验证 | GPU：`bash oneclick.sh` | 同一 GPU 实例依次完成环境检查、安装、模型下载、工作流部署、启动及真实生成验收 |

两种方式的完整操作、模型迁移及日志位置见 [验证流程：拆分与合并](docs/GPU_VALIDATION.md)。

## 方式一：拆分验证（CPU + GPU）

CPU 与 GPU 均 clone 同一版本源码，推荐相同 Linux 镜像和 Python 版本。

CPU 实例进入项目目录后执行：

```bash
export WORKSPACE_ROOT=/mnt/workspace/modelscope-comfyui-pipeline-runtime
bash scripts/check_environment.sh --cpu --network
bash oneclick.sh --cpu-prepare
```

CPU 安装固定的 CUDA 构建依赖、下载四个模型并校验、部署工作流；不要求显卡或驱动，不启动服务或推理，成功仅标记 CPU PREPARATION PASSED。

确认两个实例共享同一存储且环境兼容时，在 GPU 实例使用相同 `WORKSPACE_ROOT` 执行 `bash oneclick.sh --gpu-validate`。若存储独立，CPU 先导出并自行传输模型包：

```bash
# CPU 实例：需要额外约 25 GB 空间；通过自己的传输通道复制到 GPU
bash oneclick.sh --export-models /mnt/workspace/pipeline-models.tar
# GPU 实例：设置 WORKSPACE_ROOT 并进入相同版本源码目录后
bash oneclick.sh --import-models /mnt/workspace/pipeline-models.tar
bash oneclick.sh --gpu-validate
```

模型包仅携带权重和校验清单，不迁移虚拟环境、凭证或 PID。导入不要求先安装环境；GPU 按需重新安装依赖，联网安装仍然必需。`--gpu-validate` 强制检查 CUDA 和真实生成，只使用已准备/导入且哈希正确的模型，不自动下载缺失权重。CPU 与 GPU 日志各自保留。两个实例都有 `/mnt/workspace` 不代表它们共享数据。

## 方式二：合并验证（单 GPU）

在已由你启动的 Linux x86_64 GPU Notebook 终端执行下方命令。项目不会创建、购买或更改实例。

```bash
cd /mnt/workspace
git -c http.version=HTTP/1.1 clone https://github.com/MashyGGG/modelscope-comfyui-pipeline.git
cd modelscope-comfyui-pipeline
bash oneclick.sh
```

核心命令只有 `bash oneclick.sh`，自动按顺序执行环境检测 → 安装依赖 → 下载校验模型 → 部署工作流 → 启动 ComfyUI → SDXL 文生图 → Wan 图生视频 → 完整解码验收。需要提前查看详细环境和网络诊断时，可先单独执行 `bash scripts/check_environment.sh --network`；它不是完整验证的替代命令。

合并验证无需 CPU 实例或模型传输。两种方式都以最终 `E2E VALIDATION PASSED` 为通过标志；每次实际操作自动保存日志和 JSON 摘要。CPU PREPARATION PASSED 只表示拆分方式的准备阶段完成。

初始支持范围：Python 3.10–3.12，Linux NVIDIA 驱动 >=560.35.03，PyTorch 2.8.0/cu126。主要验收目标是 A10 24GB、32GB 主机内存；建议首次运行预留 60–80 GiB 可用磁盘。该支持范围是实现基线，尚待实机验证；不支持的驱动/Python 会明确失败。系统无需预装 torch，也不会修改系统 torch 或驱动。

ComfyUI 固定为 v0.3.60 的 commit `b8730510db30c8858e1e5d8e126ef19eac395560`，使用其原生节点。选择这个已核实接口的版本以保证工作流稳定，而非运行时追随 latest。主要依赖见 `constraints.txt`，安装成功后保留实际 `pip-freeze.txt`；未声称所有间接依赖完全锁定。

Git 下载默认使用 HTTP/1.1：安装器在首次安装、失败重试或复用安装时，为本项目受管 ComfyUI 仓库写入本地 `http.version` 配置，日志显示协议和作用目录。不会修改全局 Git 配置或旧 ComfyUI。默认值位于 `config/defaults.yaml` 的 `git_http_version`。首次 clone 时项目代码尚未运行，因此下方示例显式使用 `git -c http.version=HTTP/1.1 clone ...`。

已有源码更新后重试 CPU 准备：

```bash
git -c http.version=HTTP/1.1 pull --ff-only
bash oneclick.sh --cpu-prepare
```

保留上次失败的受管安装目录即可重试。HTTP/1.1 配置用于改善 Git 传输兼容性，实际网络仍可能中断；它不影响 Python 对 `raw.githubusercontent.com` 的独立网络探测。

## 常用命令

```bash
bash oneclick.sh                               # 合并验证：GPU 实例执行完整流程
bash oneclick.sh --cpu-prepare                 # CPU：安装环境、下载模型、部署工作流；不执行 CUDA/推理
bash oneclick.sh --export-models /mnt/workspace/pipeline-models.tar
bash oneclick.sh --import-models /mnt/workspace/pipeline-models.tar
bash oneclick.sh --gpu-validate                # GPU：只复用已准备模型，强制 CUDA + E2E
bash scripts/check_environment.sh --cpu --network # CPU 实例环境检测
bash scripts/check_environment.sh --network    # 安装前独立检测，保存详细日志和 JSON
bash scripts/check_environment.sh --require-cuda # 安装后严格验证受管 PyTorch/CUDA
bash oneclick.sh --dry-run                     # 无写入、安装、下载或推理
bash oneclick.sh --skip-download               # 仍验证每个模型的大小与 SHA256
bash oneclick.sh --only-start                  # 仅使用已完成的安装及模型启动服务
bash oneclick.sh --stop                        # 只停止身份验证通过的受管进程
bash oneclick.sh --resume 20261114T000000Z-abcd1234
bash oneclick.sh --stage download              # 独立重试模型阶段
bash oneclick.sh --stage prepare
bash oneclick.sh --stage e2e                   # 已启动服务时，创建一次新的真实推理
bash oneclick.sh --stage e2e --resume RUN_ID    # 复用原 run 中已验收的图片，重做视频阶段
bash oneclick.sh --stage report --resume RUN_ID
```

首次在实例验证请先在 [完整验证流程](docs/GPU_VALIDATION.md) 中选择拆分或合并方式。环境检测只需要标准库，不安装软件或下载权重；默认检查系统、Python、Git/Bash、venv、GPU/驱动、显存、主机内存、磁盘、目录权限、端口和已有 PyTorch CUDA 运算。CPU 检测用 `--cpu` 明确跳过 GPU/驱动/端口门槛，只检查包导入；不能与 `--require-cuda` 合用。首次没有 torch 或系统 torch 不匹配时提示 WARN，安装后使用 `--require-cuda` 则视为失败。`--network` 额外探测四类下载服务，连通性检查不能保证后续大文件下载一定成功。

实际操作自动在终端和独立日志中同步输出 UTC 时间、开始/结束、耗时、命令输出、成功/警告/失败和异常堆栈。包括预检失败、安装、下载、工作流部署、服务启动/停止和 E2E。每次命令生成 `logs/<操作>-<UTC时间>-<随机ID>.log` 及同名 JSON 摘要，摘要记录每阶段状态与耗时；不会覆盖上次日志。`--dry-run` 继续保持无写入，仅在终端说明计划。不要使用 `bash -x`，避免 shell 打印环境凭证。

```bash
RUNTIME="${WORKSPACE_ROOT:-/mnt/workspace/modelscope-comfyui-pipeline-runtime}"
ls -lt "$RUNTIME/logs"
cat "$RUNTIME/state/latest-operation.json"
# 实时看指定的操作日志；服务长期输出在 comfyui-*.log
tail -f "$RUNTIME/logs/<终端打印的日志文件名>"
```

恢复运行必须传入原来的 `--config`、`--models`、环境变量和 `--set`，配置不一致会拒绝恢复。任务失败后复制终端中的 run_id；旧失败报告保存在该 run 的 `attempt-*` 目录。已经通过的 run 不能恢复，应新建运行。

```bash
bash oneclick.sh --set video.width=640 --set video.height=352 \
  --set video.frames=25 --set vae_tiled=true
bash oneclick.sh --set image.prompt='A lighthouse at sunrise' \
  --set video.prompt='Waves roll gently, the camera slowly moves forward'
bash oneclick.sh --set task_timeout=7200 --set port=8190
```

## 配置和存储

配置优先级：命令行 `--set` > 环境变量 > `--config` 文件。默认 YAML 使用 JSON 子集，便于没有第三方包时进行无副作用的 dry-run；运行环境安装 PyYAML 后也接受普通 YAML。普通 YAML 在全新系统 Python 下启动时需要先安装 PyYAML，或使用默认格式。

| 环境变量 | 含义 |
|---|---|
| `WORKSPACE_ROOT` | 运行资产根目录，默认 `/mnt/workspace/modelscope-comfyui-pipeline-runtime` |
| `COMFYUI_DIR` | 独立受管 ComfyUI 目录，默认 `$WORKSPACE_ROOT/ComfyUI` |
| `MODEL_CACHE_DIR` | SDK 缓存目录，默认 `$WORKSPACE_ROOT/model-cache` |
| `COMFYUI_PORT` | 服务端口，默认 8188 |
| `COMFYUI_REF` | 不可变 40 位 commit；更换后需要新运行根目录及重新核实节点 |
| `MODELSCOPE_API_TOKEN` | 可选，仅从环境读取，不写入源码、运行配置或报告 |
| `PYTHON` | 启动脚本使用的 Python，默认 python3 |

```text
$WORKSPACE_ROOT/
  ComfyUI/                      # 原生安装；模型目录链接到 SDK 缓存
  venv/                         # Python 环境
  model-cache/                  # 四个指定文件，断点与完整文件复用
  workflows/api/                # 部署后的 API JSON
  state/                        # 安装、服务身份、模型清单、操作状态
  logs/                         # 服务与执行日志
  output/
    images/<run_id>/            # SDXL 真实 PNG
    videos/<run_id>/            # Wan 真实 MP4
    reports/<run_id>/           # 状态、工作流、history、报告
    reports/e2e_report.json     # 最新报告副本
    reports/e2e_report.md
```

源码目录的 `logs/` 和 `output/` 预留给本地工具；Notebook 的实际产物统一写到上述持久化运行目录。固定报告副本会更新，各 run 的历史产物和报告保留。已损坏的目标模型不自动删除或替换，需要用户检查或选择新目录；下载中断的 SDK 缓存会保留。模型部署采用符号链接，不额外复制全部权重。

`config/models.yaml` 固定仓库、内容 commit（`revision`）、远程路径、字节大小和 SHA256，并显式配置 SDK 下载引用 `sdk_revision`。固定的 ModelScope SDK 1.29.2 实际按分支/标签解析下载版本，直接传入 commit SHA 会触发 `NotExistError`，因此 `sdk_revision` 使用已核实的 `master`。下载前分别核对固定 commit 与 SDK 分支的文件大小/SHA256，两者必须都匹配配置；下载后再次计算实际文件 SHA256。分支内容变化会明确失败，不能静默替换权重。

实际下载仍使用官方 `model_file_download` 单文件接口，SDK 实现 HTTP Range 续传和进度显示，外层有限次重试；不会下载整个仓库，不会改用其他来源。日志记录内容 commit、SDK 引用、远程路径，并为 SDK 版本/文件不存在错误提供脱敏详情。修复前已安装成功、停在模型下载的实例，只需更新源码后重跑 `bash oneclick.sh --cpu-prepare`，环境和完整文件会复用。

## 工作流与验收

`workflows/ui/` 是可以导入浏览器的 UI 工作流；自动部署到 `ComfyUI/user/default/workflows/modelscope-pipeline/`。`workflows/api/` 是供 `/prompt` 使用的 API JSON，两者由同一构建器生成，避免内容偏离。视频 UI 中的示例图片需要自行选择；自动 E2E 会绑定真实 PNG。

SDXL 默认 1024×576、24 steps、CFG 7。Wan 默认 832×480、49 帧、16 FPS、20 steps、CFG 5、shift 8。文生图与视频提示词分别配置。固定 seed 便于复现参数，不保证跨硬件逐像素一致。

服务只监听 `127.0.0.1`，关闭节点执行缓存，保存 PID、`/proc` 启动时间和命令身份；端口被其他程序占用就失败，不接管已有服务。默认让原生内存管理主动卸载模型，文本编码器使用 CPU 加载，采样始终要求 CUDA。视频阶段前请求卸载 SDXL。

调度器检查 `/object_info` 的节点、必填输入、选项、模型名和连接类型；依次提交两个任务并轮询对应 history。PNG 上传到独立输入目录，并核对上传文件与输出图片的 SHA256。仅解析目标保存节点的文件描述（SaveVideo 使用 `images` 字段），不按目录最新文件猜产物。

验收要求：PNG 可读且尺寸正确；MP4 非空、单视频流、H.264、尺寸/FPS/完整解码帧数正确、逐帧时间戳连续、时长约为帧数÷FPS。报告包含 GPU/模型版本、seed、prompt_id、耗时、文件哈希和视频参数。显存按服务进程采样，标注为**观测峰值**；采集失败时值为 null，不冒充精确 CUDA 峰值。

同一运行目录通过操作锁串行执行。执行自动验收时请让服务保持专用于该任务，避免从 UI 同时提交任务；已有任务会被检测并拒绝。超时清除本次排队任务，针对本次 prompt_id 请求中断，不发送无目标的全局中断。

服务在脚本成功或任务失败后继续运行。Notebook 会话/实例本身关闭时，平台可能终止进程；重新进入 Notebook 后使用 `--only-start`。浏览器访问使用魔搭平台提供的受认证端口代理，不构造 DSW 网关或公网隧道。

## 测试

无需 GPU 的检查（Python 3.10–3.12）：

```bash
python -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install pytest==8.4.2 requests==2.32.5 PyYAML==6.0.2 Pillow==11.3.0 av==15.1.0 modelscope==1.29.2
python scripts/check_static.py
python -m pytest -q
python scripts/audit_upstream.py   # 核实两类引用元数据和真实 SDK 版本解析，不下载权重
```

静态检查需要 Bash；Windows 自动使用 Git for Windows 的 Bash。PowerShell 使用 `.venv-dev\Scripts\python.exe` 运行同样的 Python 命令。GitHub Actions 在 Linux/Python 3.10、3.11、3.12 执行离线检查，远程 CI 结果需实际推送后查看。

魔搭 A10 的真实集成测试命令：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
bash oneclick.sh
RUNTIME="${WORKSPACE_ROOT:-/mnt/workspace/modelscope-comfyui-pipeline-runtime}"
cat "$RUNTIME/output/reports/e2e_report.md"
"$RUNTIME/venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASSED" and r["video"]["artifact"]["fully_decoded"]' "$RUNTIME/output/reports/e2e_report.json"
```

`--dry-run`、离线测试、合成测试 MP4 均不计入 GPU 验收。真实状态详见 [测试报告](docs/TEST_REPORT.md)。故障处理见 [排查文档](docs/TROUBLESHOOTING.md)。

## 开源说明

本项目调度代码使用 MIT 许可证。ComfyUI、ModelScope SDK 及各模型权重保留各自许可证；仓库不捆绑上游源码和模型权重。再分发时请保留上游来源与许可证说明。源码仓库：[MashyGGG/modelscope-comfyui-pipeline](https://github.com/MashyGGG/modelscope-comfyui-pipeline)，包含 GitHub Actions 离线检查配置。

接口核实依据：[ComfyUI 官方模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json)、[固定版本视频节点](https://github.com/Comfy-Org/ComfyUI/blob/v0.3.60/comfy_extras/nodes_video.py)、[ModelScope SDK 单文件下载](https://github.com/modelscope/modelscope/blob/v1.29.2/modelscope/hub/file_download.py)。
