# ModelScope ComfyUI Pipeline

在魔搭 GPU Notebook 上，一条命令安装独立 ComfyUI 环境，使用 SDXL 生成 PNG，再将这张真实图片交给 Wan2.2 TI2V 5B，生成并完整解码验收 MP4/H.264。

**当前状态：单任务已依据用户提供的 A10 24GB / 驱动 570.211.01 / 约 59 GiB 内存实例日志完成真实 PNG→MP4 验收；tasks 批量模式已完成离线测试，实机批量验收待执行。** 模型约 25.08 GB，首次运行会下载权重和 PyTorch。单任务通过标志为 `E2E VALIDATION PASSED`，整个批次通过标志为 `BATCH VALIDATION PASSED`。离线测试中的合成媒体仅为校验器测试输入。

## 选择验证方式

两种方式都在 clone 项目后、项目根目录执行，使用内置提示词完成真实 PNG→MP4 验收。

| 方式 | 实例及核心命令 | 执行过程 |
|---|---|---|
| 拆分验证 | CPU：`bash oneclick.sh --cpu-prepare`；GPU：`bash oneclick.sh --gpu-validate` | CPU 安装/下载/准备，GPU 检查 CUDA 并推理；独立存储需迁移模型 |
| 合并验证 | GPU：`bash oneclick.sh` | 同一 GPU 实例依次完成环境检查、安装、模型下载、工作流部署、启动及真实生成验收 |

两种方式的完整操作、模型迁移及日志位置见 [验证流程：拆分与合并](docs/GPU_VALIDATION.md)。

## 方式一：拆分验证（CPU + GPU）

CPU 与 GPU 均 clone 同一版本源码。环境按 Python 版本分别管理；CPU Python 3.11、GPU Python 3.12 可以共享同一运行根目录中的模型，由 GPU 自动安装自己的环境。

CPU 实例进入项目目录后执行：

```bash
export WORKSPACE_ROOT=/mnt/workspace/modelscope-comfyui-pipeline-runtime
bash scripts/check_environment.sh --cpu --network
bash oneclick.sh --cpu-prepare
```

CPU 安装固定的 CUDA 构建依赖、下载四个模型并校验、部署工作流；不要求显卡或驱动，不启动服务或推理，成功仅标记 CPU PREPARATION PASSED。

确认两个实例共享同一存储时，在 GPU 实例使用相同 `WORKSPACE_ROOT` 执行 `bash oneclick.sh --gpu-validate`。Python 版本不同会使用不同的虚拟环境，不改旧环境或重新下载完整权重。若存储独立，CPU 先导出并自行传输模型包：

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

运行基线：Python 3.10–3.12、PyTorch 2.8.0/cu126，主要目标 A10 24GB、32GB 主机内存。NVIDIA 驱动最低兼容范围为 525.60.13，推荐 >=560.35.03；位于两者之间（例如 550.54.15）只给 WARN，必须继续通过真实 CUDA 运算和完整 E2E，不能仅凭驱动号宣布兼容。首次运行建议预留 60–80 GiB，已有模型但需要新 Python 环境时额外预留至少 16 GiB。系统无需预装 torch，不修改系统 torch 或驱动。

针对 CPU Python 3.11 → GPU Python 3.12、A10 驱动 550.54.15 的升级操作和限制，见 [运行环境兼容说明](docs/RUNTIME_COMPATIBILITY.md)。

依赖安装分两步：仅三个带 `+cu126` 的 torch/torchvision/torchaudio 包从 PyTorch 官方 cu126 索引安装（`--no-deps`）；其余依赖默认走 `https://mirrors.aliyun.com/pypi/simple`，并应用 `constraints.txt`，例如 NumPy 1.26.4、Pillow 11.3.0。最后必须通过 `pip check` 和 CUDA 检查，临时只安装三个 wheel 不代表环境就绪。普通依赖源、读取超时和连接重试次数可用 `--set pypi_index_url=https://pypi.org/simple --set pip_timeout=30 --set pip_retries=3` 调整，不改系统 pip 配置。

ComfyUI 固定为 v0.3.60 的 commit `b8730510db30c8858e1e5d8e126ef19eac395560`，使用其原生节点。选择这个已核实接口的版本以保证工作流稳定，而非运行时追随 latest。主要依赖见 `constraints.txt`，安装成功后保留实际 `pip-freeze.txt`；未声称所有间接依赖完全锁定。

Git 下载默认使用 HTTP/1.1：安装器在首次安装、失败重试或复用安装时，为本项目受管 ComfyUI 仓库写入本地 `http.version` 配置，日志显示协议和作用目录。不会修改全局 Git 配置或旧 ComfyUI。默认值位于 `config/defaults.yaml` 的 `git_http_version`。首次 clone 时项目代码尚未运行，因此下方示例显式使用 `git -c http.version=HTTP/1.1 clone ...`。

已有源码更新后重试 CPU 准备：

```bash
git -c http.version=HTTP/1.1 pull --ff-only
bash oneclick.sh --cpu-prepare
```

保留上次失败的受管安装目录即可重试。HTTP/1.1 配置用于改善 Git 传输兼容性，实际网络仍可能中断；它不影响 Python 对 `raw.githubusercontent.com` 的独立网络探测。

## 单任务和多任务验证流程

以下命令用于模型已准备好的 GPU 实例；首次安装或双实例准备见 [完整验证流程](docs/GPU_VALIDATION.md)。先进入项目目录更新：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
git -c http.version=HTTP/1.1 pull --ff-only
```

**单任务：生成一张图片及对应视频。** 先检查细化配置，再执行真实生成：

```bash
bash oneclick.sh --dry-run --config config/single.refined.yaml
bash oneclick.sh --gpu-validate --config config/single.refined.yaml
```

原版对照时将配置换成 `config/single.original.yaml`；不传 `--config` 时仍使用 `config/defaults.yaml`。通过标志为 `E2E VALIDATION PASSED`。

**多任务：按 tasks 数组依次生成。** 首次复制一次示例，编辑本地文件中各项 name、image/video prompt 和 seed 后执行：

```bash
# 仅首次复制；后续保留并编辑已有任务文件
cp config/batch.example.yaml config/batch.local.yaml
# 编辑完成后再执行下面两条命令
bash oneclick.sh --dry-run --config config/batch.local.yaml
bash oneclick.sh --gpu-validate --config config/batch.local.yaml
```

整批通过标志为 `BATCH VALIDATION PASSED`；单项通过不代表整批通过。任务串行执行，失败即停止。批次汇总为 `$WORKSPACE_ROOT/output/batches/batch_report.md`，单任务最新报告为 `$WORKSPACE_ROOT/output/reports/e2e_report.md`；未设置 WORKSPACE_ROOT 时，默认根目录为 `/mnt/workspace/modelscope-comfyui-pipeline-runtime`。

`--dry-run` 只检查配置和工作流，不执行 GPU 推理。真实操作保存日志和独立产物，完成后还需人工查看图片并播放视频，确认视觉细节与 prompt 相符。

失败后保持原配置、模型和运行目录，恢复命令如下；将 RUN_ID 或 BATCH_ID 替换为失败日志中的实际 ID：

```bash
# 单任务：图片已通过时核对并复用原图，继续视频阶段
bash oneclick.sh --gpu-validate --config config/single.refined.yaml --resume RUN_ID
# 多任务：校验并跳过成功项，恢复失败项并继续剩余任务
bash oneclick.sh --gpu-validate --config config/batch.local.yaml --resume BATCH_ID
```

修改 prompt、seed 或任务列表后去掉 `--resume`，创建新运行。详细说明见 [单任务 prompt 微调](docs/PROMPT_TUNING.md) 和 [tasks 批量生成](docs/BATCH_GENERATION.md)。

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
bash oneclick.sh --gpu-validate --config config/batch.example.yaml # tasks 数组串行生成多组 PNG+MP4
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

保留 `config/defaults.yaml` 的顶层 `image`、`video` 单任务配置；没有 `tasks` 时每次生成一组 PNG+MP4。另提供 [批量示例](config/batch.example.yaml)：使用 `--config` 指定含非空 `tasks` 数组的文件，按顺序生成每一项。配置文件可只声明覆盖项，其余继承 defaults.yaml；每项通过唯一 `name` 和自己的 `image`、`video` 覆盖公共生成参数。任务字段优先于顶层生成默认值（含顶层 `--set image.*` / `video.*`），尺寸、步数、seed 等未指定时继承。`--set` 继续支持现有单任务操作；不支持用 `tasks.0.*` 索引修改数组，请直接编辑任务文件。

生成和恢复命令见上方 [单任务和多任务验证流程](#单任务和多任务验证流程)。`config/*.local.yaml` 被 Git 忽略，供保存自己的任务配置。每项都有独立 run_id、日志和报告，批次汇总保存在 `output/batches/<batch-ID>/`。详细配置、微调提示词及恢复流程见 [单任务与批量生成](docs/BATCH_GENERATION.md)。

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
  venv/                         # 旧版环境保留，升级后不自动执行
  venvs/py3.11/                 # 新版按 Python 版本分别安装
  venvs/py3.12/
  model-cache/                  # 四个指定文件，断点与完整文件复用
  workflows/api/                # 部署后的 API JSON
  state/                        # 服务身份、模型清单、操作状态
    environments/py3.12.json     # 该 Python 环境的独立安装记录
    environments/py3.12-pip-freeze.txt
    active-environment.json     # 最近通过安装验证的环境位置和状态
  logs/                         # 服务与执行日志
  output/
    images/<run_id>/            # SDXL 真实 PNG
    videos/<run_id>/            # Wan 真实 MP4
    reports/<run_id>/           # 状态、工作流、history、报告
    reports/e2e_report.json     # 最新报告副本
    reports/e2e_report.md
    batches/<batch-ID>/        # 批次状态、汇总报告、每项有效配置及恢复归档
    batches/batch_report.json  # 最近批次的汇总副本；不等于最近单任务报告
    batches/batch_report.md
```

源码目录的 `logs/` 和 `output/` 预留给本地工具；Notebook 的实际产物统一写到上述持久化运行目录。固定报告副本会更新，各 run 的历史产物和报告保留。已损坏的目标模型不自动删除或替换，需要用户检查或选择新目录；下载中断的 SDK 缓存会保留。模型部署采用符号链接，不额外复制全部权重。

`config/models.yaml` 固定仓库、内容 commit（`revision`）、远程路径、字节大小和 SHA256，并显式配置 SDK 下载引用 `sdk_revision`。固定的 ModelScope SDK 1.29.2 实际按分支/标签解析下载版本，直接传入 commit SHA 会触发 `NotExistError`，因此 `sdk_revision` 使用已核实的 `master`。下载前分别核对固定 commit 与 SDK 分支的文件大小/SHA256，两者必须都匹配配置；下载后再次计算实际文件 SHA256。分支内容变化会明确失败，不能静默替换权重。

实际下载仍使用官方 `model_file_download` 单文件接口，SDK 实现 HTTP Range 续传和进度显示，外层有限次重试；不会下载整个仓库，不会改用其他来源。日志记录内容 commit、SDK 引用、远程路径，并为 SDK 版本/文件不存在错误提供脱敏详情。修复前已安装成功、停在模型下载的实例，只需更新源码后重跑 `bash oneclick.sh --cpu-prepare`，环境和完整文件会复用。

## 模型与生成质量

项目下载四个权重文件，合计约 **25.08 GB**；其中两个是主生成模型，另外两个是视频所需的文本编码器和 VAE。文件、来源、大小及 SHA256 见 [模型清单](config/models.yaml)。

| 模型/组件 | 用途 |
|---|---|
| SDXL Base 1.0（`sd_xl_base_1.0.safetensors`） | 根据图片 prompt 生成 PNG |
| Wan2.2 TI2V 5B FP16（`wan2.2_ti2v_5B_fp16.safetensors`） | 根据输入 PNG 和视频 prompt 生成视频 |
| UMT5 XXL FP8（`umt5_xxl_fp8_e4m3fn_scaled.safetensors`） | 编码视频提示词 |
| Wan2.2 VAE（`wan2.2_vae.safetensors`） | 编解码视频画面 |

当前默认配置及用户 A10 实例已验证的输出规格：

| 产物 | 已验证规格 | 质量说明 |
|---|---|---|
| 图片 | 1024×576 PNG | 用户提供的雪山湖泊样例已有光影和空间层次；复杂结构、精确位置、数量和文字仍可能不符合提示词 |
| 视频 | 832×480、49 帧、16 FPS、约 3.06 秒，MP4/H.264 | 已完成真实生成、完整解码和参数验收；动作连贯性、主体形状和细节稳定性仍需人工播放检查 |

SDXL 官方模型卡说明复杂组合关系、文字和完美写实仍有局限，增加 prompt 细节不保证逐项准确呈现。[SDXL 官方说明](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

Wan2.2 TI2V 5B 官方支持 720P 级别生成；本项目目前实测的是上述 480 高度配置。提高分辨率或帧数会增加显存需求和生成耗时，需要在目标实例重新验证，不能直接将模型标称能力视为本仓库已验证的输出质量。[Wan 官方说明](https://github.com/Wan-Video/Wan2.2#run-text-image-to-video-generation)

**文件、分辨率和完整解码验收通过，不代表画面细节一定符合 prompt。** 最终还需人工查看图片并播放视频；具体实机条件及通过记录见 [测试报告](docs/TEST_REPORT.md)。

## 工作流与验收

单任务细化提示词另见 [原版保留与 prompt 微调](docs/PROMPT_TUNING.md)。`config/single.original.yaml` 保存原 image/video 参数，`config/single.refined.yaml` 细化红色帆船、白帆、雪山、倒影及慢镜头，保持 seed 和采样参数相同；执行 `bash oneclick.sh --gpu-validate --config config/single.refined.yaml`。默认配置及原 prompt 保持不变，细化版真实画面效果待 GPU 实测。

`workflows/ui/` 是可以导入浏览器的 UI 工作流；自动部署到 `ComfyUI/user/default/workflows/modelscope-pipeline/`。`workflows/api/` 是供 `/prompt` 使用的 API JSON，两者由同一构建器生成，避免内容偏离。视频 UI 中的示例图片需要自行选择；自动 E2E 会绑定真实 PNG。

SDXL 默认 1024×576、24 steps、CFG 7。Wan 默认 832×480、49 帧、16 FPS、20 steps、CFG 5、shift 8。文生图与视频提示词分别配置。固定 seed 便于复现参数，不保证跨硬件逐像素一致。

服务只监听 `127.0.0.1`，关闭节点执行缓存，保存 PID、`/proc` 启动时间和命令身份；端口被其他程序占用就失败，不接管已有服务。默认让原生内存管理主动卸载模型，文本编码器使用 CPU 加载，采样始终要求 CUDA。视频阶段前请求卸载 SDXL。

调度器检查 `/object_info` 的节点、必填输入、选项、模型名和连接类型；依次提交两个任务并轮询对应 history。PNG 上传到独立输入目录，并核对上传文件与输出图片的 SHA256。`LoadImage.image` 校验受管 input 目录内的真实文件，兼容子目录；该节点的 UI 下拉列表仅枚举根目录文件，不能据此拒绝已上传图片。其他固定选项仍严格匹配。仅解析目标保存节点的文件描述（SaveVideo 使用 `images` 字段），不按目录最新文件猜产物。

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
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASSED" and r["video"]["artifact"]["fully_decoded"]' "$RUNTIME/output/reports/e2e_report.json"
```

`--dry-run`、离线测试、合成测试 MP4 均不计入 GPU 验收。真实状态详见 [测试报告](docs/TEST_REPORT.md)。故障处理见 [排查文档](docs/TROUBLESHOOTING.md)。

## 开源说明

本项目调度代码使用 MIT 许可证。ComfyUI、ModelScope SDK 及各模型权重保留各自许可证；仓库不捆绑上游源码和模型权重。再分发时请保留上游来源与许可证说明。源码仓库：[MashyGGG/modelscope-comfyui-pipeline](https://github.com/MashyGGG/modelscope-comfyui-pipeline)，包含 GitHub Actions 离线检查配置。

接口核实依据：[ComfyUI 官方模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json)、[固定版本视频节点](https://github.com/Comfy-Org/ComfyUI/blob/v0.3.60/comfy_extras/nodes_video.py)、[ModelScope SDK 单文件下载](https://github.com/modelscope/modelscope/blob/v1.29.2/modelscope/hub/file_download.py)。
