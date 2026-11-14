# 验证流程：拆分与合并

提供两种入口，按实例使用方式选择：

| 方式 | 准备命令 | 最终验证命令 |
|---|---|---|
| 拆分：CPU 准备 → GPU 验收 | CPU 上 `bash oneclick.sh --cpu-prepare` | GPU 上 `bash oneclick.sh --gpu-validate` |
| 合并：同一 GPU 完成全流程 | 不需要单独准备 | GPU 上 `bash oneclick.sh` |

每条项目命令自动输出带时间戳的日志及 JSON 摘要。源码仓库为 `https://github.com/MashyGGG/modelscope-comfyui-pipeline`，可直接使用下方 clone 命令。两种方式均使用项目内置提示词，最终执行真实文生图、图生视频和 MP4 验收。

实例使用 Linux x86_64、Python 3.10–3.12。GPU 要求 NVIDIA 驱动 >=560.35.03，主要目标 A10 24GB。首次建议预留 60–80 GiB 可用空间。

## 方式一：拆分验证（CPU + GPU）

CPU 负责安装环境、下载并校验模型；GPU 负责驱动/CUDA 检查和真实生成。CPU 安装的是未来 GPU 使用的 PyTorch 2.8.0/cu126 构建，不是 CPU-only torch；CPU 阶段不运行 CUDA 运算、ComfyUI 服务或生成任务。

### 1. CPU 实例：准备环境和模型

```bash
cd /mnt/workspace
git clone https://github.com/MashyGGG/modelscope-comfyui-pipeline.git
cd modelscope-comfyui-pipeline
export WORKSPACE_ROOT=/mnt/workspace/modelscope-comfyui-pipeline-runtime
bash scripts/check_environment.sh --cpu --network
bash oneclick.sh --cpu-prepare
```

环境检查失败时先处理再继续。CPU 模式明确标记 GPU、驱动、显存、服务端口为 SKIP。准备命令执行系统检查、独立环境安装、包导入检查、四个权重下载及 SHA256 校验、工作流部署。

成功标志为 `CPU PREPARATION PASSED; GPU E2E: NOT RUN`。详细准备记录在 `$WORKSPACE_ROOT/state/cpu-preparation.json`；这不代表 GPU 验收成功。可重复执行相同命令，复用完整环境和模型。首次建议可用空间 60–80 GiB，导出模型包还需要额外约 25.08 GB。

### 2A. 共享存储：在 GPU 实例直接复用

仅当平台确认两个实例挂载的是同一持久化存储，并且运行根目录、缓存路径、系统镜像及 Python 兼容时，采用此方式。仅路径都叫 `/mnt/workspace` 不代表数据共享。

在 GPU 实例进入同一项目目录：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
export WORKSPACE_ROOT=/mnt/workspace/modelscope-comfyui-pipeline-runtime
bash scripts/check_environment.sh --network
bash oneclick.sh --gpu-validate
```

GPU 命令重新验证硬件/驱动、模型哈希、环境、CUDA 运算、节点和媒体产物。`installed.json` 中的 CPU 准备记录不能跳过 CUDA 门槛。若原虚拟环境不可用或镜像不同，保留现有目录，采用下面的独立存储方案，将模型导入一个新的 GPU 运行根目录重建环境；不直接复制虚拟环境、PID 或安装状态。

### 2B. 独立存储：导出、传输并导入模型

CPU 实例准备成功后导出：

```bash
bash oneclick.sh --export-models /mnt/workspace/pipeline-models.tar
```

项目逐个校验模型后写入未压缩 tar；文件已存在时拒绝覆盖。模型包只包含四个权重的真实字节和固定 revision/大小/SHA256 清单，排除符号链接、虚拟环境、访问凭证、服务状态和输出。导出额外占用约 25.08 GB 空间，日志保存在 CPU 实例的运行根目录。

通过平台文件传输、对象存储或你已有的 SSH/scp 通道，将 `pipeline-models.tar` 复制到 GPU 实例的 `/mnt/workspace/`。这一步由你选择可用通道；项目不假定实例互通，也不会上传凭证或自动操作实例。

GPU 实例分别 clone 同一版本源码，再执行：

```bash
cd /mnt/workspace
git clone https://github.com/MashyGGG/modelscope-comfyui-pipeline.git
cd modelscope-comfyui-pipeline
export WORKSPACE_ROOT=/mnt/workspace/modelscope-comfyui-pipeline-runtime
bash scripts/check_environment.sh --network
bash oneclick.sh --import-models /mnt/workspace/pipeline-models.tar
bash oneclick.sh --gpu-validate
```

如 GPU 上已有同名但不兼容的运行目录，改用新的 `WORKSPACE_ROOT`。导入无需先安装 ComfyUI 或 ModelScope SDK，逐个读取文件并重新计算 SHA256，拒绝路径越界、链接、额外文件及配置不符的模型包。完整文件可复用；损坏的已有目标不会被覆盖。导入文件先进入 `model-cache/portable/`，不会抢先创建 ComfyUI 安装目录。

`--gpu-validate` 要求模型已在共享目录或已导入，否则在安装前失败。它会在 GPU 上安装/复用独立环境，将已导入的模型部署到 ComfyUI，强制执行 CUDA 门槛，再进行完整 E2E，禁止自动补下模型。独立存储的 GPU 环境需要联网安装依赖；本流程不是完全离线安装。权重已就绪时，新环境安装磁盘门槛为 16 GiB（可用 `--set min_runtime_disk_gib=20` 调整），还需考虑导入前的模型包和权重各约 25 GB。

## 方式二：合并验证（单 GPU）

在同一个 GPU 实例 clone 项目后执行完整流程，不需要另开 CPU 实例或迁移模型：

```bash
cd /mnt/workspace
git clone https://github.com/MashyGGG/modelscope-comfyui-pipeline.git
cd modelscope-comfyui-pipeline
export WORKSPACE_ROOT=/mnt/workspace/modelscope-comfyui-pipeline-runtime
bash oneclick.sh
```

`bash oneclick.sh` 自动依次执行：系统/GPU/驱动检查 → 安装独立环境 → CUDA 运算验证 → 下载模型并检查 SHA256 → 部署工作流 → 启动 ComfyUI → 文生图 → 图生视频 → 完整解码验收并生成报告。重复执行会复用已验证的安装和模型，新推理结果使用独立 run_id。

可选：在完整命令之前执行 `bash scripts/check_environment.sh --network` 先收集环境及网络诊断；即使不单独运行它，完整命令仍会执行必需的环境和 CUDA 检查。若需要逐步排障，还可使用 `--stage install/download/prepare/e2e` 等阶段命令。

## 两种方式共同的结果检查

```bash
cat "$WORKSPACE_ROOT/output/reports/e2e_report.md"
```

流程为：内置提示词生成 SDXL PNG → 上传同一 PNG 并核对哈希 → Wan 生成 MP4 → 完整解码和元数据验证。

只有终端出现 `E2E VALIDATION PASSED` 且报告状态为 PASSED，才算 GPU 验收通过。默认输出为 832×480、49 帧、16 FPS、约 3.0625 秒的 H.264 MP4，再人工播放检查内容。

若图片成功但视频失败，保留原配置并恢复：

```bash
bash oneclick.sh --gpu-validate --resume <失败时打印的run_id>
```

合并方式使用：

```bash
bash oneclick.sh --resume <失败时打印的run_id>
```

恢复操作生成新日志并归档旧任务证据，不重新生成已验证的 PNG。

## 日志与停止

```bash
ls -lt "$WORKSPACE_ROOT/logs"
cat "$WORKSPACE_ROOT/state/latest-operation.json"
bash oneclick.sh --stop
```

拆分方式中 CPU 与 GPU 分别保留自己的 `logs/`；合并方式的全部日志保存在同一 GPU 实例。每次检测、准备、导出、导入、GPU 验收、合并全流程和停止都生成独立 `.log` 和 JSON 摘要；ComfyUI 长期日志为 `comfyui-*.log`。日志使用 UTC（北京时间加 8 小时），包含阶段、耗时、命令输出、警告/错误与异常堆栈；推理等待约每 30 秒输出进度。

同时保存 `state/cpu-preparation.json`、`state/pip-freeze.txt`、`state/model-manifest.json`、GPU 的 `output/reports/<run_id>/`。模型包不携带 CPU 日志，如需追踪两个实例，请单独保存日志文件。

`--dry-run` 保持无写入，CPU PREPARED、模型导入成功、离线测试通过都不计为 GPU 验收。本机尚未执行真实两个 Notebook 实例的安装或推理。

拆分方式的 CPU 准备必须使用 `--cpu-prepare`；合并方式的完整命令必须在 GPU 实例执行。
