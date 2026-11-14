# 故障排查

## 首先收集环境诊断

拆分验证：CPU 使用 `bash scripts/check_environment.sh --cpu --network` 和 `bash oneclick.sh --cpu-prepare`；GPU 使用 `bash oneclick.sh --gpu-validate`。独立存储先通过 `--export-models` / `--import-models` 迁移模型。合并验证：在同一 GPU 实例运行 `bash oneclick.sh`，完成全部准备和真实验收。完整步骤见 [验证流程：拆分与合并](GPU_VALIDATION.md)。

clone 完成后运行 `bash scripts/check_environment.sh --network`，将终端打印的 `.log`、同名操作 `.json` 及 `-checks.json` 保留下来。脚本会汇总全部检查，必需项失败时退出码为 1。安装后用 `--require-cuda` 检查受管环境。若默认目录不可写，使用 `--log-dir /mnt/workspace/diagnostic-logs` 指定可写目录。

两种方式的完整命令见 [验证操作流程](GPU_VALIDATION.md)。

## 环境与安装

- GitHub clone 出现 HTTP/2 framing error：使用 `git -c http.version=HTTP/1.1 clone https://github.com/MashyGGG/modelscope-comfyui-pipeline.git`。代码尚未 clone 时无法自行设置协议，所以需要命令行参数。
- ComfyUI fetch 出现 GnuTLS、EOF 或 index-pack 错误：通常表示源码包传输中断。新版安装器默认将受管仓库设置为 HTTP/1.1，更新本项目后直接重试；旧目录会保留，日志应出现 `[PASS] Git HTTP transport: HTTP/1.1`。这不保证消除所有网络断流。

```bash
# 在本项目源码目录更新并重试
git -c http.version=HTTP/1.1 pull --ff-only
bash oneclick.sh --cpu-prepare
```

Git 配置只作用于 Git 传输，独立网络探测使用的 `raw.githubusercontent.com` 地址仍可能连接失败；该域名探测失败不等于本地源码缺失，也不能单独证明 `github.com` Git fetch 必然失败。

- CPU 缺少 nvidia-smi：CPU 准备模式会 SKIP，属于正常情况；默认 GPU 命令仍会失败。CPU 安装的是 CUDA 构建 torch，导入成功并不意味着已完成 CUDA 验证。
- GPU 无法复用 CPU 虚拟环境：两个实例的基础镜像/解释器可能不同。保留旧目录，选择新的 WORKSPACE_ROOT 并导入模型包；GPU 会重建环境。不要把 CPU 的 installed.json/PID 单独复制过去冒充安装完成。
- GPU `Prepared model missing/incomplete`：两个实例可能并不共享存储。先导出/传输/导入模型包，再运行 --gpu-validate；该模式不会自动补下权重。
- 导出空间不足：tar 包不压缩，额外需要约 25.08 GB。可把导出目标放在其他可写且空间足够的挂载点。
- 导入清单或哈希不匹配：检查源码/config/models.yaml 版本是否相同、传输是否完整；不自动覆盖损坏文件，也不更换权重来源。

- Windows 正常运行仅支持 `--dry-run` 和离线测试；实际部署目标为 Linux GPU Notebook。
- 未检测到 nvidia-smi/GPU：确认 Notebook 已选择 GPU 实例。脚本不会申请或变更实例。
- 系统 Python 没有 torch 不影响首次安装；虚拟环境 CUDA 检查失败才中止。
- 驱动低于 560.35.03 或 Python 不在 3.10–3.12：使用满足支持基线的 Notebook 镜像，脚本不自动升级驱动。
- ComfyUI 目录已存在或被修改：选择新的 `WORKSPACE_ROOT`/`COMFYUI_DIR`，脚本不会覆盖陌生目录或修改过的源码。
- pip 安装失败：保留工作目录后重跑，已下载包可由 pip 缓存复用。检查网络、Python 和磁盘；成功后保存 `state/pip-freeze.txt`，后续安装检查 `state/installed.json`。
- 可用磁盘不足：默认首次要求 60 GiB，可用 `--set min_disk_gib=40` 调整，但需自行确认缓存、依赖和输出空间充足。

## 模型

- 下载失败：检查 ModelScope 连通性、可选 token 权限、磁盘；重跑完整命令或 `--stage download`。已完成且哈希正确的模型不会重新下载；未完成部分由 SDK 续传。
- 固定 revision/路径/SHA256 不一致：运行 `python scripts/audit_upstream.py` 核对。禁止把其他来源同名文件当作同一模型。
- 目标文件已经存在但损坏：原文件保留，明确失败。检查后自行处理，或改用新的运行目录，脚本不自动删除用户文件。
- `--skip-download` 不是跳过模型检查，仍会读取模型文件计算 SHA256。
- Token 通过 `MODELSCOPE_API_TOKEN` 传入，不放进 YAML、命令行参数或源码。ModelScope SDK 登录可能按其官方行为保存本机认证缓存。

## 服务

- 端口占用：更换 `--set port=8190`；不停止陌生进程。复用服务必须 PID、启动时间、命令行和配置都匹配。
- 启动失败/超时：终端输出最后 100 行；完整日志位于运行目录 `logs/comfyui-*.log`。需要更长时间可设 `--set startup_timeout=300`。
- 受管进程仍在但不健康：检查日志，用相同配置 `--stop` 后再 `--only-start`。
- 浏览器无法访问：使用 Notebook 的受认证端口代理；脚本仅监听 localhost，不发布公网地址。不承诺所有 Notebook 产品使用同一种代理入口。
- `--stop` 无法验证身份：检查配置是否与启动时一致；不要根据旧 PID 手工盲目结束进程。

## 推理与恢复

- OOM：关闭占用 GPU 的其他任务，新运行时降低参数，例如 `--set video.width=640 --set video.height=352 --set video.frames=25 --set vae_tiled=true`。
- 帧数必须为 4n+1，视频宽高是 32 的倍数；修改配置后不能恢复旧 run，应新建运行。
- SDXL 成功、视频失败：使用终端给出的 run_id 执行 `--resume RUN_ID`，保持原配置。程序检查原 PNG 哈希并重新上传，只重复未完成的视频阶段。
- 超时：记录失败，清理本次待执行任务并针对本次 prompt_id 发出中断；服务继续运行。等该任务退出、队列空闲后恢复。
- 任务缓存命中：自动验收拒绝缓存证据。使用本项目受管的 `--cache-none` 服务。
- MP4 任务完成但验收失败：查看对应 `video-history.json`、报告和服务日志，不手工将状态改为成功。完整解码可能发现帧数、编码或文件损坏问题。
- 进程被 SIGKILL/实例断电：无法保证最后一次状态写入；保留旧任务证据。重启服务后按相同 run_id 恢复，已验证的图片可继续使用。
- 不支持通过 UI 与自动验收同时抢占队列；批量任务、多 GPU、Web 后台属于后续阶段。

## 日志和报告

`state/latest-operation.json` 指向最近操作摘要。每次操作保存 `logs/<操作>-<ID>.log` 和同名 `.json`，记录时间戳、各阶段开始/结束、耗时、命令输出、失败堆栈；安装、下载、启动、停止与预检失败都包含在内。服务长期日志位于 `logs/comfyui-*.log`，推理日志约每 30 秒输出等待进度。推理阶段报告保存在 `output/reports/<run_id>/`，包含 API Prompt、history、模型清单和状态。配置在初始化前即失败时，错误日志写入 `WORKSPACE_ROOT/logs` 或默认运行根目录的 logs。

`output/reports/e2e_report.*` 为最新执行报告副本，可能是失败状态。每次恢复前将旧报告归档到 `attempt-*`。不要仅依据目录中存在 MP4 判断当前执行通过。
