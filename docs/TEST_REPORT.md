# 本地测试与 GPU 验收状态

记录日期：2026-11-14（用户时区 Asia/Shanghai）。

## 已实际执行

环境：Windows、Python 3.11.9、Git for Windows Bash；独立 `.venv-dev`。未在本机安装 GPU 推理依赖或下载模型权重。

| 检查 | 实际结果 |
|---|---|
| Python 语法、Shell `bash -n` | PASS |
| YAML 配置、模型清单、四份 UI/API JSON、API 图连接 | PASS |
| 离线 pytest | 103 passed |
| `bash oneclick.sh --dry-run` | PASS；未执行安装、下载或推理 |
| ModelScope 固定内容 revision 与 SDK 分支的四个文件路径、大小与 SHA256 | PASS，通过只读仓库 API 实查；实际 SDK 接受 master 引用 |
| 固定 ComfyUI commit 的 Wan/Video 节点、缓存开关、history 字段、定向中断接口 | PASS，通过官方源码实查 |

原始离线结果：[local-checks.log](local-checks.log)、[offline-tests.xml](offline-tests.xml)；联网核实结果：[upstream-audit.log](upstream-audit.log)。

离线测试覆盖：无副作用 dry-run、配置优先级与非法参数、缺失 GPU、路径越界、UI/API 对齐和连接、必需节点缺失、模型完整性/复用/有限重试、SDK 登录元组与单文件参数、错误信息凭证隐藏、真实本地 HTTP 空响应处理、定向任务取消、history 文件定位、失败报告及图片阶段续跑、配置变更拒绝恢复、PID 复用拒绝停止，以及使用 PyAV 实际编码的合成 H.264 测试文件的完整解码与损坏拒绝。

合成 H.264 文件和 API doubles **仅用于离线测试**，不属于 SDXL/Wan 推理结果；未将它们保存为项目 E2E 成功报告。

本次补充：独立环境检测在无 ComfyUI/torch 安装时也能生成详细结果；遇到失败继续收集后续检查。测试覆盖日志时间戳、父/子进程输出与 stderr、失败阶段和异常堆栈、耗时及 JSON 摘要、Token 脱敏、停止操作日志、配置解析失败日志、缺失受管环境时严格 CUDA 检查失败，以及 dry-run 保持无写入。日志相关测试使用离线 doubles，不代表 Linux GPU 运行验证。

双实例模式新增测试：CPU 预检跳过 GPU/驱动/端口；CPU 安装门槛仅检查 CUDA 构建包导入，禁止调用 CUDA 运算；CPU 调度仅下载/部署，不能启动服务或采样；GPU 调度仍执行 CUDA 门槛并强制 skip-download；缺失预备模型时提前失败。使用微型离线权重 fixture 测试不同目录间的真实 tar 导出/导入、导入后部署、重复复用、现有包防覆盖，以及损坏哈希/清单不匹配/路径越界/符号链接的拒绝。CPU 准备状态与 GPU E2E 通过状态分开记录。

HTTP/1.1 修复新增 5 项测试，使用真实临时 Git 仓库验证本地协议配置、全局/旧 ComfyUI 配置不变、旧配置文件回退默认 HTTP/1.1、首次 fetch 与失败后重试均先设置协议、拒绝修改非受管仓库，以及非法协议值校验。fetch 失败以离线方式注入，未将本地配置测试当成远程网络问题已解决的证据。

ModelScope 版本解析修复新增 5 项测试：使用真实 SDK 1.29.2 解析器、模拟远程分支列表，复现 commit SHA 被拒绝而 master 被接受；验证下载前先核对固定内容 commit 和 SDK 引用的元数据，再向 SDK 传入显式分支；分支元数据漂移或下载字节哈希错误均拒绝部署。离线检查依赖中加入固定 SDK，CI 同样安装，以覆盖实际解析行为。

另已只读联网复现：SDXL 与 Wan 配置中的 SHA 在 SDK 1.29.2 下返回 NotExistError，master 可以解析；四个文件的 master 元数据与固定内容 revision 的大小/SHA256 完全一致。本地没有下载完整权重，此结果仅证明版本解析与元数据符合预期。

## 实例验证状态

| 验收项 | 状态 |
|---|---|
| Linux A10 Notebook 完整依赖安装与 CUDA 运算 | PASS（用户日志：Python 3.12.13、驱动 570.211.01、torch 2.8.0+cu126；FP16 矩阵乘法/卷积/注意力、BF16 矩阵乘法通过） |
| 约 25.08 GB 权重真实下载与落盘 SHA256 | PASS（用户 CPU 实例日志，四个文件全部校验通过）；真实中断续传场景待验证 |
| 原生 ComfyUI 启动和运行时 `/object_info` 校验 | PASS（用户恢复日志已确认子目录输入校验通过，视频任务被接受并完成） |
| SDXL 真实生成 1024×576 PNG | PASS（用户日志：任务约 16 秒，PNG 尺寸及上传 SHA256 均通过） |
| Wan2.2 真实读取该 PNG 并生成 49 帧视频 | PASS（用户日志：视频任务约 80.1 秒完成） |
| 真实 GPU 输出 MP4 的完整解码验收 | PASS（用户日志出现 MP4 Validation 和 E2E VALIDATION PASSED，用户已人工播放视频） |
| A10 显存、内存、耗时测量 | 日志确认 A10 24564 MiB、系统内存约 58.85 GiB、恢复总耗时 196.528 秒；精确内存/显存峰值未提供 |
| tasks 数组真实 GPU 批量生成及恢复 | NOT RUN（新增离线回归已通过） |
| 真实 CPU Notebook 环境安装和完整模型下载 | PASS（用户提供完整日志，约 352 秒完成 CPU PREPARATION） |
| GPU A10 / 550.54.15 / Python 3.12 | 早期实例安装遇到网络故障，未完成 CUDA；后续成功证据来自驱动 570.211.01 的 A10，不代表 550.54.15 已实测通过 |
| 两个 Notebook 实例间约 25 GB 模型包传输及 GPU 复用 | NOT RUN（离线跨目录 fixture 已通过） |
| Python 3.10/3.12 和远程 GitHub Actions | NOT RUN（已提供 CI 配置） |

当前没有已连接的魔搭 A10 Notebook，本地也没有可用的 nvidia-smi。表中真实单任务通过依据为用户提供的实例日志；本地测试不作为 GPU 推理证据，批量模式仍待实例验证。

用户提供的 2026-11-14 CPU 实例日志显示：系统为 Linux x86_64、Python 3.11.11，CPU 准备预检通过；早期 ComfyUI fetch 在约 372 秒后出现 GnuTLS recv error、EOF 及 index-pack 错误，退出码 128。默认启用 Git HTTP/1.1 后，实例重试已确认源码拉取成功，见下段。

后续同日 07:16–07:22 UTC 日志确认 HTTP/1.1 fetch 在约 7.7 秒完成，ComfyUI 及依赖安装通过，模型下载报 NotExistError。根因为 SDK 分支/标签解析与 commit SHA 不兼容，已修复并保持内容 SHA256 锁定；实际约 25 GB 权重下载及 GPU E2E 仍待用户实例完成。

随后 07:31–07:37 UTC CPU 实例日志确认四个模型均下载且通过大小/SHA256 校验、工作流部署成功，最终 CPU PREPARATION PASSED，耗时 352.281 秒。07:46 UTC GPU 日志显示 A10 23028 MiB、驱动 550.54.15、Python 3.12.13，被旧的 560.35.03 驱动门槛提前拦截，尚未真正测试 CUDA。

本次新增 11 项离线回归：驱动兼容下限/推荐门槛分级；Python 3.11/3.12 选择独立环境和服务身份；无 CUDA 不可通过；新建版本环境时旧环境、旧标记、源码及模型保留；当前环境重复安装复用；模拟 CUDA 失败不能产生成功标记；解释器身份不符拒绝复用；其他 Python 版本的受管服务可明确停止但不可自动复用。环境安装和 CUDA 部分使用离线 doubles，源码复用检查使用真实临时 Git 仓库。真实 GPU 探针新增 FP16 矩阵乘法/卷积/注意力及可用时 BF16 检查；这些 GPU 运算在本地无 GPU 环境尚未执行。

本次依赖下载源修复再新增 6 项离线回归，覆盖：三个官方 cu126 wheel 使用 --no-deps 和版本约束；普通依赖使用独立 PyPI 镜像并完整解析依赖；依赖失败停止且不继续成功检查；索引、超时和重试次数可配置；非法参数拒绝。只读检查已确认阿里云索引中存在 Python 3.12 的 NumPy 1.26.4、Pillow 11.3.0 及所需 nvidia-cusparselt-cu12 0.7.1 条目，未下载大 wheel，也未把索引可访问当成实例安装通过。

后续用户提供的 12:36–13:48 UTC GPU 日志确认：A10 24564 MiB、驱动 570.211.01、Python 3.12.13；依赖安装约 3950 秒完成，CUDA 运算及四个模型复用校验通过。受管 ComfyUI 启动成功，run_id=`20261114T134803Z-6f169d0d` 的 SDXL 任务约 16 秒完成，PNG 和上传字节 SHA256 一致；随后本仓库把 LoadImage 根目录下拉选项当成完整文件列表，误拒绝 `pipeline/<run_id>/sdxl_00001_.png`，视频任务未提交。该次 E2E 状态为 FAILED，不能记录为视频验收通过。

子目录图片修复新增 9 项离线回归，覆盖真实子目录 PNG、缺失文件/目录、路径越界、模型及采样器选项仍严格匹配；扩展调度恢复测试，验证上传哈希不符时不能提交视频，哈希一致的子目录图片通过真实文件校验后进入视频提交，恢复时不重复生成图片并归档原失败证据。调度测试在视频提交处主动抛出离线标记错误，不生成 GPU 成功证据。实现仅为 LoadImage.image 传入受管 input 根目录，保留 SHA256 校验，不改上游 ComfyUI；后续实例结果见下段。

13:56–13:59 UTC 的用户恢复日志已确认：同一 run_id=`20261114T134803Z-6f169d0d` 复用原 PNG、环境、四个模型和受管服务，LoadImage 子目录路径校验通过；12 节点视频任务被接受，约 80.1 秒完成，MP4 完整解码及元数据验收通过。操作总耗时 196.528 秒，输出 E2E VALIDATION PASSED，用户已观看视频。实机视频路径为 `output/videos/20261114T134803Z-6f169d0d/wan_00001_.mp4`。该实例约 59 GiB 内存，不据此宣称 32 GiB 内存或旧驱动实例均已通过。

tasks 双模式实现新增 23 项离线测试：部分配置继承且各项独立、非法数组/重名/路径/未知参数/错误帧数拒绝、旧单任务完整配置及指纹保持兼容、批量 dry-run 无写入、逐任务 UI/API 工作流不覆盖；按顺序生成独立产物和操作日志、失败停止并恢复剩余项、成功任务重新校验并跳过、子任务已通过而批次未更新时的中断恢复、拒绝配置/模型/成功产物变化及成功状态丢失，以及 CLI 分发和批次报告重建。状态机测试使用显式离线执行器与真实编码的微型合成 PNG/MP4，仅存临时目录，不作为真实 GPU 通过证明。

## 在魔搭执行真实验收

在已经创建的 A10 GPU Notebook 内，项目源码放到 `/mnt/workspace/modelscope-comfyui-pipeline` 后执行：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
bash oneclick.sh
RUNTIME="${WORKSPACE_ROOT:-/mnt/workspace/modelscope-comfyui-pipeline-runtime}"
cat "$RUNTIME/output/reports/e2e_report.md"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASSED" and r["image"]["status"]=="PASSED" and r["video"]["artifact"]["fully_decoded"]' "$RUNTIME/output/reports/e2e_report.json"
```

只有全部通过后，终端才会显示 `E2E VALIDATION PASSED`。请保留该次 `output/reports/<run_id>/`、`logs/`、`state/environments/` 和 `state/active-environment.json`，以便补充实机兼容性和性能结果。测试失败时保留失败报告，不替换为预制产物。

批量验证按 [tasks 操作流程](BATCH_GENERATION.md) 执行，最终需要 `BATCH VALIDATION PASSED` 和 `output/batches/<batch-ID>/batch_report.json` 中所有项为 PASSED；同时保留每项原始报告和操作日志。
