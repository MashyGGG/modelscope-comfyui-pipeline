# 单任务与 tasks 批量生成

保留原有单任务入口，并新增 tasks 数组入口。一次任务固定为 SDXL 文生图 → 上传同一张 PNG 并核对 SHA256 → Wan 图生视频 → MP4 完整解码验收。批量模式在同一 GPU 上依次执行任务，复用环境、模型和服务。

## 1. 更新源码并选择模式

在已经验证成功的 GPU 实例中：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
git -c http.version=HTTP/1.1 pull --ff-only
```

`config/defaults.yaml` 保留现有 `image`、`video`，没有 tasks，执行以下命令仍只生成一组图片和视频：

```bash
bash oneclick.sh --gpu-validate
```

服务已经启动且健康时，可使用 `bash oneclick.sh --stage e2e` 直接创建下一次生成。更换 prompt 或 seed 后创建新运行，不使用已通过运行的 `--resume`。原有单任务失败恢复仍使用原 run_id。

需要更细的单任务场景和动作描述时，使用 `--config config/single.refined.yaml`；微调前配置单独保存在 `config/single.original.yaml`，默认 prompt 不变。细节、模型支持边界及对照操作见 [单任务 prompt 微调](PROMPT_TUNING.md)。

## 2. 配置 tasks 数组

首次复制示例一次，之后编辑自己的文件：

```bash
cp config/batch.example.yaml config/batch.local.yaml
```

`*.local.yaml` 已被 Git 忽略，更新仓库不会覆盖自己的任务文件。示例以 JSON 子集编写，它也是合法 YAML；保留这一格式即可使用标准库解析，不要求系统 Python 额外安装 PyYAML。普通 YAML 格式需要启动脚本使用的 Python 已有 PyYAML。

配置结构如下（为便于阅读使用普通 YAML 展示，仓库实际示例使用 JSON 子集）：

```yaml
image:
  steps: 24
video:
  frames: 49
tasks:
  - name: lake
    image:
      prompt: "A red sailboat on an alpine lake, cinematic photography"
      seed: 101
    video:
      prompt: "The sailboat moves slowly, camera pans right"
      seed: 201
  - name: lighthouse
    image:
      prompt: "A white lighthouse at sunset, cinematic photography"
      seed: 102
    video:
      prompt: "Waves roll gently, camera slowly moves forward"
      seed: 202
```

- `--config` 文件可仅包含 tasks，环境、模型位置、顶层 image/video 未指定参数继承 `config/defaults.yaml`。
- 每项 `name` 必须唯一，为 1–64 个 ASCII 字母、数字、下划线或短横线，且首字符为字母或数字。name 用于关联日志及目录。
- 每项只接受 `name`、`image`、`video`。image/video 中可覆盖现有生成字段：prompt、negative、seed、steps、cfg、尺寸、sampler、scheduler；video 另有 frames、fps、shift。环境、端口、GPU、模型和 `vae_tiled` 是批次公共设置。
- image/video 覆盖项可以省略，沿用公共默认值。任务字段最终覆盖顶层生成默认值，包括顶层 `--set image.*` / `video.*`；需要指定某一项时直接编辑任务，不使用 `--set tasks.0.*`。
- tasks 必须为非空数组，`tasks: []` 不会退回单任务。需要单任务时不提供 tasks。所有任务的结构和本地参数约束在执行前验证；节点、模型和采样器选项还需运行中的 ComfyUI 校验，dry-run 不代替该检查。
- 图片宽高为 8 的倍数，视频宽高为 32 的倍数，frames 为至少 5 的 4n+1；默认视频 832×480、49 帧、16 FPS。各项可以不同，验收按其有效参数执行。

微调提示词时，复制数组项并换一个 name，保留 seed，只修改 prompt 以便比较；需要相同提示词的不同版本时，为各项配置不同 image.seed。video.prompt 描述对应图片主体的动作和镜头运动；项目不会根据结果自动改写提示词。

## 3. 预览并执行

```bash
# 只检查配置并打印每项 prompt、seed 和尺寸，不创建目录或执行推理
bash oneclick.sh --dry-run --config config/batch.local.yaml

# 已准备好模型的 GPU 实例，检查并复用现有安装/模型/服务
bash oneclick.sh --gpu-validate --config config/batch.local.yaml
```

第一次在全新 GPU 实例安装及下载模型，可执行 `bash oneclick.sh --config config/batch.local.yaml`。两个实例准备方式仍可使用 `--cpu-prepare`，CPU 不生成任何图片或视频；最后在 GPU 端指定批量配置执行 `--gpu-validate`。

已有健康受管服务且只需生成时，也可以执行：

```bash
bash oneclick.sh --stage e2e --config config/batch.local.yaml
```

任务按数组顺序串行执行，失败时立即停止，后续项保持 PENDING。每项通过可看到 `E2E VALIDATION PASSED`；只有整批任务都通过，才会输出 **`BATCH VALIDATION PASSED`**。单项通过不表示整批通过。

## 4. 查看文件、日志和报告

```bash
RUNTIME="${WORKSPACE_ROOT:-/mnt/workspace/modelscope-comfyui-pipeline-runtime}"
cat "$RUNTIME/output/batches/batch_report.md"
ls -lt "$RUNTIME/logs"/task-*.log
```

批次使用独立 batch-ID，每个任务预先分配独立 run_id，文件结构为：

```text
$RUNTIME/
  logs/gpu-validate-<操作ID>.log/.json    # 整次操作和阶段日志
  logs/task-<name>-<操作ID>.log/.json     # 逐任务执行或恢复校验日志
  output/images/<run_id>/                # 该任务的 PNG
  output/videos/<run_id>/                # 该任务的 MP4
  output/reports/<run_id>/               # 单任务状态、工作流、history 和报告
  output/batches/<batch-ID>/
    state.json                          # 各项状态、run_id、日志位置和已通过产物
    batch_report.json / batch_report.md  # 批次汇总
    configs/<name>.json                  # 每项执行时合并得到的完整配置
    attempt-<时间>/                      # 恢复前的旧批次证据
```

`output/batches/batch_report.*` 是最近批次汇总副本；`output/reports/e2e_report.*` 仍指向最近执行的单任务报告。每个批次/任务原目录保留，不覆盖历史文件。UI/API 预览工作流也按任务名分别部署，不互相覆盖。

## 5. 失败恢复

保持原任务文件、公共设置、运行目录及模型清单，用终端打印的 **batch-ID** 恢复整个批次：

```bash
bash oneclick.sh --gpu-validate --config config/batch.local.yaml --resume <batch-ID>
```

服务仍健康时可使用 `--stage e2e` 替代 `--gpu-validate`。恢复时归档旧批次报告；逐个核对成功任务的配置/模型、PNG/MP4 哈希及媒体验收，输出 `[SKIP] Task ...` 并跳过推理。失败任务继续原 run_id，原 PNG 已通过时校验后复用；之后执行尚未开始的项。每次任务恢复和跳过校验均另存日志。

如果进程在单任务已保存 PASSED、批次汇总尚未更新时中断，恢复会识别已通过的单任务记录，校验后跳过。配置/模型变化、成功文件被修改或成功记录丢失时拒绝继续，不自动覆盖原产物。修改 prompt、seed、任务顺序或数量后应去掉 `--resume`，创建新批次。

重新生成批次汇总（不执行推理）可使用：

```bash
bash oneclick.sh --stage report --config config/batch.local.yaml --resume <batch-ID>
```

单任务真实 GPU 通过已有用户实例日志；新增批量调度目前通过离线回归，仍需按本流程在 GPU 实例验证，不能以离线合成 MP4 作为批量生成已通过的证据。
