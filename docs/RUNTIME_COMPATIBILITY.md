# A10 550 驱动与跨 Python 版本兼容说明

## 已观察到的问题

用户实例中，CPU 使用 Python 3.11，GPU 使用 Python 3.12.13；GPU 是 A10、23028 MiB 显存、驱动 550.54.15。CPU 已完成四个模型下载和 SHA256 校验。旧代码在真正执行 CUDA 运算前，因驱动低于固定 560.35.03 门槛退出；共享的旧 Python 3.11 虚拟环境随后也不能直接供 Python 3.12 使用。

## 修复后的驱动判断

| 驱动版本 | 系统预检结果 | 后续要求 |
|---|---|---|
| <525.60.13 | FAIL | 不进入 cu126 安装/推理 |
| >=525.60.13 且 <560.35.03 | WARN | 小版本兼容候选，必须执行真实 CUDA 运算和完整模型 E2E |
| >=560.35.03 | PASS | 仍必须执行真实 CUDA 运算和完整模型 E2E |

CUDA 12.x 的 Linux 小版本兼容最低驱动为 525.60.13，但 PTX 或需要较新驱动的功能存在限制。550.54.15 不再被预检直接拦截，也不被当作已验证兼容。依据：[CUDA 12.6 发布说明](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)、[小版本兼容限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

项目仍固定安装 torch 2.8.0/cu126。CUDA 门槛会实际执行 FP16 矩阵乘法、卷积、缩放点积注意力，以及设备支持时的 BF16 矩阵乘法，并同步检查错误。失败会停止，不回退到 CPU 采样。小运算通过仍不能替代 SDXL/Wan 的完整真实推理与 MP4 验收。

## 跨 Python 版本处理

新版在同一 WORKSPACE_ROOT 下使用独立环境：

```text
venv/                                  # 旧版环境保留，不删除或搬动
venvs/py3.11/                          # 新版 Python 3.11 环境
venvs/py3.12/                          # 新版 Python 3.12 环境
state/environments/py3.12.json          # 只属于该 Python 版本的安装验证
state/environments/py3.12-pip-freeze.txt
state/active-environment.json          # 最近通过安装验证的环境
ComfyUI/                               # 固定源码和模型链接继续共享
model-cache/                           # 已校验模型继续共享
```

每个环境在复用前检查 Python 版本、sys.prefix、源码 commit 和依赖指纹。GPU 验证在当前环境内执行，不执行旧 Python 3.11 环境，也不会把它的安装标记当成 Python 3.12 已就绪的证据。

Python 官方将虚拟环境视为依赖基础解释器的环境，建议在目标位置重新创建，见 [venv 文档](https://docs.python.org/3/library/venv.html)。同一 Python 版本下如果更换了不兼容镜像或人工搬动环境，程序仍可能明确失败；可选择新运行根目录，通过模型包迁移权重并重建。

## 已完成 CPU 准备的用户如何继续

在 GPU 实例中，保留原 WORKSPACE_ROOT、模型缓存和旧环境，进入原项目目录执行：

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
git -c http.version=HTTP/1.1 pull --ff-only
bash oneclick.sh --gpu-validate
```

预期日志：

```text
[WARN] NVIDIA driver: 550.54.15; CUDA 12.x minor-version compatibility candidate ...
[INFO] Selected Python 3.12 runtime: .../venvs/py3.12 ...
[INFO] Legacy environment preserved ...
[PASS] ComfyUI source reused ...
```

首次会为 Python 3.12 下载/安装依赖，需要额外磁盘与网络；不会重新下载已完整且哈希正确的模型。读取约 25 GB 权重校验哈希可能需要一段时间，日志会逐个显示正在检查的模型。

安装完成后可单独检查实际 CUDA 运算：

```bash
bash scripts/check_environment.sh --require-cuda
```

若仍出现 `cudaErrorCallRequiresNewerDriver`、不支持的 PTX 或其他 CUDA 错误，日志应作为真实不兼容证据；需由平台提供兼容的驱动环境。脚本不升级宿主驱动，也不会以降低检查阈值冒充成功。最终通过标志仍是 `E2E VALIDATION PASSED`。

本次改动的驱动策略、环境隔离、旧数据保留和失败门槛已做离线测试；真实 A10 550.54.15 上的 CUDA 运算及模型推理尚待实例重试验证。
