# 单任务提示词细化与原版保留

本次修改的是推理时的正负提示词，沿用仓库现有 SDXL base 1.0、Wan2.2 TI2V 5B 和对应文本编码器，不需要下载新模型或训练 LoRA。图片及视频仍逐阶段真实生成并验收。新版 prompt 尚未在已连接 GPU 上执行，不能把配置可解析理解成画面已达到要求。

## 配置版本

| 文件 | 用途 |
|---|---|
| `config/defaults.yaml` | 原来的完整默认配置，提示词保持不变；原单任务命令继续可用 |
| `config/single.original.yaml` | 微调前 image/video 全部生成参数快照，含原正负提示词和 seed |
| `config/single.refined.yaml` | 细化版单任务，只有四段正负提示词与原版不同 |

这两份 single 配置都不含 tasks。seed、模型、采样器、分辨率、步数和 CFG 保持一致，便于对照提示词变化。环境参数继续继承 defaults.yaml。固定 seed 不能保证不同机器、依赖或硬件下逐像素一致。

## 新版具体细化了什么

用户参考图保留了雪山、湖水、暖光和红船，但未呈现明显帆具；原提示词的主体是 sailboat，因此新版仍采用帆船主题，明确细长木桅杆和一面白色三角帆，而非无帆小艇。

图片 prompt 把主体放在右下前景，增加亮面红船体、倒影、细小涟漪、远岸冷杉林、花岗岩雪峰、山体纵深，以及右侧山脊暖晨光和冷蓝阴影的对比。优先写主体、构图和主要光线，细节词放后面，不混入视频动作或互相冲突的风格。

视频 prompt 描述新 PNG 中的同一艘船：短距离缓慢向右漂移、轻微起伏、白帆受微风轻动，水波及倒影随运动变化；镜头仅采用非常缓慢的平稳向前推进，船始终留在画面内，山体和岸线保持稳定。原版向右横移的镜头改为单一推进镜头；变化已经随原版配置一起保留，可随时对照。

默认片段只有 49 帧、16 FPS，约 3.06 秒，因此不加入跨越整个湖面、绕船一周、快速天气变化或多个转场。负向提示词补充多船、船体/帆面畸变、闪烁、突变镜头等不希望出现的内容；它们是引导，不能保证消除所有问题。

## 模型支持范围与限制

- **SDXL** 接收文本来描述主体和摄影风格，官方示例包含颜色、细节等描述；其模型卡同时说明复杂组合关系和完美写实仍有局限。因此把“单船、桅杆、白帆”等关键需求放在前面，但不承诺船的位置、数量或帆具百分之百准确。[SDXL 官方模型卡](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **Wan2.2 TI2V 5B** 官方支持输入图片加文字描述生成视频，并给出带主体、背景和细节的自然语言示例；本配置采用与输入图一致的动作和镜头描述。[Wan 官方 TI2V 用法](https://github.com/Wan-Video/Wan2.2#run-text-image-to-video-generation)
- 本仓库的 ComfyUI 原生 CLIP 路径支持将较长文本分块编码，不是超过 77 个英文单词就直接拒绝；token 也不等同于单词。但长度增加不等于每个细节都能遵循，仍应保持主体清晰。[固定版本分词与分块实现](https://github.com/Comfy-Org/ComfyUI/blob/b8730510db30c8858e1e5d8e126ef19eac395560/comfy/sd1_clip.py#L484)
- 用户上传的参考图用于本次编写构图描述。下面命令仍从文本重新生成 SDXL PNG，再将该 PNG 交给 Wan，不会自动读取聊天附件或锁定参考图像素。若新 PNG 本身没有正确帆具，优先修正图片提示词或在后续新运行中尝试其他 image.seed，不依赖视频 prompt 补画主体结构。

## 在已验证 GPU 实例执行

```bash
cd /mnt/workspace/modelscope-comfyui-pipeline
git -c http.version=HTTP/1.1 pull --ff-only

# 检查细化后的完整配置及工作流，无写入或推理
bash oneclick.sh --dry-run --config config/single.refined.yaml

# 单次生成一张新 PNG 及对应 MP4，复用现有模型和环境
bash oneclick.sh --gpu-validate --config config/single.refined.yaml
```

已有健康受管服务时，可使用 `bash oneclick.sh --stage e2e --config config/single.refined.yaml`。每次创建新 run_id，图片、视频、配置、API prompt 和报告独立保存，不覆盖原来的运行；报告目录中的 `state.json` 和 `image-prompt.json` / `video-prompt.json` 可核对实际使用的提示词。

需要重新生成原版作对照：

```bash
bash oneclick.sh --gpu-validate --config config/single.original.yaml
```

也可直接打开此前已通过运行的 PNG 和 MP4 对照，无需重跑原版。比较时先看新图是否出现正确船体、桅杆和帆，再看视频是否保持结构、倒影和背景稳定；媒体技术验收通过不等于所有视觉细节都符合 prompt。

## 继续微调与失败恢复

第一次建立自己的版本文件，然后编辑该文件：

```bash
cp config/single.refined.yaml config/single.v2.local.yaml
bash oneclick.sh --dry-run --config config/single.v2.local.yaml
bash oneclick.sh --gpu-validate --config config/single.v2.local.yaml
```

`*.local.yaml` 不进入 Git。每次调整前另存一个版本，只改一类内容（如船、光线或镜头），先固定两阶段 seed 及其余参数。原版和细化版受版本管理，自己的运行目录也记录真实配置。需要不同构图时再改变 image.seed，需要不同视频运动采样时再改变 video.seed，不同时改所有参数。

如果只修改 video.prompt，当前单任务流程仍会重新生成一张 PNG 和对应视频；尚无“固定已有图片，只试不同视频 prompt”的入口。已通过的运行不能 `--resume`；更换 prompt 后应新建运行。

本次细化配置运行失败时，保持同一配置恢复：

```bash
bash oneclick.sh --gpu-validate --config config/single.refined.yaml --resume <失败时打印的run_id>
```

该恢复仅继续原配置的失败阶段；图片已通过时会核对原图后复用。原版运行不能带细化配置恢复，两者哈希不同。整条链路仍以 `E2E VALIDATION PASSED` 为技术通过标志。
