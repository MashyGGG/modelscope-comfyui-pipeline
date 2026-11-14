from copy import deepcopy
import json
from pathlib import Path

from .common import ROOT, atomic_json, paths


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def build(c, items, run_id="preview", image="generated.png"):
    names = {m["key"]: Path(m["file"]).name for m in items}
    i, v = c["image"], c["video"]
    a = {
        "1": node("CheckpointLoaderSimple", ckpt_name=names["sdxl"]),
        "2": node("CLIPTextEncode", clip=["1", 1], text=i["prompt"]),
        "3": node("CLIPTextEncode", clip=["1", 1], text=i["negative"]),
        "4": node("EmptyLatentImage", width=i["width"], height=i["height"], batch_size=1),
        "5": node("KSampler", model=["1", 0], positive=["2", 0], negative=["3", 0], latent_image=["4", 0],
                  seed=i["seed"], steps=i["steps"], cfg=i["cfg"], sampler_name=i["sampler"], scheduler=i["scheduler"], denoise=1),
        "6": node("VAEDecode", samples=["5", 0], vae=["1", 2]),
        "7": node("SaveImage", images=["6", 0], filename_prefix=f"images/{run_id}/sdxl"),
    }
    b = {
        "1": node("UNETLoader", unet_name=names["wan"], weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=names["umt5"], type="wan", device="cpu"),
        "3": node("VAELoader", vae_name=names["vae"]),
        "4": node("ModelSamplingSD3", model=["1", 0], shift=v["shift"]),
        "5": node("CLIPTextEncode", clip=["2", 0], text=v["prompt"]),
        "6": node("CLIPTextEncode", clip=["2", 0], text=v["negative"]),
        "7": node("LoadImage", image=image),
        "8": node("Wan22ImageToVideoLatent", vae=["3", 0], width=v["width"], height=v["height"],
                  length=v["frames"], batch_size=1, start_image=["7", 0]),
        "9": node("KSampler", model=["4", 0], positive=["5", 0], negative=["6", 0], latent_image=["8", 0],
                  seed=v["seed"], steps=v["steps"], cfg=v["cfg"], sampler_name=v["sampler"], scheduler=v["scheduler"], denoise=1),
        "10": node("VAEDecode", samples=["9", 0], vae=["3", 0]),
        "11": node("CreateVideo", images=["10", 0], fps=v["fps"]),
        "12": node("SaveVideo", video=["11", 0], filename_prefix=f"videos/{run_id}/wan", format="mp4", codec="h264"),
    }
    if c["vae_tiled"]:
        b["10"] = node("VAEDecodeTiled", samples=["9", 0], vae=["3", 0], tile_size=512, overlap=64,
                       temporal_size=16, temporal_overlap=4)
    validate(a)
    validate(b)
    return a, b


OUTPUTS = {
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"], "CLIPTextEncode": ["CONDITIONING"],
    "EmptyLatentImage": ["LATENT"], "KSampler": ["LATENT"], "VAEDecode": ["IMAGE"],
    "VAEDecodeTiled": ["IMAGE"], "SaveImage": [], "UNETLoader": ["MODEL"], "CLIPLoader": ["CLIP"],
    "VAELoader": ["VAE"], "ModelSamplingSD3": ["MODEL"], "LoadImage": ["IMAGE", "MASK"],
    "Wan22ImageToVideoLatent": ["LATENT"], "CreateVideo": ["VIDEO"], "SaveVideo": [],
}


def validate(graph, info=None):
    visited, visiting = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("Workflow contains a cycle")
        if key in visited:
            return
        visiting.add(key)
        n = graph[key]
        kind = n.get("class_type")
        if not kind or not isinstance(n.get("inputs"), dict):
            raise ValueError(f"Invalid API node {key}")
        if info is not None:
            if kind not in info:
                raise ValueError(f"Missing ComfyUI node: {kind}")
            schema = info[kind]["input"]
            required = schema.get("required", {})
            allowed = {**required, **schema.get("optional", {})}
            if set(required) - n["inputs"].keys():
                raise ValueError(f"Missing inputs on {key}/{kind}: {set(required) - n['inputs'].keys()}")
            if n["inputs"].keys() - allowed.keys():
                raise ValueError(f"Unknown inputs on {key}/{kind}")
        for name, value in n["inputs"].items():
            if isinstance(value, list):
                if len(value) != 2 or str(value[0]) not in graph or not isinstance(value[1], int) or value[1] < 0:
                    raise ValueError(f"Broken link: {key}.{name}")
                src = graph[str(value[0])]["class_type"]
                outputs = info[src]["output"] if info else OUTPUTS[src]
                if value[1] >= len(outputs):
                    raise ValueError(f"Bad output slot: {key}.{name}")
                if info:
                    expected = allowed[name][0]
                    if isinstance(expected, str) and expected != outputs[value[1]]:
                        raise ValueError(f"Link type mismatch: {key}.{name}")
                visit(str(value[0]))
            elif info:
                spec = allowed[name]
                if isinstance(spec[0], list) and value not in spec[0]:
                    raise ValueError(f"Unavailable option/model: {kind}.{name}={value}")
        visiting.remove(key)
        visited.add(key)
    for key in graph:
        visit(key)


def to_ui(graph):
    nodes, links = [], []
    consumers = {key: [[] for _ in OUTPUTS[n["class_type"]]] for key, n in graph.items()}
    for index, (key, n) in enumerate(graph.items()):
        inputs, widgets = [], []
        for name, value in n["inputs"].items():
            if isinstance(value, list):
                source, slot = str(value[0]), value[1]
                link_id = len(links) + 1
                dtype = OUTPUTS[graph[source]["class_type"]][slot]
                links.append([link_id, int(source), slot, int(key), len(inputs), dtype])
                inputs.append({"name": name, "type": dtype, "link": link_id})
                consumers[source][slot].append(link_id)
            else:
                widgets.append(value)
                if n["class_type"] == "KSampler" and name == "seed":
                    widgets.append("fixed")
                if n["class_type"] == "LoadImage" and name == "image":
                    widgets.append("image")
        nodes.append({"id": int(key), "type": n["class_type"], "pos": [(index % 4)*380, (index // 4)*340],
                      "size": [340, 280], "flags": {}, "order": index, "mode": 0,
                      "inputs": inputs, "outputs": [], "properties": {"Node name for S&R": n["class_type"]},
                      "widgets_values": widgets})
    for n in nodes:
        n["outputs"] = [{"name": dtype, "type": dtype, "links": consumers[str(n['id'])][slot]}
                        for slot, dtype in enumerate(OUTPUTS[n["type"]])]
    return {"last_node_id": max(map(int, graph)), "last_link_id": len(links), "nodes": nodes,
            "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


def prepare(c, items, destination=None):
    destination = Path(destination or paths(c)["comfy"] / "user/default/workflows/modelscope-pipeline")
    a, b = build(c, items)
    for name, graph in (("01_sdxl_txt2img", a), ("02_wan22_i2v", b)):
        atomic_json(destination / f"{name}.json", to_ui(graph))
        atomic_json(paths(c)["base"] / "workflows/api" / f"{name}_api.json", graph)
    print("[PASS] Workflow Deployment", flush=True)
