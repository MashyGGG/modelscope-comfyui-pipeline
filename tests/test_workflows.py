import copy
import json

import pytest

from pipeline.common import ROOT, config
from pipeline.models import specifications
from pipeline.workflows import OUTPUTS, build, to_ui, validate


def test_bundled_workflows_match_builder():
    for name, graph in zip(("01_sdxl_txt2img", "02_wan22_i2v"), build(config(), specifications())):
        assert json.loads((ROOT / f"workflows/api/{name}_api.json").read_text()) == graph
        assert json.loads((ROOT / f"workflows/ui/{name}.json").read_text()) == to_ui(graph)


def test_ui_links_are_bidirectional_and_have_correct_slots():
    for graph in build(config(), specifications()):
        ui = to_ui(graph)
        nodes = {n["id"]: n for n in ui["nodes"]}
        for link_id, src, out, dest, inp, dtype in ui["links"]:
            assert nodes[dest]["inputs"][inp]["link"] == link_id
            assert link_id in nodes[src]["outputs"][out]["links"]
            assert nodes[src]["outputs"][out]["type"] == dtype


@pytest.mark.parametrize("mutation", ["cycle", "slot", "missing"])
def test_invalid_links_fail(mutation):
    graph = build(config(), specifications())[0]
    if mutation == "cycle":
        graph["1"]["inputs"]["loop"] = ["7", 0]
    elif mutation == "slot":
        graph["6"]["inputs"]["vae"] = ["1", 9]
    else:
        graph["6"]["inputs"]["vae"] = ["99", 0]
    with pytest.raises(ValueError):
        validate(graph)


def test_real_image_is_bound_and_native_sampling_present():
    _, b = build(config(), specifications(), "run-a", "pipeline/run-a/result.png")
    assert b["7"]["inputs"]["image"] == "pipeline/run-a/result.png"
    assert b["8"]["inputs"]["start_image"] == ["7", 0]
    assert b["4"]["class_type"] == "ModelSamplingSD3"
    assert b["12"]["inputs"]["codec"] == "h264"


def test_object_info_missing_node_is_error():
    with pytest.raises(ValueError, match="Missing ComfyUI node"):
        validate(build(config(), specifications())[0], {})


def load_image_info():
    # Pinned ComfyUI LoadImage.INPUT_TYPES enumerates only input's direct files.
    return {"LoadImage": {"input": {"required": {
        "image": [["root.png"], {"image_upload": True}],
    }}, "output": ["IMAGE", "MASK"]}}


def test_uploaded_subdirectory_image_is_validated_as_file(tmp_path, capsys):
    from PIL import Image
    image = tmp_path / "pipeline/run-a/result.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 32)).save(image)
    graph = {"7": {"class_type": "LoadImage", "inputs": {"image": "pipeline/run-a/result.png"}}}
    info = load_image_info()
    original = copy.deepcopy(info)
    validate(graph, info, input_dir=tmp_path)
    assert info == original  # Do not inject a path into the server's option list.
    assert "[PASS] LoadImage input file:" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Unavailable option/model"):
        validate(graph, info)  # No file-based exception without a managed input root.


@pytest.mark.parametrize("image", ["pipeline/missing.png", "root.png", "pipeline"])
def test_missing_image_or_directory_rejected(tmp_path, image):
    (tmp_path / "pipeline").mkdir()
    graph = {"7": {"class_type": "LoadImage", "inputs": {"image": image}}}
    with pytest.raises(ValueError, match="LoadImage input file missing"):
        validate(graph, load_image_info(), input_dir=tmp_path)


@pytest.mark.parametrize("absolute", [False, True])
def test_load_image_cannot_escape_input_root(tmp_path, absolute):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    image = outside.as_posix() if absolute else "../outside.png"
    graph = {"7": {"class_type": "LoadImage", "inputs": {"image": image}}}
    with pytest.raises(ValueError):
        validate(graph, load_image_info(), input_dir=input_dir)


@pytest.mark.parametrize("kind,name,valid", [
    ("UNETLoader", "unet_name", "wan.safetensors"),
    ("KSampler", "sampler_name", "euler"),
])
def test_input_directory_does_not_relax_fixed_options(tmp_path, kind, name, valid):
    (tmp_path / "wrong").write_bytes(b"not an option")
    graph = {"1": {"class_type": kind, "inputs": {name: "wrong"}}}
    info = {kind: {"input": {"required": {name: [[valid]]}}, "output": []}}
    with pytest.raises(ValueError, match=f"Unavailable option/model: {kind}"):
        validate(graph, info, input_dir=tmp_path)
