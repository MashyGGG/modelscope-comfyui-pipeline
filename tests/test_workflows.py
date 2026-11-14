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
