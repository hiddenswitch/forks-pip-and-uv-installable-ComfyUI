import pytest

from comfy_extras.nodes.nodes_logic import BranchNode
from comfy_extras.nodes.nodes_model_merging_model_specific import ModelMergeMOSSImage


def test_branch_requests_only_the_selected_lazy_input():
    branches = {"branch0": None, "branch1": None}
    assert BranchNode.check_lazy_status("second", ["first", "second"], branches) == ["autogrow.branch1"]
    selected = object()
    assert BranchNode.execute("second", ["first", "second"], {"branch1": selected}) == (selected,)


@pytest.mark.parametrize("branch,names,values", [("missing", ["first"], {"branch0": None}), ("first", ["first"], {})])
def test_branch_rejects_missing_selection(branch, names, values):
    with pytest.raises(ValueError):
        BranchNode.check_lazy_status(branch, names, values)


def test_moss_merge_addresses_all_transformer_blocks():
    inputs = ModelMergeMOSSImage.INPUT_TYPES()["required"]
    assert {key for key in inputs if key.startswith("layers.")} == {f"layers.{i}." for i in range(42)}
    assert "text_embedder.blocks.0." in inputs
    assert "text_embedder.blocks.1." in inputs
    assert "final_linear_video." in inputs
