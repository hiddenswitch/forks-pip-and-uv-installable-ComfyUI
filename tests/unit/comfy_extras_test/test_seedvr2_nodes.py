import inspect

from comfy_extras.nodes import nodes_seedvr


def test_seedvr_node_signature_matches_schema():
    for node_cls in (nodes_seedvr.SeedVR2Preprocess, nodes_seedvr.SeedVR2PostProcessing, nodes_seedvr.SeedVR2Conditioning):
        schema_ids = [i.id for i in node_cls.define_schema().inputs]
        exec_params = [
            p for p in inspect.signature(node_cls.execute).parameters.keys()
            if p != "cls"
        ]
        assert schema_ids == exec_params, (
            f"{node_cls.__name__} schema/execute drift: "
            f"schema_ids={schema_ids}, exec_params={exec_params}"
        )
