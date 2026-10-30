"""HTTP integration server with the native nodes needed by routing tests."""

import comfy_extras.nodes
from comfy_extras.nodes.nodes_post_processing import PostProcessingExtension
from comfy_extras.nodes.nodes_replacements import register_replacements
from tests.background_server import main
from tests.inference.testing_pack.specific_tests import TestDisabledNode, TestExpandsToDisabledNode


class RoutingExtension(PostProcessingExtension):
    async def on_load(self):
        await register_replacements()


async def comfy_entrypoint():
    return RoutingExtension()


if __name__ == "__main__":
    comfy_extras.nodes.comfy_entrypoint = comfy_entrypoint
    comfy_extras.nodes.NODE_CLASS_MAPPINGS = {
        "TestDisabledNode": TestDisabledNode,
        "TestExpandsToDisabledNode": TestExpandsToDisabledNode,
    }
    main()
