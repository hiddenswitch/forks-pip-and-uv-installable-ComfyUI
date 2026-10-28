import sys

import pytest

from comfy.execution_context import current_execution_context

from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1


@pytest.mark.parametrize("block_installation", [True, False])
def test_pixeloe_loads_without_importing_obsolete_runtime_installer(tmp_path, monkeypatch, block_installation):
    monkeypatch.setitem(current_execution_context().configuration, "block_runtime_package_installation", block_installation)
    package = tmp_path / 'PixelOE'
    nodes = package / 'nodes'
    nodes.mkdir(parents=True)
    (package / '__init__.py').write_text('from .nodes.pixel import NODE_CLASS_MAPPINGS\n')
    (nodes / '__init__.py').write_text('')
    library = package / 'src' / 'pixeloe_test_library'
    library.mkdir(parents=True)
    (library / '__init__.py').write_text('VALUE = 42\n')
    (nodes / 'installer.py').write_text('raise AssertionError("Runtime installer must not be imported")\n')
    (nodes / 'pixel.py').write_text(
        'from .installer import install_pixeloe\n'
        'install_pixeloe()\n'
        'from pixeloe_test_library import VALUE\n'
        'assert VALUE == 42\n'
        'class PixelOE: pass\n'
        'NODE_CLASS_MAPPINGS = {"PixelOE": PixelOE}\n'
    )
    try:
        exported = _vanilla_load_custom_nodes_1(str(package))
        assert 'PixelOE' in exported.NODE_CLASS_MAPPINGS
        assert 'PixelOE.nodes.installer' not in sys.modules
        assert str(package / 'src') not in sys.path
    finally:
        sys.modules.pop('pixeloe_test_library', None)
        for name in list(sys.modules):
            if name == 'PixelOE' or name.startswith('PixelOE.'):
                del sys.modules[name]
