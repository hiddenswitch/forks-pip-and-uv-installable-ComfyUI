import pytest

from comfy.cli_args import default_configuration
from comfy.cmd.folder_paths import init_default_paths
from comfy.component_model.folder_path_types import FolderNames
from comfy.execution_context import context_configuration, context_folder_names_and_paths
from comfy_api import feature_flags


@pytest.fixture(autouse=True)
def folder_names(tmp_path):
    config = default_configuration()
    config.offline = False
    config.disable_all_custom_nodes = True
    names = FolderNames(base_paths=[tmp_path], is_root=False)
    init_default_paths(names, config, base_paths_from_configuration=False)
    flags = feature_flags.SERVER_FEATURE_FLAGS.copy()
    with context_configuration(config), context_folder_names_and_paths(names):
        yield names
    feature_flags.SERVER_FEATURE_FLAGS.clear()
    feature_flags.SERVER_FEATURE_FLAGS.update(flags)
