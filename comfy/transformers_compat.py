import sys
import types


def patch_transformers_finegrained_fp8_import(torch_module=None):
    if torch_module is None:
        import torch as torch_module

    if hasattr(torch_module, "float8_e8m0fnu"):
        return

    module_name = "transformers.integrations.finegrained_fp8"
    if module_name in sys.modules:
        return

    # transformers 5.10 imports integrations.finegrained_fp8 from
    # modeling_utils and unconditionally reads torch.float8_e8m0fnu.
    # NVIDIA's torch 2.7.0a0 nv25.03 build does not expose it. Leave
    # torch untouched and disable the optional FP8 experts integration
    # so ordinary transformers imports still work.
    module = types.ModuleType(module_name)
    module.ALL_FP8_EXPERTS_FUNCTIONS = {}
    sys.modules[module_name] = module


patch_transformers_finegrained_fp8_import()

from transformers import PreTrainedTokenizerFast

try:
    from transformers import CLIPTokenizer
except (ImportError, ModuleNotFoundError):
    try:
        from transformers import CLIPTokenizerFast as CLIPTokenizer
    except (ImportError, ModuleNotFoundError):
        CLIPTokenizer = object

try:
    from transformers import PreTrainedTokenizerBase
except (ImportError, ModuleNotFoundError):
    PreTrainedTokenizerBase = object

try:
    from transformers import T5TokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import T5Tokenizer as T5TokenizerFast

try:
    from transformers import LlamaTokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import LlamaTokenizer as LlamaTokenizerFast

try:
    from transformers import CLIPTokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import CLIPTokenizer as CLIPTokenizerFast

try:
    from transformers import GPT2TokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import GPT2Tokenizer as GPT2TokenizerFast

try:
    from transformers import BertTokenizerFast
except (ImportError, ModuleNotFoundError):
    from transformers import BertTokenizer as BertTokenizerFast

try:
    from transformers import Qwen2TokenizerFast
except (ImportError, ModuleNotFoundError):
    try:
        from transformers import Qwen2Tokenizer as Qwen2TokenizerFast
    except (ImportError, ModuleNotFoundError):
        # Fallback if neither exists, primarily for earlier versions or specific environments
        Qwen2TokenizerFast = None

# Alias Qwen2Tokenizer to the "Fast" version we found/aliased, as we might use either name
Qwen2Tokenizer = Qwen2TokenizerFast

try:
    from transformers import ByT5TokenizerFast
except ImportError:
    try:
        from transformers import ByT5Tokenizer as ByT5TokenizerFast
    except (ImportError, ModuleNotFoundError):
        ByT5TokenizerFast = None

ByT5Tokenizer = ByT5TokenizerFast

__all__ = [
    "T5TokenizerFast",
    "LlamaTokenizerFast",
    "CLIPTokenizer",
    "CLIPTokenizerFast",
    "PreTrainedTokenizerBase",
    "PreTrainedTokenizerFast",
    "GPT2TokenizerFast",
    "BertTokenizerFast",
    "Qwen2Tokenizer",
    "Qwen2TokenizerFast",
    "ByT5Tokenizer",
    "patch_transformers_finegrained_fp8_import",
]
