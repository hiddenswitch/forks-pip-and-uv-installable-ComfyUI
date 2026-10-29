# What this fork adds to ComfyUI

Use this fork when you want ComfyUI’s workflow ecosystem with straightforward
package installation, automation, embedded execution, and integrated multi-GPU
inference. You do not need to redesign ordinary workflows to use these features.

| Task | What this fork provides | Start here |
|---|---|---|
| Install and upgrade | A distributable Python package and `comfyui` CLI; supported custom nodes and extension wheels through the package index | [Installation](installing.md) |
| Automate generation | Headless execution, workflow-specific CLI options, JSON results, optional model and node installation | [Run workflows](workflows.md) |
| Use several GPUs | Integrated TP, PP, and sequence parallelism for supported models | [Multi-GPU](distributed.md) |
| Avoid repeated transfers | Resident-model reuse, LoRA residency improvements, quantized memory accounting, and container-aware host-memory budgets | [Configuration](configuration.md) |
| Embed workflows | Python client with async execution, previews, and configurable isolation | [Embedded Python](embedded.md) |
| Install ecosystem nodes | Package facade plus compatibility loading for ordinary custom nodes | [Custom nodes](custom_nodes.md) |
| Serve more requests | RabbitMQ workers separated from the API frontend | [Workers](workers.md) |
| Connect application inputs | Request-parameter nodes, URL inputs, and output handling for remote applications | [API](api.md) |

## Compatibility and shared capabilities

Upstream already supplies the workflow editor, broad model support, DynamicVRAM,
native text generation, and many high-bit-depth/HDR media features. This fork
extends that foundation; those capabilities alone are not exclusive differences.
Raster/SVG conversion and the application integration paths complement it.

Custom nodes still need compatible dependencies and frontend APIs. Multi-GPU
support depends on the model, backend, and topology; two GPUs do not universally
halve generation time. Check the linked guides before choosing an installation
or deployment configuration.
