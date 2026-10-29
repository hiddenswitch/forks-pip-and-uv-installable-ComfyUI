# Workflows, models, and headless execution

Start the editor with `comfyui serve`, open `http://localhost:8188`, and save a
workflow as JSON. The fork accepts UI-format graphs and API-format node mappings
in its headless runner and embedded client. Native text generation and HDR media
nodes are also part of ComfyUI upstream; they are available here alongside the
fork's packaging and execution tools.

## Find a workflow and inspect its inputs

```console
comfyui workflows list
comfyui run-workflow workflow.json --help
comfyui workflows params workflow.json --all
```

You can supply a template name, local file, literal JSON, `-` for stdin, or a
supported URI such as HTTPS, `hf://`, or `s3://`. Workflow-specific help displays
author notes and discovered inputs. It may need access to the workflow source;
the available flags depend on its nodes and metadata.

## Run without a server

```console
comfyui run-workflow workflow.json --prompt "a small observatory in a snowy forest" --seed 42 --steps 20
comfyui run-workflow first.json second.json > results.jsonl
comfyui run-workflow - < workflow.json
```

The runner executes and exits. Each workflow's outputs are written as a JSON
line to stdout; normal application logs go to stderr. Output fields depend on
the workflow's output nodes. Read those results rather than guessing generated
filenames. Custom nodes that print directly to stdout can interfere with piping.

## Parameters

| Option | Use |
|---|---|
| `--prompt`, `--negative-prompt` | Override supported text conditioning |
| `--seed`, `--steps`, `--cfg` | Override supported sampling inputs |
| `--width`, `--height`, `--batch-size` | Change supported latent/image dimensions and batch inputs |
| `--quantity` | Execute repeated jobs; distinct from a latent batch |
| `--checkpoint`, `--diffusion-model` | Select a compatible loader's model |
| `--image`, `--video`, `--audio` | Supply media inputs where supported |
| `--set` | Set a specific node input |
| `--add-lora` | Insert a compatible LoRA into supported model/CLIP paths |
| `--compile` | Enable compilation on supported models; includes warmup cost |

Use workflow help to confirm applicable inputs. For an API workflow with node `6`:

```console
comfyui run-workflow workflow.json --set '6.inputs.text=a lighthouse at dusk'
```

LoRA arguments accept `name[:model_strength[:clip_strength]]`, including local
names and supported model URIs. Match the LoRA's architecture to the base model;
similar product names do not establish compatibility. Compilation and alternative
attention kernels can help some workloads but have no universal speedup.

An explicit seed supports reproducible comparisons. Repeated jobs increment a
provided base seed; UI workflow seed controls can also affect repetition. Record
the workflow, model revisions, package versions, and overrides together.

## Prepare missing dependencies

```console
comfyui run-workflow workflow.json --dry-run
comfyui run-workflow workflow.json --all
```

The dry run reports planned node installations and model downloads without
inference. It can still read remote workflow/registry metadata. `--all` installs
missing packages, discovers existing models, and downloads resolvable model
requirements. Registry metadata and author notes are not exhaustive: unresolved
nodes or files need explicit setup. Run preparation before an external distributed
launcher; `--all` and `--dry-run` are rejected inside that launch.

## Models and existing files

Reuse a workspace or register folders instead of duplicating weights:

```console
comfyui serve --cwd /path/to/ComfyUI
comfyui run-workflow workflow.json --add-model-folder-path checkpoints=/srv/models/checkpoints
comfyui models --help
```

`--extra-model-paths-config` accepts an existing model-path configuration.
The `models` commands support discovery and lookup; consult their help for
provider-specific search options. Automatic discovery uses available OS indexes
and filesystem fallbacks, so explicit paths are useful when a directory is not
indexed or cannot be classified.

Known model filenames can resolve through the download registry. Model sources
also support URLs and fsspec-backed storage where the relevant loader/download
path supports them. Access to object storage requires its backend and credentials.
This does not imply that every custom-node file input accepts every URI scheme.

For gated Hugging Face repositories, accept the model's terms and authenticate
with `hf auth login` or `HF_TOKEN`. Authentication does not bypass access approval.
Use `--disable-known-models` to disable known-model automatic resolution; prepare
all assets and dependencies separately for offline execution. See
[registering models](embedded.md#adding-known-models-for-automatic-download).

## Media and prompting

- [Text-generation workflows](llm.md)
- [SVG conversion and string saving](other_features.md)
- [Vendor prompting guides](prompting_guides/README.md), with source attribution

Select loaders and output nodes appropriate to your data. HDR/high-bit-depth
support depends on the image format, installed codec, and node; a PNG preview does
not establish that the saved artifact retained HDR data. Vendor documentation may
describe hosted APIs whose parameters or models are unavailable in local nodes.

## Repeated generation and agents

For a batch, reuse a single [embedded client](embedded.md#running-multiple-workflows)
or a persistent server to retain eligible models and execution caches between
prompts. A new CLI process for each item pays initialization and loading costs again.
Model parallelism and memory tuning are covered in [performance](distributed.md);
multiple workers for throughput are covered in [deployment](deployment.md).

[llms.txt](../llms.txt) provides the agent playbook: inspect the environment and
workflow, prepare dependencies, generate, inspect the artifact, and report the
settings needed to reproduce it.
