# Run workflows from the command line

Execute a workflow without launching the web server:

```console
comfyui workflows list
comfyui run-workflow workflow.json --help
comfyui run-workflow workflow.json --all
comfyui run-workflow workflow.json --prompt "a sunset" --steps 20 --seed 42
```

Pass the workflow to `--help` to inspect its actual parameters. The runner accepts
workflow paths, URLs, template names, JSON, and `-` for standard input. Results
are JSON on stdout; logs go to stderr. UI-format workflows are converted for
execution as well as API-format workflows.

`--all` enables installation of missing custom nodes and downloading of known
models. Gated models still require access and credentials. For an offline
installation, prepare dependencies and model files first and omit `--all`.

Use [the agent guide](../llms.txt) for model selection and worked creative tasks,
[prompting guides](prompting_guides/README.md) for model-specific guidance, or
[embedded Python](embedded.md) to run workflows inside your own application.
