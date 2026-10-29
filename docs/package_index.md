# Host a custom-node package index

### Self-Hosting the Package Index

The `serve-pip` command runs a local PEP 503 "simple" index backed by the Comfy registry. The facade builds wheels on demand from ComfyUI-Manager registry metadata, injects dependency metadata, and loads the actual nodes through the existing vanilla custom-node mitigation layer.

```bash
comfyui serve-pip --listen 0.0.0.0 --port 8190
```

Then install from it:

```bash
uv pip install --extra-index-url http://localhost:8190/simple/ comfyui-wanvideowrapper
```

The registry snapshot (`--pip-facade-snapshot-uri`), the wheel cache
(`--pip-facade-cache-prefix`) and the `snapshot-pip-registry` output are fsspec
URLs, all opened the same way: a local path, `file://`, `s3://`, `gs://`,
`memory://` or any other fsspec filesystem.

```bash
comfyui serve-pip --pip-facade-cache-prefix=/tmp/comfyui-pip-facade
comfyui serve-pip --pip-facade-cache-prefix=s3://my-bucket/comfyui/pip-facade
```

Backend options come from fsspec's configuration: `FSSPEC_<PROTOCOL>_<OPTION>`
environment variables or an fsspec config file. For an S3-compatible gateway
such as SeaweedFS:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
FSSPEC_S3_ENDPOINT_URL=http://seaweedfs-s3:8333 \
comfyui serve-pip \
  --pip-facade-snapshot-uri=s3://my-bucket/registry-snapshot.db \
  --pip-facade-snapshot-max-age-seconds=2100 \
  --pip-facade-cache-prefix=s3://my-bucket/comfyui/pip-facade
FSSPEC_S3_ENDPOINT_URL=http://seaweedfs-s3:8333 \
comfyui snapshot-pip-registry --pip-facade-snapshot-overwrite s3://my-bucket/registry-snapshot.db
```

`--pip-facade-cache-s3-endpoint-url` is equivalent to `FSSPEC_S3_ENDPOINT_URL`.

serve-pip reads the snapshot into memory and checks its source for a new
generation every 30 seconds in the background. Requests and the `/livez`,
`/readyz` probes are answered without touching storage, so a slow filesystem
delays only new generations and cache misses. While the source is unreachable
the last generation is served until it is older than
`--pip-facade-snapshot-max-age-seconds`. Each snapshot replaces the previous one
in one step, and each cache entry is published whole: a local directory through
a temporary file and a rename, an object store as one completed upload.

Use `--pip-facade-only-known-nodes` to restrict to tested nodes. Use `snapshot-pip-registry` to pre-generate a sqlite snapshot for faster startup.

The installed wheel is only a transport wrapper. It exposes a `comfyui.custom_nodes` entry point, points ComfyUI at the vendored custom-node directory, and then the node is imported as a vanilla custom node with the same compatibility shims used for `custom_nodes/` clones.

### Local Facade Build for Long-Tail Packages

The bundled `pip_facade_registry_snapshot.sqlite.xz` carries the `class_type → canonical_name` mapping for registry packages; the public index builds and caches wheels on demand. When `comfyui run-workflow ... --all` resolves a workflow's custom-node requirements and a package isn't on the index, the CLI falls back to building a facade wheel **locally in-process** rather than installing from git directly:

1. `comfy.custom_node_facade.repo_lookup.resolve_package_repo_url(canonical_name)` looks up the source repository URL via comfy.org's node registry (`api.comfy.org/nodes/<name>`), then ComfyUI-Manager's `extension-node-map.json` as a second source.
2. `comfy.custom_node_facade.local_build.build_local_facade_wheel(canonical_name)` synthesizes a one-shot `FacadeRegistry`, points `FacadeWheelBuilder` at the GitHub or GitLab archive URL (`https://api.github.com/repos/<owner>/<repo>/zipball` or `https://gitlab.com/api/v4/projects/<encoded>/repository/archive.zip`), and produces the same kind of wheel the remote facade serves.
3. `uv pip install <wheel-path>` installs it.

This goes through the same `_appmana_facade_<name>` rename and stub-shim path as a remotely-built wheel, so ComfyUI's node loader picks it up without any special-casing. Examples that resolve via this path: `comfyui-umeairt-toolkit` (GitLab), `daxnodes` (GitHub but no wheel on the index), `comfyui-fbcnn`, `comfyui-image-saver`. Direct installers are not used because they would skip the rename and break ComfyUI's vanilla-custom-node import path.

Cache lives at the configured ComfyUI cache location; rebuild it through the
facade cache command rather than installing the repository directly.

---

### How the Package Index Works

The `serve-pip` command exposes vanilla custom nodes as PEP 503 Python packages. It:

- enumerates installable custom nodes from ComfyUI-Manager's registry data
- resolves versions from the Comfy registry (`api.comfy.org`)
- injects known missing dependencies from this fork's compatibility tables
- builds wheels on demand
- publishes `comfyui.custom_nodes` entry points that point back to vendored vanilla custom-node directories

Each generated wheel contains the upstream custom node repository contents, generated `METADATA` dependencies merged from registry metadata, and a `comfyui.custom_nodes` entry point. At runtime, the entry point advertises the vendored custom-node directory to `comfy.nodes.package`, and ComfyUI imports the vendored repo through the vanilla custom-node importer. Facade-installed nodes behave like ecosystem custom nodes, not like native packaged extensions.

For nodes without published versions on the Comfy Node Registry, `inject_version` in `comfy/component_model/node_registry.py` generates a synthetic version from the GitHub archive. URL dependencies (like `sam2 @ git+https://...`) are rewritten to plain package names using `packaging.requirements.Requirement`.

Use `snapshot-pip-registry` to pre-generate a sqlite snapshot of the full registry for faster startup. It writes to any fsspec URL, and a separate process can update it periodically; serve-pip picks up each new generation.

#### Wheel cache

Built wheels are cached at `{cache-prefix}/v{revision}/{project}/{filename}.whl`.
`--pip-facade-cache-revision` selects the cache revision.

#### Dependency rewriting

The facade rewrites certain dependencies in generated wheel metadata:

- **Stripped versions**: `numpy`, `jax`, `jaxlib`, `image-reward`, `timm` are emitted without version constraints so the resolver picks the latest compatible version.
- **opencv homogenization**: All opencv variants (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`, `opencv-contrib-python-headless`) are rewritten to bare `opencv-contrib-python-headless`.
- **onnxruntime platform expansion**: `onnxruntime` (and all variants) is expanded to platform-conditional dependencies: `onnxruntime` on macOS/ARM Linux, `onnxruntime-gpu` on x86 Linux and Windows.
- **URL dependency stripping**: URL dependencies like `sam2 @ git+https://...` are rewritten to plain package names.

