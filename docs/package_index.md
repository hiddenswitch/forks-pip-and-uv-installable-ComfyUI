# Package index hosting

The facade exposes ecosystem custom nodes as Python wheels through a PEP 503
simple index. It resolves registry versions, packages repository contents,
adds dependency metadata and a `comfyui.custom_nodes` entry point, then caches
the resulting wheel. At runtime these wheels use vanilla-node compatibility
loading; they are transport wrappers, not rewrites of the upstream nodes.

## Start an index

```console
comfyui serve-pip --listen 0.0.0.0 --port 8190
```

Install from it in your application environment:

```console
uv pip install --extra-index-url http://localhost:8190/simple/ comfyui-wanvideowrapper
```

The public service is [nodes.appmana.com](https://nodes.appmana.com/simple/).
Its accelerator-specific indexes provide extension wheels for matching backend
builds; select a wheel compatible with your Python, platform, Torch and CUDA stack.

## Cache and registry

Registry snapshots (`--pip-facade-snapshot-uri`), wheel caches
(`--pip-facade-cache-prefix`) and snapshot output all accept fsspec URLs:
local paths, `file://`, `s3://`, `gs://`, `memory://` and other supported backends.

```console
comfyui serve-pip --pip-facade-cache-prefix=/tmp/comfyui-pip-facade
comfyui serve-pip --pip-facade-cache-prefix=s3://my-bucket/comfyui/pip-facade
```

For an S3-compatible service, configure credentials through its normal environment
and provide the endpoint:

```console
comfyui serve-pip \
  --pip-facade-cache-prefix=s3://my-bucket/comfyui/pip-facade \
  --pip-facade-cache-s3-endpoint-url=http://seaweedfs-s3:8333
```

Backend options also use fsspec configuration, including
`FSSPEC_<PROTOCOL>_<OPTION>` environment variables. For example,
`FSSPEC_S3_ENDPOINT_URL` is equivalent to `--pip-facade-cache-s3-endpoint-url`.
Local cache entries are published through a temporary file and rename; object
stores receive complete uploads.
Cache keys use `{cache-prefix}/v{revision}/{project}/{filename}.whl`;
`--pip-facade-cache-revision` selects the revision.

`--pip-facade-only-known-nodes` restricts the registry to the configured known
node set; it is not a guarantee that every workflow using those nodes works.
`comfyui snapshot-pip-registry --help` describes snapshot generation. A prepared
SQLite snapshot reduces registry startup work. Update snapshots separately from
the serving process, using `--pip-facade-snapshot-overwrite` to replace an output.
The server loads each generation into memory and checks for updates every
30 seconds. Requests and `/livez` and `/readyz` probes do not access snapshot
storage. If storage is unavailable, the last generation remains available until
it exceeds `--pip-facade-snapshot-max-age-seconds`; slow storage delays new
generations and cache misses. Each loaded snapshot replaces its predecessor
atomically.

## Dependency rewriting

Generated metadata incorporates compatibility rules:

- Constraints for `numpy`, `jax`, `jaxlib`, `image-reward`, and `timm` are stripped.
- OpenCV variants are normalized to `opencv-contrib-python-headless`.
- ONNX Runtime variants become platform-conditional CPU or GPU dependencies.
- Direct URL requirements become package names, resolved through configured indexes.

These transformations can differ from a repository's original requirements.
Inspect the resolved environment when diagnosing a node-specific failure.
Repositories without published registry versions can receive synthetic archive
versions. For workflow installation, the CLI also supports a local facade build
fallback when a package is missing from the index.

See [node installation](custom_nodes.md) and the
[management reference](custom_node_reference.md) for runtime loading details.
