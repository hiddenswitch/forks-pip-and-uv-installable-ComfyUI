# Upstream merge compatibility

## PromptServer references

Built-in nodes report execution progress through the current execution context,
not the global HTTP server. Translate upstream `PromptServer.instance.send_sync`
and `PromptServer.instance.send_progress_text` calls to the context-owned server:

```python
from comfy.execution_context import current_execution_context

server = current_execution_context().server
server.send_progress_text(text, node_id)
```

Resolve the server inside the executing method. Do not capture it at import time,
store it on a node class, or add a `PromptServer` import merely to satisfy lint.
The context owns an `ExecutorToClientProgress` implementation. It can be an HTTP
server, an embedded client's progress receiver, or a no-op `ServerStub`; a web
server need not exist. Preserve the upstream node-ID guard and message payload.
`context_execute_prompt` binds the receiver for an execution. Process peers that
use `copy_process_context` intentionally receive a `ServerStub` instead of the
parent's live server and request state.

HTTP route registration has a different owner. Use
`comfy.component_model.plugins.prompt_server_instance_routes` for deferred
route decorators. Do not substitute the execution-context server for route,
application, queue, or HTTP-server lifecycle APIs.

Unmodified third-party nodes retain their compatibility layer:
`comfy_compatibility.vanilla.prepare_vanilla_environment` aliases the upstream
`server` module and installs `_PromptServerStub` when necessary. That import-time
shim does not replace context-based progress reporting in maintained built-ins.
Keep real `PromptServer` references in the HTTP server and compatibility layer
where those objects are actually required.

Regression coverage should bind a recording receiver with
`context_execute_prompt`, execute the node with its hidden node ID, and assert
both its outputs and the delivered progress event. This catches misrouted or
missing embedded progress that importing `PromptServer` would conceal. See
`tests/unit/test_get_latent_size.py` and `GetImageSize` in
`comfy_extras/nodes/nodes_images.py`.
