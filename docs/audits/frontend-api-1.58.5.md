# Frontend API compatibility audit

Audited against the installed frontend pin, `comfyui-frontend-package==1.58.5`,
using Comfy-Org/ComfyUI_frontend tag `v1.58.5`
(`f7bcaf00095ddead1be4480b01f7cd6e851f726d`). The reference contracts are
`src/scripts/api.ts`, `src/platform/remote/comfyui/jobs/jobTypes.ts`,
`src/platform/remote/comfyui/execution/types.ts`,
`src/platform/assets/schemas/assetSchema.ts`, and the workspace packages
`@comfyorg/object-info-parser` and `@comfyorg/ingest-types`.

## Findings fixed

- **Execution events were discarded at transport boundaries.** CLI history flattened
  events into strings; embedded and distributed execution manufactured empty success
  statuses or string errors. Preserve the executor's events, including timestamps,
  cached node IDs, error details, and interruption details. Keep sensitive input
  values out of public history without reducing events to a timestamp allowlist.
- **Embedded queue items could lack creation times.** An explicitly supplied sensitive
  data dictionary bypassed timestamp initialization. Capture creation before execution.
- **Embedded/distributed history pagination and missing-ID lookup were incorrect.**
  Apply limits relative to offsets, return no entry for unknown IDs, and enforce retention.
- **Queue responses differed from the public tuple contract.** Serialize five fields
  consistently and remove authentication data at queue/history publication boundaries.
- **Default jobs pagination returned a null limit.** The frontend requires a number;
  the default page size is now 200.
- **WebSocket reconnect omitted the current prompt ID.** Include it with the current node.
- **Native V3 node metadata violated the frontend's node-definition schema.** Omit absent
  optional fields, provide a display-name fallback, and preserve output positions with
  string tooltip/match-type entries. All 853 native definitions in the isolated test
  installation now pass the frontend's actual Zod validator.
- **Dataset text nodes emitted null list flags and incorrectly nested scalar outputs.**
  Resolve automatic list mode consistently with their execution results.
- **The fork's synchronous prompt API manufactured success for failed execution.**
  Read the stored execution status. Native image outputs also work without the fork's
  optional `abs_path` field, for both JSON URL responses and PNG responses.
- **OpenAPI described string events and required nonstandard image paths.** Update both
  history contracts and generated models; allow custom node UI output fields.

## Coverage

| API family | Verification |
| --- | --- |
| Prompt, queue, history, jobs | Real CPU server, native success/failure workflows, WebSocket/history event equality, durations, outputs, validation errors, pagination, unknown IDs; embedded thread/process execution and RPC JSON round trip |
| Cancellation and WebSockets | Production cancellation routes, queue state, reconnect, feature negotiation, binary preview metadata and image decoding; interrupted-history job classification |
| Node and model metadata | Entire native `/object_info` response against frontend Zod; single-node route; model listing/preview unit tests; live folders, embeddings, extensions, templates, replacements, subgraphs, i18n, system stats and features |
| Users, settings and files | Real settings write/read, userdata write/list/read/delete, image upload/view; existing user-manager and system-user endpoint tests |
| Assets | Existing route/service suite covering listing/cursors, uploads, content, metadata, tags, hash lookup/creation, seed/status/cancel, pruning, disabled mode and errors; live upload/detail/update/tag/content/delete responses |
| Export and task polling | Existing export tests; live export creation and task completion parsed with the frontend task schema |
| Internal endpoints | Reviewed terminal subscription/log envelopes; live raw logs and folder-path responses |
| Fork prompt API | Real failed workflow returns structured HTTP 500; native image workflow returns usable URLs and PNG bytes |

The cross-language check imported the frontend's actual schemas into Vitest and
validated live response captures: **19 checks passed**. It initially reproduced the
missing duration, null pagination limit, and invalid native node metadata. This was
an ad-hoc check in the exact frontend checkout, not a new runtime dependency.
Permanent Python regression tests exercise production serializers, executors and HTTP
handlers. Cancellation tests no longer reproduce the handlers in test code, and the
literal-only feature-message test was replaced by a real WebSocket exchange.

## Boundaries

Cloud account/billing/authentication, workspace policy, secrets, Hub publishing and
remote downloads are separate service contracts; this audit does not claim to test or
implement those services. Agent/workflow hosting and custom-node manager routes belong
to optional packages. Their availability must follow their distribution/feature gates.
The local frontend explicitly tolerates unavailable job-assets enrichment and a 404
for an empty userdata directory. Assets remain opt-in; disabled routes return their
existing structured 503 response. No network service or inference-weight dependency
was added for this audit.

These checks establish compatibility for the pinned local frontend and native nodes;
they do not certify arbitrary custom-node payloads or future frontend versions.
