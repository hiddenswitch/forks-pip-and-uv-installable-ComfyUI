# Ideogram 4.5 API (Ideogram developer docs)

Vendored snapshots from: https://developer.ideogram.ai/ideogram-api/api-overview.md and https://developer.ideogram.ai/api-reference/images/generate/ideogram-4-5.md

Fetched: 2026-10-02. API additions reviewed: 2026-10-29.

The current v2 generation endpoint adds `background` (`transparent`, `opaque`,
or `auto`) and a required response discriminator `generation_kind` (`sampling`
or `workflow`). See the [live generation schema](https://developer.ideogram.ai/api-reference/images/generate/ideogram-4-5.md).
For Precise Edit, large whole-image edits are resized; use the
[context-window rules](ideogram4_5_precise_edit_api.md) when preserving the full
input canvas. The historical overview's unrestricted same-size wording is not
the endpoint's current contract. This is a hosted API reference, not evidence
of a local 4.5 checkpoint.

Ideogram 4.5 is API-only at this snapshot (rechecked 2026-10-05, when both pages were unchanged): Ideogram announced open weights as coming "soon" with no date, license or size, and nothing is published on Hugging Face (https://huggingface.co/ideogram-ai) or https://github.com/ideogram-oss. Its prompt is natural language or the Ideogram 4.0 structured JSON prompt, so the Ideogram 4 caption schema in `ideogram4_prompting.md` applies to it. The Precise Edit endpoint is vendored in `ideogram4_5_precise_edit_api.md`, and Ideogram's prompting guide, which covers 4.5, in `ideogram4_5_prompting.md`.

---

> For clean Markdown of any page, append .md to the page URL.
> For a complete documentation index, see https://developer.ideogram.ai/llms.txt.
> For AI client integration (Claude Code, Cursor, etc.), connect to the MCP server at https://developer.ideogram.ai/_mcp/server.

> Overview of the Ideogram API v2 — precise image editing with Ideogram 4.5, generation, video, and where to start.

The Ideogram API brings Ideogram's image and video models into your product. Every v2 endpoint follows one pattern, `POST /v2/{content}/{action}/{model}`, so the model you call is always in the URL: for example `/v2/image/generate/ideogram-4-5`.

## Ideogram 4.5: the most precise edit model

Tell Ideogram 4.5 what to change. [Precise Edit](ideogram4_5_precise_edit_api.md) preserves unchanged regions subject to its context-window and output-size rules.

#### Targeted edits

Change a colour, material, or detail and keep the rest of the image untouched. Add a `mask` to limit the edit to one area.

#### Consistent across edits

Chain edit after edit with minimal drift: shape, texture, and colour hold steady from one turn to the next.

#### Guided by references

Add up to four `reference_images` to steer the edit with a product, a material, or a style.

*Original, then "Change the product's colour and finish to slate-blue knit", then two edits later with a gum sole and a pale lilac backdrop. The shoes keep their exact shape and position throughout.*

![](https://developer.ideogram.ai/_fern-files/ideogram.docs.buildwithfern.com/3ab860bf2397ba02f3539f9f20b1ea5b8f25463195ba4bda1ce80836d0f610ac/assets/v2/precise-edit-shoes-original.webp)

![](https://developer.ideogram.ai/_fern-files/ideogram.docs.buildwithfern.com/c1881492a1074bc19ae5f5d9ce2fbeb6b3140ef9d3beb304540f1a16551f750b/assets/v2/precise-edit-shoes-recolor.webp)

![](https://developer.ideogram.ai/_fern-files/ideogram.docs.buildwithfern.com/cd1cd4f72cbf471cb9a808670ce3692760b0ae24332d15e4899ff371d06cd1a0/assets/v2/precise-edit-shoes-backdrop.webp)

## What you can build

#### [Generate](https://developer.ideogram.ai/api-reference/images/generate/ideogram-4-5)

Create images from a prompt with Ideogram 4.5 and 4.0, known for prompt fidelity and crisp typography, or with GPT Image and Nano Banana.

#### [Remix and inpaint](https://developer.ideogram.ai/api-reference/images/remix/ideogram-4)

Transform an existing image with a prompt, or repaint just the masked region.

#### [Upscale](https://developer.ideogram.ai/api-reference/images/upscale/auto)

Enlarge images with Topaz and Nano Banana Pro, or let Ideogram pick the model.

#### [Backgrounds](https://developer.ideogram.ai/api-reference/images/remove-background/ideogram-1)

Remove a background for a transparent cutout, or replace it with a generated scene.

#### [Video](https://developer.ideogram.ai/api-reference/video/generate/seedance-2-5-text-to-video)

Generate video from text, a first frame, or reference images with Seedance, MiniMax H3, and Kling.

#### [Tools](https://developer.ideogram.ai/api-reference/tools/ad-resizer)

Ready-made workflows for ads and commerce: resize, localize, recolour, and swap materials.

## Quickstart

Create an API key by following the [Setup guide](https://developer.ideogram.ai/ideogram-api/api-setup). Then generate an image with Ideogram 4.5 and make a precise edit to it. Image endpoints return results directly; image URLs expire, so download anything you want to keep.

#### Python

```python
import requests

API = "https://api.ideogram.ai"
HEADERS = {"Api-Key": "<apiKey>"}

# Generate an image with Ideogram 4.5
response = requests.post(
  f"{API}/v2/image/generate/ideogram-4-5",
  headers=HEADERS,
  json={"prompt": "Knit running shoes in neon coral on a warm stone paper sweep, soft window light"},
)
response.raise_for_status()
with open("shoes.png", "wb") as f:
  f.write(requests.get(response.json()["data"][0]["url"]).content)

# Precisely edit it: only the change you describe
with open("shoes.png", "rb") as image:
  response = requests.post(
    f"{API}/v2/image/precise-edit/ideogram-4-5",
    headers=HEADERS,
    data={"prompt": "Change the shoes to slate-blue knit. Keep everything else exactly the same."},
    files={"image": image},
  )
response.raise_for_status()
print(response.json()["data"][0]["url"])
```

#### TypeScript

```typescript
import { readFile, writeFile } from "node:fs/promises";

const API = "https://api.ideogram.ai";
const headers = { "Api-Key": "<apiKey>" };

// Generate an image with Ideogram 4.5
const generated = await fetch(`${API}/v2/image/generate/ideogram-4-5`, {
  method: "POST",
  headers: { ...headers, "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Knit running shoes in neon coral on a warm stone paper sweep, soft window light",
  }),
}).then((response) => response.json());
const image = await fetch(generated.data[0].url).then((response) => response.arrayBuffer());
await writeFile("shoes.png", Buffer.from(image));

// Precisely edit it: only the change you describe
const form = new FormData();
form.append("prompt", "Change the shoes to slate-blue knit. Keep everything else exactly the same.");
form.append("image", new Blob([await readFile("shoes.png")]), "shoes.png");
const edited = await fetch(`${API}/v2/image/precise-edit/ideogram-4-5`, {
  method: "POST",
  headers,
  body: form,
}).then((response) => response.json());
console.log(edited.data[0].url);
```

#### cURL

```bash
# Generate an image with Ideogram 4.5
curl -X POST https://api.ideogram.ai/v2/image/generate/ideogram-4-5 \
  -H "Api-Key: <apiKey>" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Knit running shoes in neon coral on a warm stone paper sweep, soft window light"}'

# Precisely edit an image: only the change you describe
curl -X POST https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5 \
  -H "Api-Key: <apiKey>" \
  -F image=@shoes.png \
  -F prompt="Change the shoes to slate-blue knit. Keep everything else exactly the same."
```

### Long-running requests

Video and tool endpoints always run asynchronously, and any image endpoint does too when you set `async` or supply a `webhook_url`. They return a `generation_id` straight away: poll [`GET /v2/generations/{generation_id}`](https://developer.ideogram.ai/api-reference/generations/get-generation) until `status` is `completed` or `failed`, or receive the result at your [webhook](https://developer.ideogram.ai/ideogram-api/webhooks).

```python
import time

with open("shoes.png", "rb") as image:
  response = requests.post(
    f"{API}/v2/image/precise-edit/ideogram-4-5",
    headers=HEADERS,
    data={"prompt": "Change the backdrop to a soft pale lilac.", "async": "true"},
    files={"image": image},
  )
generation_id = response.json()["generation_id"]

while True:
  generation = requests.get(f"{API}/v2/generations/{generation_id}", headers=HEADERS).json()
  if generation["status"] != "pending":
    break
  time.sleep(2)
print(generation["status"], generation.get("data"))
```

## Moving from v1

The v1 API keeps working, and its reference stays available under **v1** in the version switcher. Each v1 endpoint has a v2 equivalent with the model in the path; request fields differ in places (for example, v1's `text_prompt` is v2's `prompt`), so check each endpoint's reference.

| v1 endpoint                               | v2 endpoint                                    |
| ----------------------------------------- | ---------------------------------------------- |
| `POST /v1/ideogram-v4/generate`           | `POST /v2/image/generate/ideogram-4`           |
| `POST /v1/ideogram-v3/generate`           | `POST /v2/image/generate/ideogram-3`           |
| `POST /v1/ideogram-v4/remix`              | `POST /v2/image/remix/ideogram-4`              |
| `POST /v1/ideogram-v3/remix`              | `POST /v2/image/remix/ideogram-3`              |
| `POST /v1/ideogram-v3/inpaint`            | `POST /v2/image/inpaint/ideogram-3`            |
| `POST /v1/ideogram-v3/reframe`            | `POST /v2/image/reframe/ideogram-3`            |
| `POST /v1/ideogram-v3/replace-background` | `POST /v2/image/replace-background/ideogram-3` |
| `POST /v1/remove-background`              | `POST /v2/image/remove-background/ideogram-1`  |
| `POST /v1/ideogram-v4/describe`           | `POST /v2/image/describe/ideogram-4`           |
| `GET /v1/generations/{generation_id}`     | `GET /v2/generations/{generation_id}`          |

## Enterprise scale

The Ideogram API serves thousands of API customers generating millions of images daily. If you need more than the default rate limit of 10 in-flight requests, contact us at *[partnership@ideogram.ai](mailto:partnership@ideogram.ai)* and we'll help fit your needs.

---

> For clean Markdown of any page, append .md to the page URL.
> For a complete documentation index, see https://developer.ideogram.ai/llms.txt.
> For AI client integration (Claude Code, Cursor, etc.), connect to the MCP server at https://developer.ideogram.ai/_mcp/server.

# Generate with Ideogram 4.5

POST https://api.ideogram.ai/v2/image/generate/ideogram-4-5
Content-Type: multipart/form-data

Generate images with Ideogram 4.5 from a natural-language or structured
JSON prompt. Optionally upload source images as `images` using
`multipart/form-data` to edit them with the prompt. Returns results
directly by default; set `async` or supply a `webhook_url` to get a
`generation_id` and poll `GET /v2/generations/{generation_id}`.

Reference: https://developer.ideogram.ai/api-reference/images/generate/ideogram-4-5

## Authentication

- `Api-Key` header (required) — API key for access control. Use in the header with the name \"Api-Key\"

## Request

### Query parameters

- `dry_run` (boolean, optional, default: false) — When true, the request is validated and priced but not run: nothing is generated, stored, or billed, and no safety review is performed. The response is a `PriceQuote` object instead of the usual response for this endpoint. Send exactly the request you would send to generate, so the quote reflects the same options.

### Body (multipart/form-data)

This endpoint expects a multipart form with multiple files.

- `prompt` (string, required) — The prompt to generate images from, or the edit instruction when source images are supplied. Natural language or a structured Ideogram 4.0 JSON prompt.
- `magic_prompt` (enum, optional) — Controls how a natural-language prompt is prepared. `auto` (the default) and `on` rewrite and expand the prompt before generation. `off` keeps your wording and only converts it into a structured prompt. A valid structured JSON prompt skips magic prompt unless `magic_prompt` is `on`. With source images, every mode converts the edit instruction into a structured edit prompt.
- `images` (files, optional) — Optional source images to edit (max 5, max 25MB each; JPEG, PNG, or WEBP). The first image is the one being edited; any others are extra references. Multipart requests only.
- `mask` (file, optional) — An optional mask that limits the edit to part of the first source image (max 25MB; JPEG, PNG, or WEBP). Multipart requests only. Black marks the area to edit and white the area to keep; values in between are rounded to the nearer of the two. The mask must have the same width and height as the first source image and contain both black and white areas. A masked request can include at most three other source images. With a mask, the output is always the first source image's own size, so `size` cannot be set.
- `size` (string, optional) — The output size: "auto", "source" or an exact "WIDTHxHEIGHT". Without source images, an exact size must be one of the supported 1K/2K presets (for example 1024x1024, 2048x2048 or 1440x2880); "auto" or omitted picks a supported size based on the prompt. "source" is rejected. With source images, "auto" (the default) picks a supported 2K size based on the source images and the prompt, and "source" returns the output at the first source image's own width and height (scaled down, keeping its proportions, if it is too large for the model). Every source image's aspect ratio must be between 1:6 and 6:1. An exact size must have both sides a multiple of 32 and at least 256px, a total of at most 2048x2048 pixels, and an aspect ratio of at most 6:1. With source images, an exact size reshapes the source to it. Pricing is tiered by the resolved output pixels: up to 1024x1024 bills as 1K, above that as 2K. An "auto" size bills as 2K.
- `quality` (enum, optional) — The rendering quality to use. Higher quality takes longer and costs more. Defaults to `medium` with source images and `high` without. `very_low`, the fastest and cheapest, requires source images.
- `seed` (integer, optional) — Random seed. Set for reproducible generation.
- `num_images` (integer, optional) — The number of images to generate.
- `enable_copyright_detection` (boolean, optional) — Optional. Run copyright detection on the generated images. Adds latency; flagged images are returned with `is_image_safe: false`.
- `async` (boolean, optional) — When false (the default), the request waits until the images are ready and returns them in `data`. When true, the request returns as soon as it is accepted; poll `GET /v2/generations/{generation_id}` with the returned `generation_id` for the result.
- `webhook_url` (string, optional) — HTTPS URL that Ideogram delivers the generated result to. Ideogram sends a JSON POST to this URL once all images for the request have finished generating. The body mirrors the synchronous generate response: `request_id`, `created`, and a `data` array containing every generated image (`url`, `prompt`, `resolution`, `seed`, `is_image_safe`). Each delivery is signed with Ed25519 and verifiable against the public keys at `https://api.ideogram.ai/v1/.well-known/jwks.json`. Must be HTTPS; private and loopback hosts and the cloud metadata service are rejected.

## Response

### 200

The generated images (synchronous requests), or an acknowledgement to poll with `GET /v2/generations/{generation_id}` (`async` requests).

- `generation_id` (string, required) — URL-safe base64 ID of the generation. Use it to poll `GET /v2/generations/{generation_id}`.
- `seed` (integer, required) — Random seed. Set for reproducible generation.
- `data` (list of GeneratedImageObject, optional) — The generated images, in generation order. Present only for synchronous requests (`async` omitted or false).

## Errors

### 400 Bad Request Error

Invalid input provided.

- `any`

### 401 Unauthorized Error

Unauthorized.

- `any`

### 402 Payment Required Error

Insufficient credits or quota.

- `error` (string, required) — A message describing why the generation request was rejected.
- `reject_reason` (enum, required) — The account or usage limit that rejected a generation request.
  - Allowed values: `insufficient_funds`, `subscription_required`, `daily_limit`, `priority_credit_required`, `inflight_limit`, `feature_limit`
- `max_inflight_requests` (integer, optional) — How many generations the account may have in progress at once on the queue this request resolved to. Present when `reject_reason` is `inflight_limit`.
- `task_completion_speed` (enum, optional) — The queue this request resolved to. Present when `reject_reason` is `inflight_limit`.
  - Allowed values: `fast`, `slow`

### 422 Unprocessable Entity Error

The prompt did not pass safety checks.

- `any`

### 429 Too Many Requests Error

Too many requests.

- `error` (string, required) — A message describing why the generation request was rejected.
- `reject_reason` (enum, required) — The account or usage limit that rejected a generation request.
  - Allowed values: `insufficient_funds`, `subscription_required`, `daily_limit`, `priority_credit_required`, `inflight_limit`, `feature_limit`
- `max_inflight_requests` (integer, optional) — How many generations the account may have in progress at once on the queue this request resolved to. Present when `reject_reason` is `inflight_limit`.
- `task_completion_speed` (enum, optional) — The queue this request resolved to. Present when `reject_reason` is `inflight_limit`.
  - Allowed values: `fast`, `slow`

### 500 Internal Server Error

Internal server error.

- `any`

### 503 Service Unavailable Error

The endpoint is temporarily unavailable.

- `any`

## Types

### GeneratedImageObject

One generated output image.

- `prompt` (string, required) — The final prompt the image was generated from.
- `resolution` (string, required) — The resolution of the generated image, formatted as "WIDTHxHEIGHT".
- `is_image_safe` (boolean, required) — Whether the image passed safety checks. If false, `url` is empty.
- `seed` (integer, required) — Random seed. Set for reproducible generation.
- `url` (string, optional, nullable) — The direct link to the generated image. Empty when the image did not pass safety checks.

## Examples

**Request**

```json
{
  "images": [],
  "mask": "<file: <file1>>",
  "prompt": "string"
}
```

**Response**

```json
{
  "generation_id": "generation_id",
  "seed": 12345,
  "data": [
    {
      "prompt": "prompt",
      "resolution": "1024x1024",
      "is_image_safe": true,
      "seed": 12345,
      "url": "https://openapi-generator.tech"
    },
    {
      "prompt": "prompt",
      "resolution": "1024x1024",
      "is_image_safe": true,
      "seed": 12345,
      "url": "https://openapi-generator.tech"
    }
  ]
}
```

**Upstream SDK placeholders**

These autogenerated snippets contain missing values and are not runnable examples. For complete requests, see the [API quickstart](ideogram4_5_api.md#quickstart).

```python
import requests

url = "https://api.ideogram.ai/v2/image/generate/ideogram-4-5"

files = { "mask": "open('<file1>', 'rb')" }
payload = {
    "async": ,
    "enable_copyright_detection": ,
    "magic_prompt": ,
    "num_images": ,
    "prompt": "string",
    "quality": ,
    "seed": ,
    "size": ,
    "webhook_url": 
}
headers = {"Api-Key": "<apiKey>"}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.json())
```

```javascript
const url = 'https://api.ideogram.ai/v2/image/generate/ideogram-4-5';
const form = new FormData();
form.append('async', '');
form.append('enable_copyright_detection', '');
form.append('magic_prompt', '');
form.append('mask', '<file1>');
form.append('num_images', '');
form.append('prompt', 'string');
form.append('quality', '');
form.append('seed', '');
form.append('size', '');
form.append('webhook_url', '');

const options = {method: 'POST', headers: {'Api-Key': '<apiKey>'}};

options.body = form;

try {
  const response = await fetch(url, options);
  const data = await response.json();
  console.log(data);
} catch (error) {
  console.error(error);
}
```

```go
package main

import (
	"fmt"
	"strings"
	"net/http"
	"io"
)

func main() {

	url := "https://api.ideogram.ai/v2/image/generate/ideogram-4-5"

	payload := strings.NewReader("-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"magic_prompt\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n")

	req, _ := http.NewRequest("POST", url, payload)

	req.Header.Add("Api-Key", "<apiKey>")

	res, _ := http.DefaultClient.Do(req)

	defer res.Body.Close()
	body, _ := io.ReadAll(res.Body)

	fmt.Println(res)
	fmt.Println(string(body))

}
```

```ruby
require 'uri'
require 'net/http'

url = URI("https://api.ideogram.ai/v2/image/generate/ideogram-4-5")

http = Net::HTTP.new(url.host, url.port)
http.use_ssl = true

request = Net::HTTP::Post.new(url)
request["Api-Key"] = '<apiKey>'
request.body = "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"magic_prompt\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n"

response = http.request(request)
puts response.read_body
```

```java
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;

HttpResponse<String> response = Unirest.post("https://api.ideogram.ai/v2/image/generate/ideogram-4-5")
  .header("Api-Key", "<apiKey>")
  .body("-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"magic_prompt\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n")
  .asString();
```

```php
<?php
require_once('vendor/autoload.php');

$client = new \GuzzleHttp\Client();

$response = $client->request('POST', 'https://api.ideogram.ai/v2/image/generate/ideogram-4-5', [
  'multipart' => [
    [
        'name' => 'mask',
        'filename' => '<file1>',
        'contents' => null
    ],
    [
        'name' => 'prompt',
        'contents' => 'string'
    ]
  ]
  'headers' => [
    'Api-Key' => '<apiKey>',
  ],
]);

echo $response->getBody();
```

```csharp
using RestSharp;

var client = new RestClient("https://api.ideogram.ai/v2/image/generate/ideogram-4-5");
var request = new RestRequest(Method.POST);
request.AddHeader("Api-Key", "<apiKey>");
request.AddParameter("undefined", "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"magic_prompt\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n", ParameterType.RequestBody);
IRestResponse response = client.Execute(request);
```

```swift
import Foundation

let headers = ["Api-Key": "<apiKey>"]
let parameters = [
  [
    "name": "async",
    "value": 
  ],
  [
    "name": "enable_copyright_detection",
    "value": 
  ],
  [
    "name": "magic_prompt",
    "value": 
  ],
  [
    "name": "mask",
    "fileName": "<file1>"
  ],
  [
    "name": "num_images",
    "value": 
  ],
  [
    "name": "prompt",
    "value": "string"
  ],
  [
    "name": "quality",
    "value": 
  ],
  [
    "name": "seed",
    "value": 
  ],
  [
    "name": "size",
    "value": 
  ],
  [
    "name": "webhook_url",
    "value": 
  ]
]

let boundary = "---011000010111000001101001"

var body = ""
var error: NSError? = nil
for param in parameters {
  let paramName = param["name"]!
  body += "--\(boundary)\r\n"
  body += "Content-Disposition:form-data; name=\"\(paramName)\""
  if let filename = param["fileName"] {
    let contentType = param["content-type"]!
    let fileContent = String(contentsOfFile: filename, encoding: String.Encoding.utf8)
    if (error != nil) {
      print(error as Any)
    }
    body += "; filename=\"\(filename)\"\r\n"
    body += "Content-Type: \(contentType)\r\n\r\n"
    body += fileContent
  } else if let paramValue = param["value"] {
    body += "\r\n\r\n\(paramValue)"
  }
}

let request = NSMutableURLRequest(url: NSURL(string: "https://api.ideogram.ai/v2/image/generate/ideogram-4-5")! as URL,
                                        cachePolicy: .useProtocolCachePolicy,
                                    timeoutInterval: 10.0)
request.httpMethod = "POST"
request.allHTTPHeaderFields = headers
request.httpBody = postData as Data

let session = URLSession.shared
let dataTask = session.dataTask(with: request as URLRequest, completionHandler: { (data, response, error) -> Void in
  if (error != nil) {
    print(error as Any)
  } else {
    let httpResponse = response as? HTTPURLResponse
    print(httpResponse)
  }
})

dataTask.resume()
```
