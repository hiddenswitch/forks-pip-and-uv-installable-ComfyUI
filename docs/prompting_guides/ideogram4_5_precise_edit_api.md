# Ideogram 4.5 Precise Edit API (Ideogram developer docs)

Vendored snapshot from: https://developer.ideogram.ai/api-reference/images/precise-edit/ideogram-4-5.md

Fetched: 2026-10-05

---

> For clean Markdown of any page, append .md to the page URL.
> For a complete documentation index, see https://developer.ideogram.ai/llms.txt.
> For AI client integration (Claude Code, Cursor, etc.), connect to the MCP server at https://developer.ideogram.ai/_mcp/server.

# Precise edit with Ideogram 4.5

POST https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5
Content-Type: multipart/form-data

Edit an image with Ideogram 4.5 and get the result back at that image's
exact width and height. Upload the image as `image` using
`multipart/form-data`, with optional `reference_images` and a `mask`.
Returns results directly by default; set `async` or supply a
`webhook_url` to get a `generation_id` and poll
`GET /v2/generations/{generation_id}`.

Reference: https://developer.ideogram.ai/api-reference/images/precise-edit/ideogram-4-5

## Authentication

- `Api-Key` header (required) — API key for access control. Use in the header with the name \"Api-Key\"

## Request

### Query parameters

- `dry_run` (boolean, optional, default: false) — When true, the request is validated and priced but not run: nothing is generated, stored, or billed, and no safety review is performed. The response is a `PriceQuote` object instead of the usual response for this endpoint. Send exactly the request you would send to generate, so the quote reflects the same options.

### Body (multipart/form-data)

This endpoint expects a multipart form with multiple files.

- `prompt` (string, required) — The edit instruction, in natural language or as a structured JSON prompt. Natural language is automatically converted into a structured prompt; valid structured JSON is used as is.
- `image` (file, required) — The image to edit (max 25MB; JPEG, PNG, or WEBP). Multipart requests only. The output always matches this image's width and height, and pixels the edit did not meaningfully change are copied exactly from it. Images too large for the model are scaled down, keeping their proportions, and returned as rendered. Images with an aspect ratio outside 1:6 to 6:1 are rejected.
- `reference_images` (files, optional) — Optional images to guide the edit (max 4, max 25MB each; JPEG, PNG, or WEBP). They are never edited themselves; only `image` is. Multipart requests only. A request with a `mask` can include at most three, because the mask takes up one reference slot.
- `mask` (file, optional) — An optional mask that limits the edit to part of `image` (max 25MB; JPEG, PNG, or WEBP). Multipart requests only. Black marks the area to edit and white the area to keep; values in between are rounded to the nearer of the two. The mask must have the same width and height as `image` and contain both black and white areas. A masked request can include at most three `reference_images`.
- `quality` (enum, optional) — The rendering quality to use. `very_low` is the fastest and cheapest, and `high` takes longer and is priced higher.
- `seed` (integer, optional) — Random seed. Set for reproducible generation.
- `num_images` (integer, optional) — The number of images to generate.
- `enable_copyright_detection` (boolean, optional) — Optional. Run copyright detection on the generated images. Adds latency; flagged images are returned with `is_image_safe: false`.
- `async` (boolean, optional) — When false (the default), the request waits until the images are ready and returns them in `data`. When true, the request returns as soon as it is accepted; poll `GET /v2/generations/{generation_id}` with the returned `generation_id` for the result.
- `webhook_url` (string, optional) — HTTPS URL that Ideogram delivers the generated result to. Ideogram sends a JSON POST to this URL once all images for the request have finished generating. The body mirrors the synchronous generate response: `request_id`, `created`, and a `data` array containing every generated image (`url`, `prompt`, `resolution`, `seed`, `is_image_safe`). Each delivery is signed with Ed25519 and verifiable against the public keys at `https://api.ideogram.ai/v1/.well-known/jwks.json`. Must be HTTPS; private and loopback hosts and the cloud metadata service are rejected.

## Response

### 200

The edited images (synchronous requests), or an acknowledgement to poll with `GET /v2/generations/{generation_id}` (`async` requests).

- `generation_id` (string, optional) — URL-safe base64 ID of the generation. Use it to poll `GET /v2/generations/{generation_id}`.
- `data` (list of GeneratedImageObject, optional) — The edited images, in generation order. Present only for synchronous requests (`async` omitted or false).
- `seed` (integer, optional) — The seed that was used, including when you did not set one.

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
  "image": "<file: string>",
  "mask": "<file: <file1>>",
  "prompt": "string",
  "reference_images": []
}
```

**Response**

```json
{
  "generation_id": "generation_id",
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
  ],
  "seed": 0
}
```

**Upstream SDK placeholders**

These autogenerated snippets contain missing values and are not runnable examples. For complete requests, see the [API quickstart](ideogram4_5_api.md#quickstart).

```python
import requests

url = "https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5"

files = {
    "image": "open('string', 'rb')",
    "mask": "open('<file1>', 'rb')"
}
payload = {
    "async": ,
    "enable_copyright_detection": ,
    "num_images": ,
    "prompt": "string",
    "quality": ,
    "seed": ,
    "webhook_url": 
}
headers = {"Api-Key": "<apiKey>"}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.json())
```

```javascript
const url = 'https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5';
const form = new FormData();
form.append('async', '');
form.append('enable_copyright_detection', '');
form.append('image', 'string');
form.append('mask', '<file1>');
form.append('num_images', '');
form.append('prompt', 'string');
form.append('quality', '');
form.append('seed', '');
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

	url := "https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5"

	payload := strings.NewReader("-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"image\"; filename=\"string\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n")

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

url = URI("https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5")

http = Net::HTTP.new(url.host, url.port)
http.use_ssl = true

request = Net::HTTP::Post.new(url)
request["Api-Key"] = '<apiKey>'
request.body = "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"image\"; filename=\"string\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n"

response = http.request(request)
puts response.read_body
```

```java
import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;

HttpResponse<String> response = Unirest.post("https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5")
  .header("Api-Key", "<apiKey>")
  .body("-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"image\"; filename=\"string\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n")
  .asString();
```

```php
<?php
require_once('vendor/autoload.php');

$client = new \GuzzleHttp\Client();

$response = $client->request('POST', 'https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5', [
  'multipart' => [
    [
        'name' => 'image',
        'filename' => 'string',
        'contents' => null
    ],
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

var client = new RestClient("https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5");
var request = new RestRequest(Method.POST);
request.AddHeader("Api-Key", "<apiKey>");
request.AddParameter("undefined", "-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"async\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"enable_copyright_detection\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"image\"; filename=\"string\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"<file1>\"\r\nContent-Type: application/octet-stream\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"num_images\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nstring\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"quality\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n\r\n-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"webhook_url\"\r\n\r\n\r\n-----011000010111000001101001--\r\n", ParameterType.RequestBody);
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
    "name": "image",
    "fileName": "string"
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

let request = NSMutableURLRequest(url: NSURL(string: "https://api.ideogram.ai/v2/image/precise-edit/ideogram-4-5")! as URL,
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