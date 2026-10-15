# Qwen Image Edit guide

Vendored snapshot from: https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-guide

Fetched: 2026-06-03

---

<div id="app">

<div class="aliyun-app-layout">

<div class="products-wrapper$tip" spm="879954">

<div id="products" class="section intl-product container">

<div class="row">

<div id="navDocument" class="col-lg-3 col-md-3 col-sm-4 col-xs-12">

<div class="doc-nav">

<a href="https://www.alibabacloud.com/help/en" class="doc-center"><em></em> Document Center</a>

</div>

<div class="placeholder-ele">

 

</div>

</div>

<div class="col-lg-7 col-md-7 col-sm-8 col-xs-12 doc-content">

<div id="J_AllProducts" class="all-products action">

<div class="all-products-head">

<div class="all-products-title">

All Products

</div>

<div class="all-products-search-wrapper">

<div class="all-products-search focus">

</div>

</div>

</div>

<div id="J_AllProductsBody" class="all-products-body">

</div>

</div>

<div class="search-product-modal">

Search

</div>

<div class="breadcrumbs-top" spm="a1">

<div class="row nav-breadcrumb">

<div class="col-md-12 hidden-sm hidden-xs">

- <a href="https://www.alibabacloud.com/help/en" class="active" title="Document Center">Document
  Center</a>
- <a href="https://www.alibabacloud.com/help/en/model-studio/" class="active"
  title="Alibaba Cloud Model Studio">Alibaba Cloud Model Studio</a>
- <a href="https://www.alibabacloud.com/help/en/model-studio/model-user-guide/" class="active"
  title="User Guide (Models)">User Guide (Models)</a>
- <a href="https://www.alibabacloud.com/help/en/model-studio/model-inference/" class="active"
  title="Inference">Inference</a>
- <a href="https://www.alibabacloud.com/help/en/model-studio/image-editing-and-generation/"
  class="active" title="Image generation">Image generation</a>
- <a href="https://www.alibabacloud.com/help/en/model-studio/image-edit-guide/" class="active"
  title="Image editing">Image editing</a>
- <a href="https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-guide" class="active"
  title="Image editing - Qwen">Image editing - Qwen</a>

</div>

<div class="col-md-4 col-sm-12">

<span class="icon-bar"></span> <span class="icon-bar"></span>
all-products-head <span class="icon-bar"></span>

</div>

</div>

<div class="select">

<span class="placeholder">This Product</span>

- This Product
- All Products

</div>

<div class="collapse-menus select">

<span class="placeholder">Alibaba Cloud Model
Studio:Qwen-Image-Edit</span>

</div>

</div>

<div id="header-faq" class="clearfix">

<a href="https://www.alibabacloud.com/help/en" class="doc-center"><em></em> Document Center</a>

<div class="download-links">

</div>

# Alibaba Cloud Model Studio:Qwen-Image-Edit

<div class="doc-status">

Last Updated:Mar 15, 2026

</div>

</div>

<div class="icms-help-docs-content" lang="en">

<div id="main-3061937" role="main">

Qwen-Image-Edit supports multi-image input and output. It can modify
text in images, add/delete/move objects, change subject actions,
transfer styles, and enhance details.

<div id="ea726feb66x1d" class="body" tag="body">

<div id="1fff24bb33edl" class="section section">

## **Getting started**

This example shows how to use `qwen-image-2.0-pro` to generate two
edited images from three input images and a prompt.

> Input prompt: The girl in Image 1 wears the black dress from Image 2
> and sits in the pose from Image 3.

<div id="1894d88341vvj" class="section section">

|                                                                                             |                                                                                             |                                                                                             |                                                                                             |                                                                                             |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **Input image 1**                                                                           | **Input image 2**                                                                           | **Input image 3**                                                                           | **Output images (multiple images)**                                                         |                                                                                             |
| <img                                                                                        
 src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011682.webp"  
 id="a38e339269umd" class="image break" data-placement="break"                                
 width="250" alt="image99" />                                                                 | <img                                                                                        
                                                                                               src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011684.webp"  
                                                                                               id="cdb793eb09h0v" class="image break" data-placement="break"                                
                                                                                               width="250" alt="image98" />                                                                 | <img                                                                                        
                                                                                                                                                                                             src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011683.webp"  
                                                                                                                                                                                             id="5bbf3c2fcab62" class="image break" data-placement="break"                                
                                                                                                                                                                                             width="250" alt="image89" />                                                                 | <img                                                                                        
                                                                                                                                                                                                                                                                                           src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011681.webp"  
                                                                                                                                                                                                                                                                                           id="8f704f029dxj0" class="image break" data-placement="break"                                
                                                                                                                                                                                                                                                                                           width="250" alt="image100" />                                                                | <img                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                         src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6903291671/p1022524.webp"  
                                                                                                                                                                                                                                                                                                                                                                                         id="49470f10d4jmj" class="image break" data-placement="break"                                
                                                                                                                                                                                                                                                                                                                                                                                         width="250" alt="imageout2" />                                                               |

</div>

<div id="7dc8442b1eo8n" class="section section" ref-searchable="yes"
source="reuse_library" docid="4712131" is-conref="true">

Before making a call,
<a href="https://www.alibabacloud.com/help/en/model-studio/get-api-key" id="772f04a6e87ne"
class="xref">get an API key</a> and <a
href="https://www.alibabacloud.com/help/en/model-studio/configure-api-key-through-environment-variables"
id="fece23da404bv" class="xref">export the API key as an environment
variable</a>.

To call the API using the SDK,
<a href="https://www.alibabacloud.com/help/en/model-studio/install-sdk" id="7437001534qbr"
class="xref">install the DashScope SDK</a>. The SDK is available for
Python and Java.

</div>

The Qwen image editing models support one to three input images. The
`qwen-image-2.0`, `qwen-image-edit-max`, and `qwen-image-edit-plus`
series can generate one to six images. `qwen-image-edit` can generate
only one image. The URLs for the generated images are **valid for 24
hours**.
<a href="#e2278796ed73n" id="8c539426f4oyi" class="xref">Download the
images to your local device</a> promptly.

<div id="223003115a515" class="section tabbed-content-box section"
tag="tabbed-content-box" outputclass="tabbed-content-box">

<div id="487518b656dji" class="section section">

## **Python**

``` pre
import json
import os
import dashscope
from dashscope import MultiModalConversation

# The following is the URL for the Singapore region. If you use a model in the Beijing region, replace the URL with: https://dashscope.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# The model supports one to three input images.
messages = [
    {
        "role": "user",
        "content": [
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/thtclx/input1.png"},
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/iclsnx/input2.png"},
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/gborgw/input3.png"},
            {"text": "The girl from Image 1 is wearing the black dress from Image 2 and sitting in the pose from Image 3."}
        ]
    }
]

# The API keys for the Singapore and Beijing regions are different. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key.
# If you have not configured the environment variable, replace the next line with: api_key="sk-xxx"
api_key = os.getenv("DASHSCOPE_API_KEY")

# The qwen-image-2.0, qwen-image-edit-max, and qwen-image-edit-plus series support one to six output images. This example generates two.
response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-2.0-pro",
    messages=messages,
    stream=False,
    n=2,
    watermark=False,
    negative_prompt=" ",
    prompt_extend=True,
    size="1024*1536",
)

if response.status_code == 200:
    # To view the full response, uncomment the next line.
    # print(json.dumps(response, ensure_ascii=False))
    for i, content in enumerate(response.output.choices[0].message.content):
        print(f"URL of output image {i+1}: {content['image']}")
else:
    print(f"HTTP status code: {response.status_code}")
    print(f"Error code: {response.code}")
    print(f"Error message: {response.message}")
    print("For more information, see https://www.alibabacloud.com/help/en/model-studio/error-code")
```

<div id="60b4207279i4d" class="section section">

<div id="46130b2bbbyx1" class="collapse" outputclass="collapse">

**Response example**

<div id="a76b8fde64bei" class="expandable-content"
tag="expandable-content">

``` pre
{
    "status_code": 200,
    "request_id": "fa41f9f9-3cb6-434d-a95d-4ae6b9xxxxxx",
    "code": "",
    "message": "",
    "output": {
        "text": null,
        "finish_reason": null,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "image": "https://dashscope-result-hz.oss-cn-hangzhou.aliyuncs.com/xxx.png?Expires=xxx"
                        },
                        {
                            "image": "https://dashscope-result-hz.oss-cn-hangzhou.aliyuncs.com/xxx.png?Expires=xxx"
                        }
                    ]
                }
            }
        ],
        "audio": null
    },
    "usage": {
        "input_tokens": 0,
        "output_tokens": 0,
        "characters": 0,
        "height": 1536,
        "image_count": 2,
        "width": 1024
    }
}
```

</div>

</div>

</div>

</div>

<div id="88877f6a46eof" class="section section">

## Java

``` pre
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversation;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationParam;
import com.alibaba.dashscope.aigc.multimodalconversation.MultiModalConversationResult;
import com.alibaba.dashscope.common.MultiModalMessage;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.exception.ApiException;
import com.alibaba.dashscope.exception.NoApiKeyException;
import com.alibaba.dashscope.exception.UploadFileException;
import com.alibaba.dashscope.utils.JsonUtils;
import com.alibaba.dashscope.utils.Constants;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

public class QwenImageEdit {

    static {
        // The following URL is for the Singapore region. If you use a model in the Beijing region, replace the URL with https://dashscope.aliyuncs.com/api/v1.
        Constants.baseHttpApiUrl = "https://dashscope-intl.aliyuncs.com/api/v1";
    }
    
    // The API keys for the Singapore and Beijing regions are different. To obtain an API key, see https://www.alibabacloud.com/help/zh/model-studio/get-api-key.
    // If you have not configured the environment variable, replace the following line with your DashScope API key: apiKey="sk-xxx".
    static String apiKey = System.getenv("DASHSCOPE_API_KEY");

    public static void call() throws ApiException, NoApiKeyException, UploadFileException, IOException {

        MultiModalConversation conv = new MultiModalConversation();

        // The model supports one to three input images.
        MultiModalMessage userMessage = MultiModalMessage.builder().role(Role.USER.getValue())
                .content(Arrays.asList(
                        Collections.singletonMap("image", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/thtclx/input1.png"),
                        Collections.singletonMap("image", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/iclsnx/input2.png"),
                        Collections.singletonMap("image", "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/gborgw/input3.png"),
                        Collections.singletonMap("text", "The girl from Image 1 is wearing the black dress from Image 2 and sitting in the pose from Image 3.")
                )).build();
        // The qwen-image-2.0, qwen-image-edit-max, and qwen-image-edit-plus series models support one to six output images. This example generates two images.
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("watermark", false);
        parameters.put("negative_prompt", " ");
        parameters.put("n", 2);
        parameters.put("prompt_extend", true);
        parameters.put("size", "1024*1536");

        MultiModalConversationParam param = MultiModalConversationParam.builder()
                .apiKey(apiKey)
                .model("qwen-image-edit-max")
                .messages(Collections.singletonList(userMessage))
                .parameters(parameters)
                .build();

        MultiModalConversationResult result = conv.call(param);
        // To view the complete response, uncomment the following line.
        // System.out.println(JsonUtils.toJson(result));
        List<Map<String, Object>> contentList = result.getOutput().getChoices().get(0).getMessage().getContent();
        int imageIndex = 1;
        for (Map<String, Object> content : contentList) {
            if (content.containsKey("image")) {
                System.out.println("URL of output image " + imageIndex + ": " + content.get("image"));
                imageIndex++;
            }
        }
    }

    public static void main(String[] args) {
        try {
            call();
        } catch (ApiException | NoApiKeyException | UploadFileException | IOException e) {
            System.out.println(e.getMessage());
        }
    }
}
```

<div id="4aa34ebed3uil" class="section section">

<div id="fa326e07141nf" class="collapse" outputclass="collapse">

**Sample response**

<div id="02523e8589ss5" class="expandable-content"
tag="expandable-content">

``` pre
{
    "requestId": "46281da9-9e02-941c-ac78-be88b8xxxxxx",
    "usage": {
        "image_count": 2,
        "width": 1024,
        "height": 1536
    },
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                        },
                        {
                            "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                        }
                    ]
                }
            }
        ]
    }
}
```

</div>

</div>

</div>

</div>

<div id="8e57260438ub1" class="section section">

## **curl**

<div id="8836544e017az" class="section section" props="intl"
cond-props="intl">

The following command uses the URL for the Singapore region. If you use
a model in the China (Beijing) region, replace the URL with:
`https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation`

</div>

<div id="840842898d3s8" class="section section">

``` pre
curl --location 'https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $DASHSCOPE_API_KEY" \
--data '{
    "model": "qwen-image-2.0-pro",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/thtclx/input1.png"
                    },
                    {
                        "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/iclsnx/input2.png"
                    },
                    {
                        "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/gborgw/input3.png"
                    },
                    {
                        "text": "The girl from Image 1 is wearing the black dress from Image 2 and sitting in the pose from Image 3."
                    }
                ]
            }
        ]
    },
    "parameters": {
        "n": 2,
        "negative_prompt": " ",
        "prompt_extend": true,
        "watermark": false,
        "size": "1024*1536"
    }
}'
```

</div>

<div id="cde0d253e3goj" class="section section">

<div id="fc1853fee7cnz" class="collapse" outputclass="collapse">

**Response example**

<div id="8ffb0c5a3153w" class="expandable-content"
tag="expandable-content">

``` pre
{
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                        },
                        {
                            "image": "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
                        }
                    ]
                }
            }
        ]
    },
    "usage": {
        "width": 1536,
        "image_count": 2,
        "height": 1024
    },
    "request_id": "bf37ca26-0abe-98e4-8065-xxxxxx"
}
```

</div>

</div>

</div>

</div>

</div>

<div id="e2278796ed73n" class="collapse" outputclass="collapse">

**Download images to your local device**

<div id="f99da94284yqk" class="expandable-content"
tag="expandable-content">

<div id="9f8431a446lcq" class="tabbed-codeblock-box"
outputclass="tabbed-codeblock" tag="fig">

<div class="tab-box">

</div>

<div class="codeblock-item">

``` pre
# You need to install requests to download the image: pip install requests
import requests


def download_image(image_url, save_path='output.png'):
    try:
        response = requests.get(image_url, stream=True, timeout=300)  # Set a timeout.
        response.raise_for_status()  # Raise an exception if the HTTP status code is not 200.
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Image successfully downloaded to: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Image download failed: {e}")


image_url = "https://dashscope-result-sz.oss-cn-shenzhen.aliyuncs.com/xxx.png?Expires=xxx"
download_image(image_url, save_path='output.png')
```

</div>

<div class="codeblock-item">

``` pre
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
 
public class ImageDownloader {
    public static void downloadImage(String imageUrl, String savePath) {
        try {
            URL url = new URL(imageUrl);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(300000);
            connection.setRequestMethod("GET");
            InputStream inputStream = connection.getInputStream();
            FileOutputStream outputStream = new FileOutputStream(savePath);
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();
 
            System.out.println("Image downloaded successfully to: " + savePath);
        } catch (Exception e) {
            System.err.println("Image download failed: " + e.getMessage());
        }
    }
 
    public static void main(String[] args) {
        String imageUrl = "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/xxx?Expires=xxx";
        String savePath = "output.png";
        downloadImage(imageUrl, savePath);
    }
}
```

</div>

</div>

</div>

</div>

</div>

<div id="418c71d0b07el" class="section section">

## **Model recommendations**

- <div id="d3e4f5a6b7c02">

  **`qwen-image-2.0-pro`** **series (Recommended):** A fused model for
  image generation and editing with enhanced capabilities in text
  rendering, realistic textures, and semantic adherence.

  </div>

- <div id="d3e4f5a6b7c05">

  **`qwen-image-2.0`** **series:** An accelerated version of the fused
  image generation and editing model that balances quality and
  performance.

  </div>

For the models supported in each region, see
<span props="intl"><a href="https://www.alibabacloud.com/help/en/model-studio/models#809eb92b1fyko" id="b2527a74ecejv"
class="xref">Model list</a></span>.

</div>

<div id="cfa18c76f3wii" class="section section">

## **Input instructions**

<div id="fe3b38309f65z" class="section section">

### **Input images (messages)**

The `messages` parameter is an array that must contain a single object.
This object must include the `role` and `content` properties. The `role`
property must be set to `user`. The `content` property must include both
`image` (one to three images) and `text` (one editing instruction).

The input images must meet the following requirements:

- <div id="4bc11a9e7akc9">

  The supported image formats are JPG, JPEG, PNG, BMP, TIFF, WEBP, and
  GIF.

  > The output image is in PNG format. For animated GIFs, only the first
  > frame is processed.

  </div>

- <div id="bd0b3fa03fien">

  For best results, the image resolution should be between 384 and 3072
  pixels for both width and height. A low resolution may result in a
  blurry output, while a high resolution increases processing time.

  </div>

- <div id="8d41b31f23fio">

  The size of a single image file cannot exceed 10 MB.

  </div>

``` pre
"messages": [
    {
        "role": "user",
        "content": [
            { "image": "Public URL or Base64 data of Image 1" },
            { "image": "Public URL or Base64 data of Image 2" },
            { "image": "Public URL or Base64 data of Image 3" },
            { "text": "Your editing instruction, for example: 'The girl in Image 1 wears the black dress from Image 2 and sits in the pose from Image 3'" }
        ]
    }
]
```

</div>

<div id="b4157792f1m0a" class="section section">

### **Image input order**

When you provide multiple input images, their order is defined by their
sequence in the array. The editing instruction must correspond to the
order of the images in the `content` field, such as 'Image 1' and 'Image
2'.

<table id="a9affa756bw45" class="table" data-tablewidth="100"
data-tablecolswidth="25 25 25 25" data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="3a4b00b5bb24g" class="odd">
<td id="23512e5969hgq"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Input
image 1</strong></p></td>
<td id="578fed11a4rpp"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Input
image 2</strong></p></td>
<td colspan="2" id="702dd3c2canub"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="76869cf1d10gt" class="even">
<td id="b7a3d0bd55arj" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011989.webp"
id="54c0053792qqw" class="image break" data-placement="break"
width="250" alt="image95" /></p></td>
<td id="82f06e78f46no" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011988.webp"
id="5c90c995aejze" class="image break" data-placement="break"
width="250" alt="image96" /></p></td>
<td id="b14f3fac31poy" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1012120.webp"
id="e485073efbg2s" class="image break" data-placement="break"
width="250" alt="5" /></p>
<p>Replace the clothes of the girl in Image 1 with the clothes of the
girl in Image 2.</p></td>
<td id="8fa7069b2ebhh" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1012119.webp"
id="33bc27ec74j33" class="image break" data-placement="break"
width="250" alt="4" /></p>
<p>Replace the clothes of the girl in Image 2 with the clothes of the
girl in Image 1.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="c110bdcd2erhh" class="section section">

### **Image input methods**

**Public URL**

- <div id="39d9abd354k1u">

  You can provide a publicly accessible image URL that supports the HTTP
  or HTTPS protocol.

  </div>

- <div id="a183b3ae5fjew">

  Example value: `https://xxxx/img.png`.

  </div>

**Base64 encoding**

Convert the image file to a Base64-encoded string and concatenate it in
the following format: `data:{mime_type};base64,{base64_data}`.

- <div id="61afe6f424yqf">

  `{mime_type}`: The media type of the image, which must correspond to
  the file format.

  </div>

- <div id="a02fd438268pq">

  `{base64_data}`: The Base64-encoded string of the file.

  </div>

- <div id="ff2b103459ewu">

  Example value: `data:image/jpeg;base64,GDU7MtCZz...` (The example is
  truncated for demonstration purposes.)

  </div>

For complete code examples, see
<a href="https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-api#a3ad9a3b6d9if"
id="98543900ee1dx" class="xref">Python SDK call</a> and
<a href="https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-api#589b80853e6rn"
id="5fe76d2ca41ne" class="xref">Call using the Java SDK</a>.

</div>

<div id="ac07d69b4ag98" class="section section">

### **More parameters**

Adjust the generation results using the following **optional**
parameters:

- <div id="59d7bd7024n7r">

  **n**: The number of images to generate. The default value is 1. The
  qwen-image-2.0, qwen-image-edit-max, and qwen-image-edit-plus series
  of models support generating one to six images. The `qwen-image-edit`
  model supports generating only one image.

  </div>

- <div id="5ee8224addvay">

  **negative_prompt**: Describes content to exclude from the image, such
  as "blur" or "extra fingers". This parameter helps optimize the
  quality of the generated image.

  </div>

- <div id="2f11c7318bi7u">

  **watermark**: Specifies whether to add a "Qwen-Image" watermark to
  the bottom-right corner of the image. The default value is `false`.
  The following image shows the watermark style:

  <img
  src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1012089.jpg"
  id="ca03175b6b1yk" class="image inline" data-init-id="1f43d4c796wzz"
  data-placement="inline" width="117" height="35" alt="1" />

  </div>

- <div id="0be6fb360cd83">

  **seed**: The random number seed. The value must be an integer from
  `[0, 2147483647]`. If this parameter is not specified, the algorithm
  generates a random number to use as the seed. Using the same seed
  value helps ensure consistent generation results.

  </div>

The following **optional** parameters are available only for the
qwen-image-2.0, qwen-image-edit-max, and qwen-image-edit-plus series of
models:

- <div id="df97c73408sj6">

  **size**: The resolution of the output image. The format is
  `width*height`, such as `"1024*2048"`. For the qwen-image-2.0 series
  models, you can set the width and height freely. The total pixels of
  the output image must be between 512 × 512 and 2048 × 2048. By
  default, the resolution is the same as the input image (the last image
  if multiple are provided). For the qwen-image-edit-max and
  qwen-image-edit-plus series models, the width and height can range
  from 512 to 2048 pixels. By default, the output image has a resolution
  close to `1024*1024` and an aspect ratio similar to the original
  image.

  </div>

- <div id="918842e101ers">

  **prompt_extend:** Enables or disables the prompt rewriting feature.
  The default value is `true`. If enabled, the model optimizes the
  prompt. This feature can significantly improve the results for simple
  or less descriptive prompts.

  </div>

For a complete list of parameters, see
<a href="https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-api" id="0d6b2d2723e82"
class="xref">Qwen-Image-Edit API reference</a>.

</div>

</div>

<div id="c90a7d9194k2z" class="section section">

## **Overview**

<div id="f319c68261oqy" class="section section">

### **Multi-image fusion**

<table id="cda0f59ff3uk3" class="table" data-tablewidth="96"
data-tablecolswidth="24 24 24 24" data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="29e405be96g3n" class="odd">
<td id="7e5fda09f2lwe"
style="background-color: #e5e5e5"><p><strong>Input image
1</strong></p></td>
<td id="e047bfb2601rr"
style="background-color: #e5e5e5"><p><strong>Input image
2</strong></p></td>
<td id="f154597c3dls0"
style="background-color: #e5e5e5"><p><strong>Input image
3</strong></p></td>
<td id="af62592d3f8m6"
style="background-color: #e5e5e5"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="6ebd26dcc910u" class="even">
<td id="fd28558199csb" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011712.webp"
id="6be95a1ea95mi" class="image break" data-placement="break"
width="250" alt="image83" /></p></td>
<td id="1fbd424213jm9" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011753.webp"
id="65bbc952d0nxy" class="image break" data-placement="break"
width="250" alt="image103" /></p></td>
<td id="b9fb392ea1zg6" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/4405461671/p1012002.webp"
id="f5fe21667dhom" class="image break" data-placement="break"
width="250" alt="1" /></p></td>
<td id="e360d0cf63ugv" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/4405461671/p1012004.webp"
id="cd79136e52wv3" class="image break" data-placement="break"
width="250" alt="2" /></p>
<p>The girl in Image 1 wears the necklace from Image 2 and carries the
bag from Image 3 on her left shoulder.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="33ac7fee16bml" class="section section">

### **Subject consistency**

<div id="a231f89af3ilx" class="section section">

<table id="1100dbba213i4" class="table" data-tablewidth="100"
data-tablecolswidth="25 25 25 25" data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="0a3f02fd14hdk" class="odd">
<td id="55b5621358vkp"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Input
image</strong></p></td>
<td id="758c8efcd5xwj"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image 1</strong></p></td>
<td id="d89838dd177en"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image 2</strong></p></td>
<td id="002609568cep7"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image 3</strong></p></td>
</tr>
<tr id="58df5e2c7bcaf" class="even">
<td rowspan="2" id="018576e71dz59" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011789.webp"
id="ece000fa886dn" class="image break" data-placement="break"
width="250" alt="image5" /></p></td>
<td rowspan="2" id="7f1248b704xsw" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011790.webp"
id="26e08c978eo7m" class="image break" data-placement="break"
width="250" alt="image4" /></p>
<p>Change the image to an ID photo with a blue background. The person is
wearing a white shirt, a black suit, and a striped tie.</p></td>
<td rowspan="2" id="213031e4ef69r" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011791.webp"
id="d02b384ff4zdt" class="image break" data-placement="break"
width="250" alt="image6" /></p>
<p>The person is wearing a white shirt, a gray suit, and a striped tie.
One hand rests on the tie. The background is light-colored.</p></td>
<td rowspan="2" id="a9d77d5eb9l3h" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011792.webp"
id="309326b5a07l3" class="image break" data-placement="break"
width="250" alt="image7" /></p>
<p>The person is wearing a black hoodie with "Qwen Image" in a thick
brushstroke font. They are leaning on a guardrail with sunlight in their
hair. A bridge and the sea are in the background.</p></td>
</tr>
<tr id="9b051b317amde" class="odd" style="height:33px">
</tr>
<tr id="d46e3d902dkvj" class="even">
<td rowspan="2" id="79fea55904w2a" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1012037.webp"
id="83f87e2a27emn" class="image break" data-placement="break"
width="250" alt="image12" /></p></td>
<td rowspan="2" id="1371d2d63bnf3" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1012038.webp"
id="ffe0f5b2acwhs" class="image break" data-placement="break"
width="250" alt="image13" /></p>
<p>The air conditioner is placed in a living room next to a
sofa.</p></td>
<td rowspan="2" id="bcea60a58dc3t" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1012039.webp"
id="ac5993a6dewil" class="image break" data-placement="break"
width="250" alt="image14" /></p>
<p>Mist is added from the air conditioner's vent, extending over the
sofa. Green leaves are also added.</p></td>
<td rowspan="2" id="64b70120e5lrx" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1012040.webp"
id="e18dd3e57fusn" class="image break" data-placement="break"
width="250" alt="image15" /></p>
<p>The white handwritten text "自然新风 畅享呼吸" is added at the
top.</p></td>
</tr>
<tr id="fa78a879earel" class="odd" style="height:33px">
</tr>
</tbody>
</table>

</div>

</div>

<div id="4feadcda55a5u" class="section section">

### Sketch creation

<table id="72d5bad3eeoii" class="table" data-tablewidth="99"
data-tablecolswidth="33 33 33" data-autofit="true">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody class="tbody">
<tr id="f14557ee0eprp" class="odd">
<td id="a3656c2bc3ujv"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Input
image</strong></p></td>
<td colspan="2" id="01ebed39db5wn"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="0cbab26d2a5vc" class="even">
<td id="06e6706fc3116" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011821.webp"
id="4bb41a0775p7x" class="image break" data-placement="break"
width="300" alt="image42" /></p></td>
<td id="1e6edb1bc48h8" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011822.webp"
id="85ca83a5a4652" class="image break" data-placement="break"
width="300" alt="image43" /></p>
<p>Generate an image that matches the detailed shape outlined in Image 1
and follows this description: A young woman smiles on a sunny day. She
wears round brown sunglasses with a leopard print frame. Her hair is
neatly tied up, she wears pearl earrings, a dark blue scarf with purple
star patterns, and a black leather jacket.</p></td>
<td id="bfb26ca6f3rcl" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011824.webp"
id="e621639f4eykw" class="image break" data-placement="break"
width="300" alt="image44" /></p>
<p>Generate an image that matches the detailed shape outlined in Image 1
and follows this description: An elderly man smiles at the camera. His
face is wrinkled, his hair is messy in the wind, and he wears
round-framed reading glasses. He has a worn-out red scarf with star
patterns around his neck and is wearing a cotton-padded jacket.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="128ea035c4fii" class="section section">

<div id="fddad781bd3dy" class="section section">

### **Creative product generation**

<table id="bceec883c2yhz" class="table" data-tablewidth="100"
data-tablecolswidth="25 25 25 25" data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="1c6513e32cbp4" class="odd">
<td id="85dd665fb7qiq"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Input
image</strong></p></td>
<td colspan="3" id="55a210d141va3"
style="background-color: #e5e5e5"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="3c4aa1d21fota" class="even">
<td rowspan="2" id="5b6660d8a1yfp"
style="vertical-align: middle"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999719.png"
id="89fa0f7d66heb" class="image break" data-placement="break"
width="300" alt="图片 1" /></p></td>
<td id="32809e1318cir" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011685.webp"
id="3222a4305elfs" class="image break" data-placement="break"
width="300" alt="image23" /></p>
<p>Make this bear sit under the moon (represented by a light gray
crescent outline on a white background), holding a guitar, with small
stars and speech bubbles with phrases such as "Be Kind" floating
around.</p></td>
<td id="31c8875c43rv0" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011686.webp"
id="5fdf31eb19tcs" class="image break" data-placement="break"
width="300" alt="image22" /></p>
<p>Print this design on a T-shirt and a paper tote bag. A female model
is displaying these items. The woman is also wearing a baseball cap with
"Be kind" written on it.</p></td>
<td id="293da59419pux" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011687.webp"
id="532ef095c61v6" class="image break" data-placement="break"
width="300" alt="image21" /></p>
<p>A hyper-realistic 1/7 scale character model, designed as a commercial
finished product, is placed on a desk with an iMac that has a white
keyboard. The model stands on a clean, round, transparent acrylic base
with no labels or text. Professional studio lighting highlights the
sculpted details. On the iMac screen in the background, the ZBrush
modeling process for the same model is displayed. Next to the model,
place a packaging box with a transparent window on the front, showing
only the clear plastic shell inside. The box is slightly taller than the
model and reasonably sized to hold it.</p></td>
</tr>
<tr id="3ba1d62a8f8dc" class="odd">
<td id="5c84c71a1c3vk" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999632.png"
id="0262f125b8zsk" class="image break" data-placement="break"
width="300" alt="image" /></p>
<p>This bear is wearing an astronaut suit and pointing into the
distance.</p></td>
<td id="25494a07cf5wb" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999633.png"
id="05c580d5bd1g0" class="image break" data-placement="break"
width="300" alt="image" /></p>
<p>This bear is wearing a gorgeous ball gown, with its arms spread in an
elegant dance pose.</p></td>
<td id="20ae9d41f54qp" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999630.png"
id="c37642d8dcmu6" class="image break" data-placement="break"
width="300" alt="image" /></p>
<p>This bear is wearing sportswear, holding a basketball, with one leg
bent.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="3fd397ff4fgbf" class="section section">

### Generate image from depth map

<table id="73d3d0ada8rck" class="table" data-tablewidth="99"
data-tablecolswidth="33 33 33" data-autofit="true">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody class="tbody">
<tr id="04e95324d6x64" class="odd">
<td id="e4b9e099d1pd2"
style="background-color: #e5e5e5"><p><strong>Input
image</strong></p></td>
<td colspan="2" id="5058873792to3"
style="background-color: #e5e5e5"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="a65ad3a54afml" class="even">
<td id="252d5afd4c7ae" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011810.webp"
id="b3d73108fa4lk" class="image break" data-placement="break"
width="300" alt="image36" /></p></td>
<td id="43556243cfvc9" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011811.webp"
id="d7118ac3ea0zo" class="image break" data-placement="break"
width="300" alt="image37" /></p>
<p>Generate an image that matches the depth map outlined in Image 1 and
follows this description: A blue bicycle is parked in a side alley, with
a few weeds growing from cracks in the stone in the background.</p></td>
<td id="f1986fe1e4tbs" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011812.webp"
id="c8b41c8747gix" class="image break" data-placement="break"
width="300" alt="image38" /></p>
<p>Generate an image that matches the depth map outlined in Image 1 and
follows this description: A worn-out red bicycle is parked on a muddy
path, with a dense primeval forest in the background.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="e362a38799sv5" class="section section">

### Generate image from keypoints

<table id="00b2693ec5tuu" class="table" data-tablewidth="99"
data-tablecolswidth="33 33 33" data-autofit="true">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody class="tbody">
<tr id="9f8da12f7a8x3" class="odd">
<td id="ea4be86cfcluy"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Input
image</strong></p></td>
<td colspan="2" id="32aea33057uoc"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="22e5dfb79e7xg" class="even">
<td id="9e3a4e66a2n4l" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011817.webp"
id="93ea93e4a15mn" class="image break" data-placement="break"
width="300" alt="image40" /></p></td>
<td id="85d9e5105e7x1" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011818.webp"
id="a8c5f789f1yy5" class="image break" data-placement="break"
width="300" alt="image41" /></p>
<p>Generate an image that matches the human pose outlined in Image 1 and
follows this description: A Chinese woman in a Hanfu is holding an
oil-paper umbrella in the rain, with a Suzhou garden in the
background.</p></td>
<td id="578f0b0607kql" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011819.webp"
id="c86626c7669gl" class="image break" data-placement="break"
width="300" alt="image39" /></p>
<p>Generate an image that matches the human pose outlined in Image 1 and
follows this description: A young man stands on a subway platform. He
wears a baseball cap, a T-shirt, and jeans. A train is speeding by
behind him.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="ab5f7e9f13c0t" class="section section">

### **Text editing**

<table id="cc31126f919gr" class="table" data-tablewidth="100"
data-tablecolswidth="25.03 25.03 25.03 24.909999999999997"
data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="93aff62762m0m" class="odd">
<td id="37595dad15p7e"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Input
image</strong></p></td>
<td id="3703c45173dfg"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Output
image</strong></p></td>
<td id="1f6f2fd76cxzf"
style="background-color: #e5e5e5"><p><strong>Input
image</strong></p></td>
<td id="f45e5a29f991z"
style="background-color: #e5e5e5"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="ff000fa84036u" class="even">
<td id="f1f00900f3v6z" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999641.png"
id="cbe119c3d2djo" class="image break"
data-comment_a358430d-d09c-4db7-bde7-cb42baa503b2="comment"
data-placement="break" width="250" alt="image" /></p></td>
<td id="16c3addbe6kwb" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999642.png"
id="6b8c16fad6a3i" class="image break"
data-comment_a358430d-d09c-4db7-bde7-cb42baa503b2="comment"
data-placement="break" width="250" alt="image" /></p>
<p>Replace 'HEALTH INSURANCE' on the Scrabble tiles with
'<strong>明天会更好</strong>'.</p></td>
<td id="9f6f8615da3do" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1000039.png"
id="61dbf7b6a24if" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="f1d487d236zw1" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1000062.png"
id="a481a79d9e68e" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Change the phrase "Take a Breather" on the note to "<strong>Relax and
Recharge</strong>".</p></td>
</tr>
</tbody>
</table>

<table id="61cb7d3437j3m" class="table" data-tablewidth="100"
data-tablecolswidth="25 25 25 25" data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="de03855ec2s4g" class="odd">
<td id="f89d06349crhs"
style="background-color: #e5e5e5"><p><strong>Input
image</strong></p></td>
<td colspan="3" id="73b0cdc0fb9pu"
style="background-color: #e5e5e5"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="cea1266171rzq" class="even">
<td rowspan="3" id="e0b3a1f5a5ek4"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011772.webp"
id="5f85c121fe4yd" class="image break" data-placement="break"
alt="image53" /></p></td>
<td id="cd170a2fb2lxn"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011774.webp"
id="da297543c3qih" class="image break" data-placement="break"
alt="image45" /></p>
<p>Change "Qwen-Image" to a black ink-drip font.</p></td>
<td id="cb164f223ch1m"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011775.webp"
id="f406054bb5xxa" class="image break" data-placement="break"
alt="image46" /></p>
<p>Change "Qwen-Image" to a black handwriting font.</p></td>
<td id="21d21b802b221"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011776.webp"
id="f84d1b55b32o1" class="image break" data-placement="break"
alt="image49" /></p>
<p>Change "Qwen-Image" to a black pixel font.</p></td>
</tr>
<tr id="3c3575280bobx" class="odd">
<td id="f10f533df67nn"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011777.jpeg"
id="3aecb214d3lyz" class="image break" data-placement="break"
alt="image54" /></p>
<p>Change "Qwen-Image" to red.</p></td>
<td id="02c4a8f05bv01"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011779.jpeg"
id="ec0e923bd5v0p" class="image break" data-placement="break"
alt="image57" /></p>
<p>Change "Qwen-Image" to a blue-purple gradient.</p></td>
<td id="de37c89a5d6u0"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011780.jpeg"
id="c4896c1350at0" class="image break" data-placement="break"
alt="image59" /></p>
<p>Change "Qwen-Image" to candy colors.</p></td>
</tr>
<tr id="30f76fac0amzw" class="even">
<td id="82033c44aarvz"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011783.webp"
id="c68aac0f98hw3" class="image break" data-placement="break"
alt="image63" /></p>
<p>Change the material of "Qwen-Image" to metal.</p></td>
<td id="039d600a401yp"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011784.webp"
id="4e653d3aa8xsg" class="image break" data-placement="break"
alt="image64" /></p>
<p>Change the material of "Qwen-Image" to clouds.</p></td>
<td id="6fe1d09f4bf39"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011786.webp"
id="f3401faf983hv" class="image break" data-placement="break"
alt="image67" /></p>
<p>Change the material of "Qwen-Image" to glass.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="e1a7cc7da089f" class="section section">

### **Add, delete, modify, and replace**

<table id="af9e58f76baoe" class="table" data-tablewidth="100"
data-tablecolswidth="20 40 40" data-autofit="true">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody class="tbody">
<tr id="b26a552bf98c7" class="odd">
<td id="a4203a9344ka8"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Capability</strong></p></td>
<td id="ab72cbb4b2yft"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Input
image</strong></p></td>
<td id="04f1dad895fwu"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="922d1a9e94rm1" class="even">
<td id="dc97c62a53qab" style="vertical-align: middle"><p><strong>Add
element</strong></p></td>
<td id="8ba3d566fafj5" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999647.png"
id="97161a9214jkz" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="a6d6e75babxzq" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999648.png"
id="9b07a998c70mj" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Add a small wooden sign in front of the penguin that says "Welcome to
Penguin Beach".</p></td>
</tr>
<tr id="301063eaffpyf" class="odd">
<td id="65e01884aeuf7" style="vertical-align: middle"><p><strong>Delete
element</strong></p></td>
<td id="34254a3fb0kp5" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999649.png"
id="af67ca028703k" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="f5a9eb5869f2a" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999650.png"
id="14a7e428a3cak" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Remove the hair from the plate.</p></td>
</tr>
<tr id="0ebf67dae5nf1" class="even">
<td id="0658429445itl" style="vertical-align: middle"><p><strong>Replace
element</strong></p></td>
<td id="1dd5c115638eu" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999696.png"
id="9d11f6a776noy" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="54f5501610iui" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999698.png"
id="9f2427a14bkw0" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Change the peaches to apples.</p></td>
</tr>
<tr id="467ce8aefe9zp" class="odd">
<td id="79cbcc19bcjnk"
style="vertical-align: middle"><p><strong>Portrait
modification</strong></p></td>
<td id="1976ddbe532i1" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999657.png"
id="e308b098caksz" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="6d43113d0dkkd" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999695.png"
id="8063b9e3799jh" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Make her close her eyes.</p></td>
</tr>
<tr id="efd07e1bfbz3v" class="even">
<td id="eef5d34c3a8wg"><p><strong>Pose modification</strong></p></td>
<td id="0a83d12745air" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011690.webp"
id="9905c05bf8mko" class="image break" data-placement="break"
width="250" height="447" alt="image8" /></p></td>
<td id="d05958708b5wv" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011691.webp"
id="779f3ad2786bf" class="image break" data-placement="break"
width="250" height="447" alt="image9" /></p>
<p>She raises her hands with palms facing the camera and fingers spread
in a playful pose.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="3698d5e344uvu" class="section section">

### **Viewpoint transformation**

<table id="4ad8a7848cm6u" class="table" data-tablewidth="100"
data-tablecolswidth="25 25 25 25" data-autofit="true">
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<tbody class="tbody">
<tr id="0444ba716932h" class="odd">
<td id="092814f8c29sb"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Input
image</strong></p></td>
<td id="15e933201fiyg"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Output
image</strong></p></td>
<td id="f63a1f8580kra"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Input
image</strong></p></td>
<td id="9746127b485qn"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="3ec02cd978a3i" class="even">
<td id="68ee0643a8vgr" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999964.png"
id="bd64e0472dmjq" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="6271d1c15d9mn" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999968.png"
id="e9d0320c71t7f" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Get a front view.</p></td>
<td id="3ebde4336753g" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999969.png"
id="5ebff36e69jtq" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="76c522b5ecxev" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999970.png"
id="75638d89b977q" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Face left.</p></td>
</tr>
<tr id="cdc6dba2f8qwp" class="odd">
<td id="986369f81dsrs" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999974.png"
id="22c0b997237vc" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="129fcf4effe91" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999975.png"
id="cd4ff23221tvg" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Get a rear view.</p></td>
<td id="8e88b6f38c9b7" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999971.png"
id="52481d31f527l" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="140abc6e65dw3" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999972.png"
id="a2cfab7f5c7ps" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Face right.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="03d3c85d46xsc" class="section section">

### **Background replacement**

<table id="814ac22bf56hr" class="table" data-tablewidth="100"
data-tablecolswidth="33.35 33.35 33.3" data-autofit="true">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody class="tbody">
<tr id="25c4707baeav4" class="odd">
<td id="56eb1362d6jae"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Input
image</strong></p></td>
<td colspan="2" id="8a32b81c8bfq3"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="a2821a8cdanzu" class="even">
<td id="afad3e21e8bo5" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p999639.png"
id="a642b08f2axru" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="effe02bd7fuql" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1000030.png"
id="055e01c9c6jgp" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Change the background to a beach.</p></td>
<td id="724a069718pkc" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999640.png"
id="1d4ae666987ry" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Replace the original background with a realistic modern classroom
scene. In the center of the background is a traditional dark green or
black blackboard. The Chinese characters "Qwen" are neatly written on
the blackboard in white chalk.</p></td>
</tr>
</tbody>
</table>

</div>

<div id="03f071047dvho" class="section section">

### **Old photo processing**

<table id="7bc7bd7a47cwd" class="table" data-tablewidth="100"
data-tablecolswidth="20.04 40.01 39.95" data-autofit="true">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<tbody class="tbody">
<tr id="a9903f2c388be" class="odd">
<td id="10013ea89dwdq"
style="background-color: #e5e5e5; vertical-align: middle"><p><strong>Capability</strong></p></td>
<td id="a5426d9fb0cf8"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Input
image</strong></p></td>
<td id="f40f7935edum4"
style="background-color: #e5e5e5; vertical-align: top"><p><strong>Output
image</strong></p></td>
</tr>
<tr id="2c6aad9a4729b" class="even">
<td rowspan="2" id="910ead5bcbam2"
style="vertical-align: middle"><p><strong>Old photo restoration and
colorization</strong></p></td>
<td id="3305981ec9phc" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999552.png"
id="3a13835687veu" class="image break" data-placement="break"
width="250" alt="image" /></p></td>
<td id="5bafc8dfb9dq1" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p999554.png"
id="336e758a486zm" class="image break" data-placement="break"
width="250" alt="image" /></p>
<p>Restore the old photo, remove scratches, reduce noise, enhance
details, high resolution, realistic image, natural skin tone, clear
facial features, no distortion.</p></td>
</tr>
<tr id="6bb4700911rko" class="odd">
<td id="8fd026ce8arr6" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/5844029571/p1011757.webp"
id="adc865be9ads6" class="image break" data-placement="break"
width="250" alt="image31" /></p></td>
<td id="467fcde782h3b" style="vertical-align: top"><p><img
src="https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/6844029571/p1011759.webp"
id="1bb50e1a5cdnx" class="image break" data-placement="break"
width="250" alt="image32" /></p>
<p>Intelligently colorize the image based on its content to make it more
vivid.</p></td>
</tr>
</tbody>
</table>

</div>

</div>

</div>

<div id="f99795f0638ax" class="section section">

## **Billing and rate limiting**

See
<span props="intl"><a href="https://www.alibabacloud.com/help/en/model-studio/models#809eb92b1fyko" id="9d0089a0ca797"
class="xref">Model list and pricing</a></span> for the free quota,
pricing, and
<span props="intl"><a href="https://www.alibabacloud.com/help/en/model-studio/rate-limit#11371335d3feh"
id="35a47c36a3dnv" class="xref">rate limits</a></span>.

**Billing details:**

- <div id="bcf0f7c917d18">

  Billing is based on successfully generated images only. Failed calls
  do not incur fees or consume the free quota.

  </div>

- <div id="dc3f36554en0e">

  Enable 'Free quota only' to avoid charges after the quota is depleted.
  See <a href="https://www.alibabacloud.com/help/en/model-studio/new-free-quota" id="a1fe6e3b935b3"
  class="xref">Free quota for new users</a>.

  </div>

</div>

<div id="eea0c1fcf9qf7" class="section section">

## **API reference**

See
<a href="https://www.alibabacloud.com/help/en/model-studio/qwen-image-edit-api" id="bda7da790aaa9"
class="xref">Qwen-Image Edit</a> for API parameters.

</div>

<div id="3e1e579599soq" class="section section" docid="4944974"
is-conref="true">

## **Error codes**

If the model call fails and returns an error message, see
<a href="https://www.alibabacloud.com/help/en/model-studio/error-code" id="d62f02c305acr"
class="xref">Error messages</a> for resolution.

</div>

<div id="2d3c365129zh8" class="section section">

## **FAQ**

<div id="5eb914c827539" class="section section" docid="5931840"
is-conref="true">

### **Q: What languages does the Qwen Image Editing model support?**

A: The model currently supports **Simplified Chinese and English**. You
can try other languages, but performance is not guaranteed.

<div id="129f7dfb3axu1" class="section section" docid="6637149"
is-conref="true">

#### **Q: How do I view model invocation metrics?**

<div id="2d06e060e42ac" class="section section">

A: One hour after a model invocation completes, go to the
<span id="b88ba1341aa27" class="ph" tag="ph" props="intl"
cond-props="intl"><a
href="https://modelstudio.console.alibabacloud.com/?tab=dashboard#/model-telemetry"
id="fd3b8e5b85447"><strong>Monitoring</strong> (Singapore)</a> or <a
href="https://bailian.console.alibabacloud.com/?tab=model#/model-telemetry"
id="5f9c1332949gi"><strong>Monitoring</strong> (China (Beijing))</a></span>
page to view metrics such as invocation count and success rate. For more
information, see
<a href="https://www.alibabacloud.com/help/en/model-studio/bill-query-and-cost-management"
id="6fa2c5a755h21" class="xref">Bill query and cost management</a>.

</div>

</div>

<div id="6b49b1be69q4c" class="section section" docid="6625971"
is-conref="true">

#### **Q: How do I get the domain name whitelist for image storage?**

A: Images generated by models are stored in OSS. The API returns a
temporary public URL. **To configure a firewall whitelist for this
download URL**, note the following: The underlying storage may change
dynamically. This topic does not provide a fixed OSS domain name
whitelist to prevent access issues caused by outdated information. If
you have security control requirements, contact your account manager to
obtain the latest OSS domain name list.

</div>

</div>

See the <a href="https://www.alibabacloud.com/help/en/model-studio/image-faq" id="28cc6c191fy8o"
class="xref">Image generation FAQ</a>.

</div>

</div>

</div>

</div>

<div class="nav-footer">

</div>

<div class="alicloud-document-ask clearfix">

</div>

<div class="feedback-message" style="display:none;">

<span class="hasfeedback acon acon-done-16"></span> Thank you! We've
received your feedback.

</div>

</div>

<div class="col-log-2 col-md-2 col-sm-0 col-xs-0 nav-catalog">

<div id="articleCatalog" class="article-catalog">

</div>

</div>

</div>

</div>

</div>

</div>

</div>
