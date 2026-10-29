# FLUX.2 Text to Image

Reviewed: 2026-10-29. This source is the FLUX.2 family API page, not a dedicated
Klein prompting handbook. Klein receives the supplied prompt without automatic
expansion: write coherent prose with explicit lighting and scene details.
See the [current family reference](https://docs.bfl.ml/guides/prompting_unified_reference.md)
and [Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
Base and distilled checkpoints require their own sampling settings. The hosted
Klein limit is four references; max/pro/flex limits in this snapshot do not
apply to Klein. [FLUX 3 Klein](flux3_prompting.md) is a separate generation.

Vendored snapshot from: https://docs.bfl.ml/flux_2/flux2_text_to_image

Fetched: 2026-06-03

Note: Original `llms.txt` URL `https://docs.bfl.ml/guides/prompting_guide_flux2_klein` returned 404; this current BFL page covers FLUX.2 text-to-image including [klein].

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bfl.ml/llms.txt
> Use this file to discover all available pages before exploring further.

# FLUX.2 Text to Image

> FLUX.2 is the recommended model for text-to-image generation. Generate high-fidelity images with advanced control, exact colors, and flexible aspect ratios.

**FLUX.2** brings enterprise-grade efficiency and professional precision to text-to-image generation. It closes the gap between generated and real imagery with accurate hands, faces, and textures—all while respecting brand guidelines through hex-code color steering. **\[max]** offers the highest quality with grounding search for real-time information.

<Tip>
  **Try it live** — Test FLUX.2 \[max], \[pro], \[flex] and \[klein] in the [playground](https://playground.bfl.ai). \[klein] is also available on [Hugging Face](https://huggingface.co/black-forest-labs).
</Tip>

## Examples

### Photorealistic

<Frame>
  <img src="https://cdn.sanity.io/images/2gpum2i6/production/4088463e10bb2a5d7591d30b26bdb5f8775d6167-1440x752.png" alt="Cinematic landscape with sentient tree" />
</Frame>

<Prompt description="At high noon on a blustery day, capture the surreal presence of a sentient tree, seemingly rooted underwater just off a tumultuous ocean shore. Employ a sweeping panning shot, bathing the scene in cinematic natural light and a stark palette of winter whites and greys, as if glimpsing a spectral sentinel through a watery veil." actions={[]} />

### Typography & Design

<Frame>
  <img src="https://cdn.sanity.io/images/2gpum2i6/production/7730d0482ee19d0d3207b5cbbb55074c3b66d554-1456x1920.png" alt="Samsung Galaxy S25 Ultra product advertisement" />
</Frame>

<Prompt description="Samsung Galaxy S25 Ultra product advertisement, 'Ultra-strong titanium' headline, 'Shielded in a strong titanium frame, your Galaxy S25 Ultra always stays protected' subtext, close-up of phone edge showing titanium frame, dark gradient background, clean minimalist tech aesthetic, professional product photography" actions={[]} />

<AccordionGroup>
  <Accordion title="Photorealism & Detail">
    <Columns cols={2}>
      <Frame caption="Cinematic lighting">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/98f466fde69d875943d47cc6238c401d09780537-1440x1152.png" alt="Cinematic lighting portrait" />
      </Frame>

      <Frame caption="Realistic skin texture and lighting">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/e62db22fa2b77c59d28d02d71133623bf1153cdc-1920x1920.jpg" alt="Realistic skin texture close-up" />
      </Frame>
    </Columns>

    <Columns cols={2}>
      <Frame caption="Hyper-realistic close up">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/20370abfd1d798f34419ccd241a068ca1c997ed2-1552x656.png" alt="Hyper-realistic close-up portrait" />
      </Frame>

      <Frame caption="Product photography quality">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/8be740a34ccb79731d3c4f59e67b73ca871b1d06-1552x656.png" alt="Product photography quality output" />
      </Frame>
    </Columns>
  </Accordion>

  <Accordion title="Typography & Design">
    <Columns cols={2}>
      <Frame caption="Data visualization with clean typography">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/2355f71eb41d1da454ef3c1b820b3d7ce644bd16-1920x1920.jpg" alt="Freiburg infographic with clean typography" />
      </Frame>

      <Frame caption="Ad creative with embedded text">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/216bd6da356153b3b3c9d3cce90225fab86fdc43-1440x960.jpg" alt="Ad creative with embedded text" />
      </Frame>
    </Columns>

    <Columns cols={2}>
      <Frame caption="Magazine cover layout">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/00ddd4ce8b582891f3b174462dc635dac4e45d46-1456x1920.jpg" alt="Magazine cover layout" />
      </Frame>

      <Frame caption="Automotive ad with headline">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/dd06b0f7b82ba5776e09d6827605733dcb5a7526-1456x1920.jpg" alt="Automotive advertisement with headline" />
      </Frame>
    </Columns>
  </Accordion>

  <Accordion title="Grounding Search">
    FLUX.2 \[max] searches the web when prompted, so you can generate visuals grounded in real-time information — sports scores, weather, or historical events.

    <Columns cols={2}>
      <Frame caption="Score of a previous football game">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/5d9c090250ee74dcd77a9b600bafeb6e53c0f692-1680x1680.jpg" alt="Football game score generated from real-time data" />
      </Frame>

      <Frame caption="The weather in real-time of Freiburg">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/d7b0dd9dfc0236ad763cda9de4b17dea9f455b90-1936x1952.jpg" alt="Real-time weather visualization for Freiburg" />
      </Frame>
    </Columns>

    <Columns cols={2}>
      <Frame caption="Re-create historical events: 'GC4Q+2V Berlin, Nov. 9th 1989'">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/23784cfd7989338315503484278d66ede410de02-2032x1808.jpg" alt="Historical event recreation — Fall of the Berlin Wall" />
      </Frame>

      <Frame caption="Next Starlink satellite launch">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/13aa8ee65ef34a6a6b43f9417ff0de9dfda7b503-2032x1968.jpg" alt="Starlink satellite launch visualization" />
      </Frame>
    </Columns>
  </Accordion>

  <Accordion title="Exact Color Control">
    Specify brand colors via hex codes with precision matching — no approximation.

    <Frame caption="Hex colors applied to vase and flowers">
      <img src="https://cdn.sanity.io/images/2gpum2i6/production/3a3bf9b588602adf581f7293611b0e59fd50eadb-1552x1520.png" alt="Vase with gradient hex colors and pink flowers" />
    </Frame>

    <Prompt description="A vase on a table in living room, the color of the vase is a gradient of color, starting with color #02eb3c and finishing with color #edfa3c. The flowers inside the vase have the color #ff0088" actions={[]} />

    <Frame caption="Brand color matching">
      <img src="https://cdn.sanity.io/images/2gpum2i6/production/0dc0c7638df1869005b1a502211e5f0bf967dfdc-1024x768.jpg" alt="Eyeshadow palette with precise hex colors" />
    </Frame>

    <Prompt description="Luxury eyeshadow palette with 6 pans: top row #B76E79, #E8D5B7, #8B4789; bottom row #CD7F32, #F8F6F0, #800020" actions={[]} />
  </Accordion>

  <Accordion title="Structured Prompting">
    Use JSON-structured prompts for precise control over generation — ideal for production workflows and automation.

    ```json Example: Structured Prompting theme={null}
    {
      "subject": "Mona Lisa painting by Leonardo da Vinci",
      "background": "museum gallery wall, ornate gold frame",
      "lighting": "soft gallery lighting, warm spotlights",
      "style": "digital art, high contrast",
      "camera_angle": "eye level view",
      "composition": "centered, portrait orientation"
    }
    ```

    <Columns cols={2}>
      <Frame caption="Eye Level View">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/bc12fcb7269c9449f2cbc3b1d1f54c59da4850e3-1456x1424.jpg" alt="Mona Lisa in museum — eye level view" />
      </Frame>

      <Frame caption="Worm's Eye View">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/8c3fa60821f9e0f30de348f7827547cb82b564c9-1456x1424.jpg" alt="Mona Lisa in museum — worm's eye view" />
      </Frame>
    </Columns>
  </Accordion>

  <Accordion title="Creative & Culture">
    <Columns cols={2}>
      <Frame caption="Consistent comic book style across panels">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/ff4d90d49054184073fcf25b86ac9bcb96f0eb41-1440x832.jpg" alt="Comic book panel 1" />
      </Frame>

      <Frame caption="Consistent comic book style across panels">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/7b767ce5259d743e8e98e408e47f9c75fb285882-1440x832.jpg" alt="Comic book panel 2" />
      </Frame>
    </Columns>

    <Columns cols={2}>
      <Frame caption="Consistent comic book style across panels">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/c146397a89b81fad8af3c246132bcf6163f68af3-1328x800.jpg" alt="Comic book panel 3" />
      </Frame>

      <Frame caption="Consistent comic book style across panels">
        <img src="https://cdn.sanity.io/images/2gpum2i6/production/2630c974542af56eb8d894f394c340fdf2fd2004-1440x832.jpg" alt="Comic book panel 4" />
      </Frame>
    </Columns>
  </Accordion>
</AccordionGroup>

## API Integration

### Create Request

<CodeGroup>
  ```bash create_request.sh theme={null}
  request=$(curl -X POST \
    'https://api.bfl.ai/v1/flux-2-pro-preview' \
    -H 'accept: application/json' \
    -H "x-key: ${BFL_API_KEY}" \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt": "A serene mountain landscape at golden hour, soft diffused light filtering through clouds",
      "width": 1024,
      "height": 1024
    }')
  echo $request
  request_id=$(echo $request | jq -r .id)
  polling_url=$(echo $request | jq -r .polling_url)
  ```

  ```python create_request.py theme={null}
  import os
  import requests

  response = requests.post(
      'https://api.bfl.ai/v1/flux-2-pro-preview',
      headers={
          'accept': 'application/json',
          'x-key': os.environ.get("BFL_API_KEY"),
          'Content-Type': 'application/json',
      },
      json={
          'prompt': 'A serene mountain landscape at golden hour, soft diffused light filtering through clouds',
          'width': 1024,
          'height': 1024,
      },
  ).json()

  request_id = response["id"]
  polling_url = response["polling_url"]
  ```
</CodeGroup>

### Poll for Result

<CodeGroup>
  ```bash poll_result.sh theme={null}
  while true; do
    sleep 0.5
    result=$(curl -s -X 'GET' \
      "${polling_url}" \
      -H 'accept: application/json' \
      -H "x-key: ${BFL_API_KEY}")

    status=$(echo $result | jq -r .status)
    echo "Status: $status"

    if [ "$status" == "Ready" ]; then
      echo "Result: $(echo $result | jq -r .result.sample)"
      break
    elif [ "$status" == "Error" ] || [ "$status" == "Failed" ]; then
      echo "Generation failed: $result"
      break
    fi
  done
  ```

  ```python poll_result.py theme={null}
  import time
  import os
  import requests

  while True:
    time.sleep(0.5)
    result = requests.get(
        polling_url,
        headers={
            'accept': 'application/json',
            'x-key': os.environ.get("BFL_API_KEY"),
        },
        params={'id': request_id}
    ).json()

    if result['status'] == 'Ready':
        print(f"Image ready: {result['result']['sample']}")
        break
    elif result['status'] in ['Error', 'Failed']:
        print(f"Generation failed: {result}")
        break
  ```
</CodeGroup>

<Warning>
  Signed URLs are only valid for 10 minutes. Please retrieve your result within this timeframe.
</Warning>

For all available parameters, endpoints, and model options, see the [API Reference](https://docs.bfl.ml/api-reference).
