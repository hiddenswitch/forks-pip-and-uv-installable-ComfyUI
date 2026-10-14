# Ideogram 4.0 JSON prompting (Ideogram docs)

Vendored snapshot from: https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/4.-json-prompting-ideogram-4.0.md

Fetched: 2026-10-02

---

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/4.-json-prompting-ideogram-4.0.md).

# 4. JSON Prompting (Ideogram 4.0)

## Why JSON prompting?

Ideogram 4.0 was trained **exclusively on structured JSON captions**. Natural language prompts still work well — but providing a JSON object that follows the caption schema gives you capabilities that go far beyond what natural language can express:

* **Exact color control**: Specify up to 16 hex colors for the whole image, and up to 5 per individual element.
* **Bounding-box layout**: Place any element precisely within the frame using normalized coordinates.
* **Precise text placement**: Define each piece of in-image text as its own element with its own position and styling.
* **Consistent, repeatable outputs**: Two runs from the same JSON prompt stay close to each other in composition — useful for design workflows.

{% hint style="info" %}
You don't have to write JSON yourself. **Magic Prompt** can automatically expand a plain-text description into a full structured JSON caption, giving you JSON-quality results from a casual prompt. See [Using Magic Prompt](https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/7-creative-tools-in-ideogram/using-magic-prompt.md) for more.
{% endhint %}

**This section will be doing a deeper dive on JSON formatting and variables.** We know that this can be a bit complex for first-time users, but we'll do our best to make the details as clear as possible. We're also working on product updates that make JSON prompting more visual and easier for first-time users! Our goal is for everyone to use the power of Ideogram 4.0 and JSON prompting — whether you have previous technical experience or not :sunglasses:.

***

## When to use JSON vs. natural language

<table><thead><tr><th>Use case</th><th width="234.2109375">Recommended approach</th></tr></thead><tbody><tr><td>Quick exploration, creative ideas</td><td>Natural language</td></tr><tr><td>Expanding a short idea</td><td>Natural language + Magic Prompt</td></tr><tr><td>Design work, posters, branded graphics</td><td>JSON</td></tr><tr><td>Exact text placement (signage, logos, labels)</td><td>JSON</td></tr><tr><td>Specific color palette</td><td>JSON</td></tr><tr><td>Repeatable layouts across multiple generations</td><td>JSON</td></tr><tr><td>Photorealistic scenes where placement isn't critical</td><td>Natural language</td></tr></tbody></table>

***

## The JSON caption schema

A valid Ideogram 4.0 JSON caption has three top-level fields:

| Field                          | Required             | Description                                                |
| ------------------------------ | -------------------- | ---------------------------------------------------------- |
| `high_level_description`       | Strongly recommended | One or two sentences summarizing the full image            |
| `style_description`            | Optional             | Controls visual style, lighting, medium, and color palette |
| `compositional_deconstruction` | **Required**         | Defines the background and all individual elements         |

***

## `high_level_description`

A one- or two-sentence summary of the entire image. Think of it like the **Image summary** from Section 3.1 — if you could only say one thing, this is it.

`"high_level_description": "A medium-shot photograph of a barista pouring latte art in a cozy cafe."`

***

## `style_description`

Controls the visual style, lighting, medium, and color palette. It must contain **exactly one** of:

* `photo` — for photographic outputs
* `art_style` — for illustrations, paintings, 3D renders, graphic design, etc.

The fields `aesthetics`, `lighting`, and `medium` are also required when `style_description` is present. `color_palette` is optional.

**Key order matters** and depends on which style type you use:

| Caption type | Required key order                                               |
| ------------ | ---------------------------------------------------------------- |
| Photo        | `aesthetics`, `lighting`, `photo`, `medium`, `color_palette`     |
| Non-photo    | `aesthetics`, `lighting`, `medium`, `art_style`, `color_palette` |

| Field           | Type            | Description                                                                                                         |
| --------------- | --------------- | ------------------------------------------------------------------------------------------------------------------- |
| `aesthetics`    | string          | Aesthetic keywords (e.g. `"moody, cinematic, desaturated"`)                                                         |
| `lighting`      | string          | Lighting description (e.g. `"golden hour, rim light, dramatic shadows"`)                                            |
| `photo`         | string          | Camera/lens details for photos (e.g. `"35mm, f/1.4, bokeh"`). Use this **or** `art_style`, not both.                |
| `medium`        | string          | Medium type: `"photograph"`, `"illustration"`, `"3d_render"`, `"painting"`, `"graphic_design"`, etc.                |
| `art_style`     | string          | Art style for non-photo work (e.g. `"flat vector illustration, bold outlines"`). Use this **or** `photo`, not both. |
| `color_palette` | list of strings | Hex color codes that steer the image's dominant colors. Up to 16 entries. **Must be uppercase** `#RRGGBB`.          |

***

## `compositional_deconstruction`

This is where you describe the background and every individual element in the scene. Both `background` and `elements` are required, and `background` must come first.

| Field        | Type   | Description                                                          |
| ------------ | ------ | -------------------------------------------------------------------- |
| `background` | string | Description of the background/environment                            |
| `elements`   | list   | List of objects and text elements, each with optional bounding boxes |

***

## **Element types**

Each element in `elements` is either an object (`"obj"`) or a text element (`"text"`).

| Type     | Required key order                                                        |
| -------- | ------------------------------------------------------------------------- |
| `"obj"`  | `type`, `bbox` *(optional)*, `desc`, `color_palette` *(optional)*         |
| `"text"` | `type`, `bbox` *(optional)*, `text`, `desc`, `color_palette` *(optional)* |

| Field           | Type               | Description                                                                                                                          |
| --------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `type`          | string             | `"obj"` for objects/subjects, `"text"` for in-image text                                                                             |
| `bbox`          | list of 4 integers | `[y_min, x_min, y_max, x_max]` in normalized 0–1000 coordinates (origin at top-left). Optional — omit to let the model place freely. |
| `desc`          | string             | Detailed description of the element                                                                                                  |
| `text`          | string             | *(text elements only)* The literal text to render in the image                                                                       |
| `color_palette` | list of strings    | Per-element color palette. Up to 5 uppercase `#RRGGBB` hex entries. Optional.                                                        |

{% hint style="info" %}
**Important:** Key order matters. The model was trained on captions with a consistent key order, so maintaining it improves generation quality.
{% endhint %}

***

## Bounding boxes

Bounding boxes let you specify exactly where an element appears in the frame. Coordinates use a normalized 0–1000 system where `[0, 0]` is the top-left corner of the image.

```
[y_min, x_min, y_max, x_max]
```

For example, `[0, 0, 500, 1000]` places an element in the top half of the image. `[250, 250, 750, 750]` places it roughly centered.

Bounding boxes are optional for every element — if you omit them, the model will place the element freely based on its description.

***

## Color palette tips

* Include **background colors** in the palette if you want to control the overall tone
* Include **contrast pairs** — both highlight and shadow colors — for more controlled lighting
* Use **uppercase hex only**: `#RRGGBB`, not `#rgb` or lowercase

***

## A simple JSON example (no bounding boxes)

A clean starting point — great for anyone new to JSON prompting:

**Prompt:**

```json
{
   "high_level_description":"A lone sailboat on calm water at golden hour.",
   "style_description":{
      "aesthetics":"serene, warm, golden hour",
      "lighting":"golden hour backlighting, warm atmospheric haze",
      "photo":"wide angle, f/8, long exposure",
      "medium":"photograph",
      "color_palette":[
         "#FF6B35",
         "#F7C59F",
         "#004E89",
         "#1A659E",
         "#2B2D42"
      ]
   },
   "compositional_deconstruction":{
      "background":"A calm ocean stretching to a low horizon, sky washed in orange and pink with thin wisps of cloud.",
      "elements":[
         {
            "type":"obj",
            "desc":"A single sailboat with a white triangular sail, silhouetted against the setting sun."
         }
      ]
   }
}
```

**Output:**

• [Magic Prompt](https://docs.ideogram.ai/using-ideogram/generation-settings/magic-prompt.md): OFF (automatically turned off when using JSON format for Ideogram 4.0)\
• [Aspect ratio](https://docs.ideogram.ai/using-ideogram/generation-settings/aspect-ratio-and-dimensions.md): 1:2\
• [Model](https://docs.ideogram.ai/using-ideogram/generation-settings/available-models.md): 4.0

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FySvsEMEF1FI10iOm4qsG%2Fhigh-level-description-a-lone-sailboat-o_LWy40HW7XgyaVD4dEwWhOg_iesvurn3SqO0mvA8LYomEg_cover%20(1).png?alt=media&amp;token=596cfb4c-07c5-49d0-a425-1e5168618ca2" alt="Photograph of a lone sailboat silhouetted against a golden sunset over a calm ocean, the sun low on the horizon" width="375"><figcaption></figcaption></figure>

***

## A JSON example with Bounding Boxes

When exact placement matters — for design work, posters, or complex compositions:

```json
{
  "high_level_description": "A bold event poster for a jazz night called 'Blue Note Sessions' at The Velvet Room, Saturday August 9th.",
  "style_description": {
    "aesthetics": "moody, retro, sophisticated, 1960s jazz club aesthetic",
    "lighting": "dramatic, deep shadows, warm spotlight glow",
    "medium": "graphic_design",
    "art_style": "vintage poster design, textured paper, bold typography, muted color palette with warm accents",
    "color_palette": ["#1A1A2E", "#16213E", "#E8C97A", "#D4A843", "#F5F0E8", "#8B4513"]
  },
  "compositional_deconstruction": {
    "background": "Deep navy and near-black background with subtle aged paper texture and faint horizontal grain lines.",
    "elements": [
      {
        "type": "obj",
        "bbox": [150, 200, 650, 800],
        "desc": "A silhouetted jazz trumpeter in side profile, mid-performance, instrument raised. Warm golden spotlight illuminates from above, casting dramatic shadows. Stylized, slightly abstract illustration style."
      },
      {
        "type": "text",
        "bbox": [30, 50, 140, 950],
        "text": "BLUE NOTE SESSIONS",
        "desc": "Large bold all-caps serif headline in warm golden-yellow, spanning the full width near the top of the poster."
      },
      {
        "type": "text",
        "bbox": [660, 100, 760, 900],
        "text": "Live jazz every Saturday night",
        "desc": "Medium-weight italic serif subheading in off-white, centered beneath the main title."
      },
      {
        "type": "text",
        "bbox": [820, 200, 900, 800],
        "text": "THE VELVET ROOM",
        "desc": "Smaller all-caps sans-serif venue name in warm gold, centered near the bottom."
      },
      {
        "type": "text",
        "bbox": [900, 300, 970, 700],
        "text": "SAT · AUGUST 9",
        "desc": "Small light-weight serif date text in off-white, near the bottom of the poster."
      }
    ]
  }
}
```

**Output:**

• [Magic Prompt](https://docs.ideogram.ai/using-ideogram/generation-settings/magic-prompt.md): OFF (automatically turned off when using JSON format for Ideogram 4.0)\
• [Aspect ratio](https://docs.ideogram.ai/using-ideogram/generation-settings/aspect-ratio-and-dimensions.md): 1:2\
• [Model](https://docs.ideogram.ai/using-ideogram/generation-settings/available-models.md): 4.0

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FKYqvYPmYcmwUkZYfQaed%2Fimage.png?alt=media&amp;token=f636b92e-5b5e-401d-945b-4732b7fe8b2f" alt="Vintage jazz poster of a spotlit trumpeter reading Blue Note Sessions, Live jazz every Saturday night, The Velvet Room, Sat August 9" width="375"><figcaption></figcaption></figure>

***

## Using Magic Prompt to generate JSON

Don't want to write JSON by hand? Magic Prompt handles the conversion for you. It uses a language model to expand your plain-text description into a full structured JSON caption automatically, giving you the quality benefits of JSON without writing the structure yourself.

* **On Ideogram.ai**: Magic Prompt is enabled by default. Turn it on before generating, and your plain-text prompt will be expanded into a JSON caption.
* **Via the API**: Magic Prompt can be run server-side for free with an Ideogram API key.

{% hint style="info" %}
For casual exploration, Magic Prompt is the recommended starting point. For production workflows where you need precise, repeatable control over layout, color, and typography, write the JSON directly.
{% endhint %}

***

## JSON vs. Natural language: A side-by-side comparison

The two examples below describe the same subject. Notice how the JSON version provides explicit control over composition and style that the plain-text prompt leaves to interpretation.

**Plain text:**

> *A vintage 1950s collectible natural-history trading card showing a goldcrest bird (Regulus regulus) perched on a pine sprig. The card has a small halftone illustration centered on cream cardstock, with the common name "GOLDCREST" in bold black serif capitals across the top, the Latin name "Regulus regulus" in italic beneath the bird, a small caption noting it is the smallest of the British songbirds at around 9 centimetres, card number "17 OF 50" in the lower right, and the series mark "BRITISH SONGBIRDS" along the bottom in dark green ink. Slight foxing on the paper edges, fine halftone print texture, soft even lighting.*

**JSON excerpt:**

The upstream snapshot includes only the tail of this example; use one of the complete JSON examples above as a template.

```text
            "type":"text",
            "bbox":[
               775,
               150,
               860,
               850
            ],
            "text":"The smallest of the British songbirds, measuring around 9 centimetres.",
            "desc":"Small upright serif body text in dark ink, centered beneath the Latin name."
         },
         {
            "type":"text",
            "bbox":[
               868,
               680,
               930,
               940
            ],
            "text":"17 OF 50",
            "desc":"Small serif card number in dark ink, aligned to the bottom-right corner inside the border."
         },
         {
            "type":"text",
            "bbox":[
               930,
               50,
               990,
               950
            ],
            "text":"BRITISH SONGBIRDS",
            "desc":"Bold spaced green serif capitals in a solid footer band spanning the full width at the bottom of the card."
         }
      ]
   }
}
```

{% tabs %}
{% tab title="Natural language" %}

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FV1H4fCD8TFWG7zOhm8KA%2Fimage.png?alt=media&amp;token=101371df-35e4-42fc-9086-9c5563293b76" alt="Vintage trading card of a goldcrest on a pine sprig with the text Goldcrest, Regulus regulus, 17 of 50 and British Songbirds"><figcaption></figcaption></figure>
{% endtab %}

{% tab title="JSON prompt" %}

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2F1QfKVyUlvJDcA4Dwdlvh%2Fimage.png?alt=media&amp;token=ae773d67-24f6-4bd7-81c7-be5957015d70" alt="JSON-prompted goldcrest card with the same wording, each text element placed in its own fixed region of the layout"><figcaption></figcaption></figure>
{% endtab %}
{% endtabs %}

The plain-text version produced a strong result — but notice what the JSON version guarantees that the plain-text version doesn't: every element landed exactly where it was told to. The title sits in its own zone at the top, the bird has bounded space that no text can bleed into, the Latin name, caption, card number, and footer each occupy a dedicated region. Run the plain-text prompt ten times and the layout will drift. Run the JSON prompt ten times and the structure holds. That's the practical case for JSON — not quality, but **repeatability and compositional control**.
