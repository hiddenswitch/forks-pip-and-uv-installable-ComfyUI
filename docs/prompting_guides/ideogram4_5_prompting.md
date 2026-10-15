# Ideogram 4.5 prompting guide (Ideogram docs)

Vendored snapshot of the Prompting section of https://docs.ideogram.ai (index: https://docs.ideogram.ai/llms.txt): Prompting Basics, Prompt Structure, Text in Images, JSON Prompting, Refine and Iterate, Fix Common Problems and the Vocabulary Reference with its five subpages. The pages are concatenated verbatim in index order from their `.md` endpoints; each page's source URL precedes it.

Fetched: 2026-10-05

Ideogram publishes one guide for Ideogram 4.5, 4.0 and 3.0; there is no separate 4.5 guide. Ideogram 4.5 is API-only at this snapshot (see `ideogram4_5_api.md` and `ideogram4_5_precise_edit_api.md`).

---

<!-- SOURCE: https://docs.ideogram.ai/prompting/prompting.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/prompting.md).

# Prompting Basics

How Ideogram reads a prompt, when to use plain language or JSON, and what Magic Prompt does.

A prompt is the text you give Ideogram to describe the image you want. It can be one line or a full paragraph, plain or poetic. This guide covers the Ideogram models in the Image app: Ideogram 4.5, Ideogram 4.0 and Ideogram 3.0.

## How Ideogram reads a prompt

Ideogram reads your words literally and tries to show every one of them. It does not guess at what you meant the way a person would.

> A tall man in a red coat walking through a snowy forest.

From this prompt Ideogram will try to draw the height, the red coat, the walking, the snow and the forest. Leave out a detail and the model picks one for you. Add a word you don't need and it may show up in the image.

A few habits follow from this:

* **Write in sentences.** Tag lists like `man, forest, fire, dramatic, painting` can work, but full sentences tell the model how things relate: who holds what, what sits behind what.
* **Describe what can be seen.** "A sunset over the ocean with orange and pink light on the water" gives the model more to draw than "a beautiful scene." [Prompt Structure](https://docs.ideogram.ai/prompting/prompt-structure.md) covers this in depth.
* **Put the main idea first.** Words near the start of the prompt tend to carry more weight.
* **Say what you want, not what you don't.** "An empty street" works better than "a street with no people."

There is no hidden syntax. Ideogram doesn't read weights like `::1` or `(important)` or flags like `--ar` and `--style` as commands; it treats them as words. Set aspect ratio, seed and quality in the Image app settings instead.

You can write in any language. English gives the most reliable results, most of all when the image includes text. Magic Prompt can translate your prompt into English for you.

## Plain language or JSON

Ideogram 4.5 and 4.0 accept either a plain-language prompt or a structured JSON prompt. Plain language is the place to start, and it is all Ideogram 3.0 accepts. JSON lets you set exact layout, text placement and hex colors when a sentence can't pin them down. See [JSON Prompting](https://docs.ideogram.ai/prompting/json-prompting.md).

## Magic Prompt

Magic Prompt rewrites and expands a short prompt before Ideogram generates the image. On Ideogram 4.5 and 4.0 it turns your plain text into a structured JSON prompt, so you get much of the control of JSON without writing it. Turn it off when you want Ideogram to keep your exact wording. On every Ideogram model it's an on/off toggle. See the [Magic Prompt setting](https://docs.ideogram.ai/create/image-settings.md#magic-prompt) for where to find it.

## Start here

1. Pick a model: Ideogram 4.5, 4.0 or 3.0. See [available models](https://docs.ideogram.ai/create/models.md).
2. Write one sentence that says what kind of image it is and what it shows, such as `A watercolor painting of a fox curled up in snow under a pine tree.`
3. Put any text you want in the image in quotation marks, near the start.
4. Generate, look at the results, then add detail only where the image missed.
5. Change one thing at a time so you can tell what made the difference.

## Next pages

1. [Prompt Structure](https://docs.ideogram.ai/prompting/prompt-structure.md)
2. [Text in Images](https://docs.ideogram.ai/prompting/text-in-images.md)
3. [JSON Prompting](https://docs.ideogram.ai/prompting/json-prompting.md)
4. [Refine and Iterate](https://docs.ideogram.ai/prompting/refine-and-iterate.md)
5. [Fix Common Problems](https://docs.ideogram.ai/prompting/fix-common-problems.md)
6. [Vocabulary Reference](https://docs.ideogram.ai/prompting/vocabulary.md)


<!-- SOURCE: https://docs.ideogram.ai/prompting/prompt-structure.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/prompt-structure.md).

# Prompt Structure

The parts of a plain-language prompt, how to put them in order, how long to make it, and when to write literally or loosely.

Ideogram has no required format, but prompts that follow a steady order get better results. The model can tell the subject from the background, and you can see which part to change when an image misses.

## The parts of a prompt

Each part below shows two running examples: a perfume product photo and a watercolor painting. Use only the parts you need. A logo may have no secondary elements, and a quick idea may need nothing past the summary.

### 1. Image summary

One sentence that describes the whole image, as if someone glanced at it for two seconds. Name the kind of image (photo, logo, painting), the main subject and the tone. If you write only one sentence, make it this one. It is also the best input for [Magic Prompt](https://docs.ideogram.ai/create/image-settings.md#magic-prompt) to expand.

* `A product photo of a men's perfume bottle named "Nightlife for men" in a sleek studio setup.`
* `A whimsical watercolor painting of a little girl playing with her bunny in a flower-filled field.`

### 2. Main subject details

What the subject looks like: color, shape, material, texture. Put any text you want in the image here, in quotation marks. [Text in Images](https://docs.ideogram.ai/prompting/text-in-images.md) covers this in full.

* `The bottle is tall and rectangular with dark glass, a matte black cap, and silver lettering. The text "Nightlife for men" appears on the label in bold, modern font.`
* `The girl has short brown hair, a yellow dress, and rosy cheeks. She holds a fluffy white bunny in her arms, and both are smiling.`

### 3. Pose or action

What the subject is doing, or how it sits in the scene. Even a still object has a pose.

* `The bottle stands upright with a slight reflection on the surface below.`
* `The bunny is leaning into her, with its ears flopping gently.`

### 4. Secondary elements

Props and small details near the subject that fill out the scene without taking over.

* `A wristwatch and a pair of sunglasses sit nearby.`
* `Wildflowers, butterflies, and a toy picnic basket surround them.`

### 5. Setting and background

Where the image takes place: indoors or out, time of day, surroundings. For a logo or product shot this may be a flat color.

* `The scene is set on a smooth black surface with blurred city lights in the background.`
* `They are outdoors in a grassy meadow, under a wide blue sky.`

### 6. Lighting and atmosphere

How the light falls and how the image feels. This part sets the mood.

* `Lighting is moody and cool, with soft blue highlights and deep shadows.`
* `The light is soft and sunny, with warm pastel tones and a dreamy atmosphere.`

### 7. Framing and composition

Camera angle, shot type and where the subject sits in the frame. This applies to paintings and logos too: how a figure is balanced, how shapes are spaced.

* `The bottle is centered in the frame, captured at eye level.`
* `The girl and bunny are slightly off-center, framed from a gentle downward angle.`

### 8. Finish

Details that change how the image is rendered, not what it shows: lens effects, brush texture, line quality.

* `A shallow depth of field gives the image a polished, professional look.`
* `The brush strokes are loose and textured, with light color bleeds.`

## Putting it together

Join the parts in this order:

```
[Image summary]. [Main subject details]. [Pose or action]. [Secondary elements]. [Setting and background]. [Lighting and atmosphere]. [Framing and composition]. [Finish].
```

The two examples above become:

> A product photo of a men's perfume bottle named "Nightlife for men" in a sleek studio setup. The bottle is tall and rectangular with dark glass, a matte black cap, and silver lettering. The text "Nightlife for men" appears on the label in bold, modern font. The bottle stands upright with a slight reflection on the surface below. A wristwatch and a pair of sunglasses sit nearby. The scene is set on a smooth black surface with blurred city lights in the background. Lighting is moody and cool, with soft blue highlights and deep shadows. The bottle is centered in the frame, captured at eye level. A shallow depth of field gives the image a polished, professional look.

> A whimsical watercolor painting of a little girl playing with her bunny in a flower-filled field. The girl has short brown hair, a yellow dress, and rosy cheeks. She holds a fluffy white bunny in her arms, and both are smiling. The bunny is leaning into her, with its ears flopping gently. Wildflowers, butterflies, and a toy picnic basket surround them. They are outdoors in a grassy meadow, under a wide blue sky. The light is soft and sunny, with warm pastel tones and a dreamy atmosphere. The girl and bunny are slightly off-center, framed from a gentle downward angle. The brush strokes are loose and textured, with light color bleeds.

## How long to make it

Length decides how much Ideogram fills in for you. Neither length is better; they serve different goals.

**Short prompts** leave room for surprise. Use them to explore, or let Magic Prompt add the detail.

> A woman drifting through a quiet dream, shapes and colors shifting gently around her.

**Longer prompts** give you control over subject, background, light and style. Use them when you already know what you want.

> A red fox standing beneath bright autumn trees, surrounded by golden leaves falling onto a quiet forest floor.

When you write a long prompt, follow the order above and keep the main idea near the start. Too many unrelated ideas in one prompt confuse the model.

{% hint style="warning" %}
On Ideogram 3.0, keep plain-language prompts under about 150 words (roughly 200 tokens). It may weaken or ignore words past that point. Ideogram 4.5 takes up to 10,000 characters, including full JSON prompts, but a focused prompt is still easier to control.
{% endhint %}

## Literal, loose or both

How you phrase a prompt changes how closely the image follows it. You can write a short or long prompt in any of these styles.

### Visually grounded

Describe only what can be seen: subject, setting, light, color, layout. These prompts give the most consistent results and the best control over layout, which matters most when the image includes text.

> A stone lighthouse standing on a rocky cliff in heavy fog, its beam shining across calm gray waves at dusk.

> A woman sitting in a dark room beside a table, reading a book under the warm glow of a single candle.

### Abstract

Use feeling, symbol and figurative language. Results are harder to predict, which suits exploration. Ideogram handles these well when you include a few concrete anchors, like the lighthouse and waves below.

> A lighthouse fading into the mist, its light barely reaching the waves as the sea swallows the horizon.

> A woman reading by candlelight, her face half-shadowed, as time seems to pause around her.

### Hybrid

Fix the subject and style in plain terms, then let one phrase open up the mood or background. This works when you need a set subject but want the model to vary the atmosphere.

> A watercolor of a ballerina mid-spin on a quiet stage, like a moment of silence captured in motion.

> A portrait of a young woman in a red cloak, standing still. The forest behind her blurs into colors like wet paint on glass.

Once a prompt is close, see [Refine and Iterate](https://docs.ideogram.ai/prompting/refine-and-iterate.md) for how to adjust it.


<!-- SOURCE: https://docs.ideogram.ai/prompting/text-in-images.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/text-in-images.md).

# Text in Images

How to get clean, correctly spelled text in your images with Ideogram 4.5 and 4.0.

Ideogram can put text almost anywhere in an image: a caption over a photo, a headline on a poster, a brand name on a bottle. Put the exact words in quotation marks and describe where they sit.

**Prompt:**

> A risograph poster pasted on a sunlit brick wall with text that reads: "Everything you can imagine is real. – Pablo Picasso"

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-0321bdfa2251c5b49636b54e6674af626a76402f%2Ftext-picasso-poster.png?alt=media" alt="Cream risograph poster with a pink ink band pasted on a sunlit brick wall, reading Everything you can imagine is real – Pablo Picasso in bold navy letters" width="375"><figcaption><p>The prompt named the text and where it goes. The model chose the layout and lettering.</p></figcaption></figure>

## Write the text into the prompt

* **Quote the exact words.** Anything inside quotation marks is text to render. Keep spelling and capitals as you want them to appear.
* **Mention the text early.** Text named near the start of the prompt comes out more reliably than text added at the end.
* **Say where it appears.** "Painted across the mural behind the artist" or "curving along the bottom" gives the model a place to put the words.
* **Keep it short.** Each extra word raises the odds of a typo or a dropped letter. Short phrases work best.
* **Keep the scene simple.** A busy scene leaves less room for clean lettering.

> On the wall behind the artist, the phrase "Inspire Daily" is painted in large brush strokes across a mural, at the back of a vibrant and creative studio.

> A vintage poster design with the words "Ride Free" curving along the bottom in retro lettering, featuring a smiling woman on a bicycle on a countryside road.

Text can be part of an object, or the object can form the text:

<table data-card-size="large" data-view="cards"><thead><tr><th></th><th></th><th></th></tr></thead><tbody><tr><td><strong>Text on an object</strong></td><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2F6p6jS2kTbBoz7pAY9mXx%2Fimage.png?alt=media&amp;token=a10ce708-ea64-415b-85ee-bc375840c849" alt="Close-up of a red baseball cap with an embroidered snarling red panda emblem above the team name Red Panda in white script" data-size="original"></td><td><em>A close-up detailed product photo of a vibrant and energetic baseball cap, featuring a stylish red panda embroidered emblem that exudes vitality. The cap is embroidered with the team's name, "Red Panda," displayed boldly in white bold script font below the sleek red panda design.</em></td></tr><tr><td><strong>Text formed by objects</strong></td><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FGhPsG86LABMnmApvuUVp%2Fimage.png?alt=media&amp;token=7ae55472-2dd4-49ff-ab55-40dfddf455d0" alt="Milk splashing out of a glass and forming the word Milk in swirly liquid lettering above a wooden kitchen table" data-size="original"></td><td><em>A striking photograph in which a glass of milk is placed on a wooden table. A splash of milk comes out of the glass, forming the word "Milk" in a swirly manner just above the glass. The background is blurred but looks like a homely kitchen.</em></td></tr></tbody></table>

## Describe the lettering

You can't pick a font by name. Ideogram doesn't load real fonts; it learned what text styles look like from images. Describe the weight, shape and mood of the letters instead.

<table data-card-size="large" data-view="cards"><thead><tr><th></th><th></th><th></th></tr></thead><tbody><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-b8bc5c04b873f180e065faeb23152bef0849d20c%2Ftext-type-bold.png?alt=media" alt="The word Welcome in a heavy bold sans-serif typeface, forest green on cream" data-size="original"></td><td><em>The word "Welcome" written in <mark style="color:green;">bold sans-serif</mark> typeface, in deep forest green on a warm cream background.</em></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-f85dbe002cac0729ffb626bbc15a204d005850d3%2Ftext-type-thin.png?alt=media" alt="The word Welcome in an ultra thin sans-serif typeface, forest green on cream" data-size="original"></td><td><em>The word "Welcome" written in <mark style="color:green;">ultra thin sans-serif</mark> typeface, in deep forest green on a warm cream background.</em></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-505688e5d7933716464244976bbd9e6e93b64293%2Ftext-type-serif.png?alt=media" alt="The word Welcome in a classic serif typeface, forest green on cream" data-size="original"></td><td><em>The word "Welcome" written in <mark style="color:green;">serif</mark> typeface, in deep forest green on a warm cream background.</em></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-78e88c535134bd515a3bebe0a1ece3aa821f6c47%2Ftext-type-bauhaus.png?alt=media" alt="The word Welcome in thin rounded Bauhaus-style letters, forest green on cream" data-size="original"></td><td><em>The word "Welcome" written in <mark style="color:green;">thin rounded bauhaus style</mark> typeface, in deep forest green on a warm cream background.</em></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-29a25677f724f9e3c8d6d0808902735e9333a267%2Ftext-type-script.png?alt=media" alt="The word Welcome in a thin formal script with flourishes, forest green on cream" data-size="original"></td><td><em>The word "Welcome" written in <mark style="color:green;">thin and refined formal script</mark> typeface <mark style="color:green;">with flourishes</mark>, in deep forest green on a warm cream background.</em></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-8607fd58e54cc3e3b94b1058ef3984c2c99234d1%2Ftext-type-hippie.png?alt=media" alt="The word Welcome in a bulbous 1960s hippie-style typeface, forest green on cream" data-size="original"></td><td><em>The word "Welcome" written in <mark style="color:green;">an elongated and artistic 1960's hippie-style</mark> typeface, in deep forest green on a warm cream background.</em></td><td></td></tr></tbody></table>

If you leave the style out, the model picks one that suits the scene. [Magic Prompt](https://docs.ideogram.ai/create/image-settings.md#magic-prompt) will also add a fitting style when it expands your prompt.

## More than one block of text

Give each block its own words, style and place. Describe them in reading order, from the top of the image down:

> A vintage jazz poster. At the top, the headline "BLUE NOTE SESSIONS" in large gold serif capitals. Below the trumpeter, "Live jazz every Saturday night" in off-white italic serif. At the bottom, "THE VELVET ROOM" in small gold sans-serif capitals.

Plain text gets you close, but the layout can shift from one run to the next. When each line must land in an exact spot, use a JSON prompt. On Ideogram 4.5 and 4.0, each text element gets its own `text`, `desc` (how it looks), `color_palette` and `bbox`. See [JSON Prompting](https://docs.ideogram.ai/prompting/json-prompting.md) for the format and a full poster example.

## Limits

* **Long passages.** Ideogram handles titles and short blocks well. Add body copy or full documents later in a design tool.
* **Languages.** Text comes out most accurately in English. Non-Latin scripts and accented letters can come out wrong or not at all.

## Fix misspelled text

An image can still come back with a doubled letter or a missing word. Try these in order:

1. Generate again. Each run gives the text another chance.
2. Swap long or unusual words for shorter ones.
3. Fix just the text in [Studio](https://docs.ideogram.ai/edit/studio.md), so the rest of the image stays the same. Download the image, upload it in Studio, then use **AI edit** ("change the sign to read Open Daily") or **Select area**: paint over the wrong word and say what it should read. Select area needs a paid plan.

It's often easier to keep an image whose text is right and fix the scene around it than the other way round. For more on editing a result, see [Refine and Iterate](https://docs.ideogram.ai/prompting/refine-and-iterate.md).

Once you have an image you like, **Layerize text** in Studio turns its text into editable layers you can retype, move or restyle. See [Text and Layers](https://docs.ideogram.ai/edit/text-and-layers.md#layerize-text).


<!-- SOURCE: https://docs.ideogram.ai/prompting/json-prompting.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/json-prompting.md).

# JSON Prompting

Use a structured JSON prompt with Ideogram 4.5 and 4.0 to control exact text placement, layout and colors.

Ideogram 4.5 and 4.0 accept a plain-language prompt or a structured JSON prompt. Both models use the same JSON format. Ideogram 4.0 was trained on captions in this format, so a JSON prompt speaks to the model in its own terms.

Plain language works well for most images. JSON adds control that plain language can't give you:

* **Color control.** Up to 16 hex colors for the whole image and up to 5 for each element. They steer the colors strongly but don't lock every pixel.
* **Layout.** Place any element in the frame with a bounding box.
* **Text placement.** Each piece of text is its own element with its own position and style.
* **Repeatable results.** Runs from the same JSON prompt keep close to the same layout.

## When to use JSON

| Use case                                    | Approach                         |
| ------------------------------------------- | -------------------------------- |
| Quick ideas, exploring                      | Plain language                   |
| Expanding a short idea                      | Plain language with Magic Prompt |
| Posters, branded graphics, labels, signage  | JSON                             |
| A set color palette                         | JSON                             |
| The same layout across many images          | JSON                             |
| Photos where exact placement doesn't matter | Plain language                   |

## How Magic Prompt and JSON work together

You don't have to write JSON by hand. On Ideogram 4.5 and 4.0, [Magic Prompt](https://docs.ideogram.ai/create/image-settings.md#magic-prompt) turns a plain-text prompt into a structured JSON prompt before generating:

* **Magic Prompt on:** it rewrites and expands your idea, adding detail, style and layout, then builds the JSON.
* **Magic Prompt off:** it keeps your wording and only converts it into the structured format.
* **You paste valid JSON:** Ideogram uses your JSON as written and skips Magic Prompt, unless Magic Prompt is On. Then it rewrites your JSON too.

So when you've written JSON by hand and want it used exactly, turn Magic Prompt off.

If your JSON doesn't match the format below (for example, `high_level_description` is missing), Ideogram reads it as plain text. On Ideogram 4.5 the prompt box takes up to 10,000 characters, and very long JSON prompts can lose their last elements, so list the most important elements first.

## The schema

A JSON prompt has three top-level fields:

| Field                          | Required     | What it holds                                    |
| ------------------------------ | ------------ | ------------------------------------------------ |
| `high_level_description`       | **Required** | One or two sentences that sum up the whole image |
| `style_description`            | Optional     | Style, lighting, medium and color palette        |
| `compositional_deconstruction` | **Required** | The background and every element in the image    |

### `high_level_description`

If you could say only one thing about the image, say it here.

```json
{"high_level_description": "A medium-shot photograph of a barista pouring latte art in a cozy cafe."}
```

### `style_description`

Every field is optional. Use `photo` for photographs, or `art_style` for illustration, painting, 3D, graphic design and the like. The model learned from captions that used one or the other, so pick one.

| Field           | Type            | What it holds                                                                                               |
| --------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| `aesthetics`    | string          | Mood and look, for example `"moody, cinematic, desaturated"`                                                |
| `lighting`      | string          | For example `"golden hour, rim light, dramatic shadows"`                                                    |
| `photo`         | string          | Camera and lens, for example `"35mm, f/1.4, bokeh"`. Use this or `art_style`, not both.                     |
| `medium`        | string          | Free text, for example `"photograph"`, `"illustration"`, `"3D render"`, `"painting"`, `"graphic design"`    |
| `art_style`     | string          | For non-photo work, for example `"flat vector illustration, bold outlines"`. Use this or `photo`, not both. |
| `color_palette` | list of strings | Up to 16 hex colors that steer the main colors. Uppercase `#RRGGBB` only.                                   |

### `compositional_deconstruction`

Holds `background` (a string describing the setting) and `elements` (a list). Both are required.

Each element is an object (`"obj"`) or text (`"text"`):

| Type     | Key order                                                                 |
| -------- | ------------------------------------------------------------------------- |
| `"obj"`  | `type`, `bbox` *(optional)*, `desc`, `color_palette` *(optional)*         |
| `"text"` | `type`, `bbox` *(optional)*, `text`, `desc`, `color_palette` *(optional)* |

| Field           | Type               | What it holds                                                    |
| --------------- | ------------------ | ---------------------------------------------------------------- |
| `type`          | string             | `"obj"` for objects and subjects, `"text"` for text in the image |
| `bbox`          | list of 4 integers | Position as `[y_min, x_min, y_max, x_max]`. Optional.            |
| `desc`          | string             | A detailed description of the element                            |
| `text`          | string             | Text elements only: the exact words to render                    |
| `color_palette` | list of strings    | Up to 5 uppercase `#RRGGBB` colors for this element. Optional.   |

{% hint style="info" %}
Ideogram puts the keys inside each block in the order the model expects, so you don't need to. Keep the three top-level fields in the order shown in the schema table.
{% endhint %}

### Bounding boxes

Coordinates run from 0 to 1000 on each axis, with `[0, 0]` at the top-left corner, written as `[y_min, x_min, y_max, x_max]`. So `[0, 0, 500, 1000]` fills the top half of the image and `[250, 250, 750, 750]` sits in the center. Leave `bbox` out and the model places the element itself.

### Color tips

* Add the background colors to the palette if you want to set the overall tone.
* Add both highlight and shadow colors for more control over lighting.
* Write hex codes in uppercase `#RRGGBB`, never `#rgb` or lowercase.

## Example: no bounding boxes

A good first JSON prompt:

```json
{
  "high_level_description": "A lone lighthouse on a sea cliff under a glowing violet aurora at night.",
  "style_description": {
    "aesthetics": "dreamlike, majestic, quiet",
    "lighting": "aurora glow from above, warm beam from the lighthouse, deep blue shadows",
    "photo": "wide-angle, long exposure, crisp stars",
    "medium": "photograph",
    "color_palette": ["#2A1B5C", "#7B4FD6", "#3FD9B0", "#F5C46B", "#0B1026"]
  },
  "compositional_deconstruction": {
    "background": "A night sky filled with swirling violet and teal aurora ribbons and scattered stars above a dark, calm sea.",
    "elements": [
      {
        "type": "obj",
        "desc": "A white lighthouse with a red top standing on the edge of a rugged grassy cliff, its warm beam cutting across the sky."
      },
      {
        "type": "obj",
        "desc": "Gentle waves breaking white against the rocks at the base of the cliff."
      }
    ]
  }
}
```

Settings: Ideogram 4.0, Magic Prompt off, aspect ratio 1:2.

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-3ea16c160bbabbdcc4c673fa73fbabbe642eb86e%2Fjson-lighthouse-aurora.png?alt=media" alt="Photograph of a white lighthouse on a dark grassy sea cliff under swirling violet and teal aurora, its warm light glowing above the calm sea" width="375"><figcaption></figcaption></figure>

## Example: text with bounding boxes

When every line of text needs its own place:

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

Settings: Ideogram 4.0, Magic Prompt off, aspect ratio 1:2.

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FKYqvYPmYcmwUkZYfQaed%2Fimage.png?alt=media&amp;token=f636b92e-5b5e-401d-945b-4732b7fe8b2f" alt="Vintage jazz poster of a spotlit trumpeter reading Blue Note Sessions, Live jazz every Saturday night, The Velvet Room, Sat August 9" width="375"><figcaption></figcaption></figure>

For tips on wording the text itself, see [Text in Images](https://docs.ideogram.ai/prompting/text-in-images.md).


<!-- SOURCE: https://docs.ideogram.ai/prompting/refine-and-iterate.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/refine-and-iterate.md).

# Refine and Iterate

Change one thing at a time, word your prompt precisely, and pick the right tool to move an image closer to what you want.

Your first image is a draft. Small, deliberate changes to the prompt, or a quick edit on the image itself, usually get you closer than starting over.

## Change one thing at a time

Change a single word or phrase per round, so you can see what each change did. Fix the [seed](https://docs.ideogram.ai/create/image-settings.md#seed) and turn Magic Prompt off while you do this: with the same seed and settings, the only thing that moves is your wording.

> **Start:** "A black cat sitting on a windowsill during a storm"
>
> 1. "A black cat sitting on a windowsill during a **light drizzle**"
> 2. "A black **dog** sitting on a windowsill during a light drizzle"

## Build up in steps

Start short and add one detail per generation. You catch the moment a new detail pulls the image off course.

> 1. "A medieval castle"
> 2. "A medieval castle on a hilltop"
> 3. "A medieval castle on a hilltop at sunset"
> 4. "A medieval castle on a hilltop at sunset with a dragon flying overhead"

## Be precise

Ideogram reads every word. "A painting of a cat" gets you something painterly; naming the kind of painting decides which one.

<table data-view="cards"><thead><tr><th></th><th></th><th></th></tr></thead><tbody><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FrmjNJSKpa4vTJzHkvxb0%2Fimage.png?alt=media&amp;token=ca901e12-73c1-4dc4-b545-18e84369cc6e" alt="Traditional painting of a black-and-white cat sleeping on a red armchair beside a lit fireplace in a cozy room" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>A painting figuring a cat sleeping next to the fireplace.</em></p></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FEWqArUp37Fy67x0fr2gR%2Fimage.png?alt=media&amp;token=9b61e920-150c-4247-8358-00fbe9e2a7d3" alt="Thick impasto painting with heavy brushstrokes of an orange tabby cat curled asleep on the floor beside a fireplace" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>An <mark style="color:green;">abstract impasto</mark> painting figuring a cat sleeping next to the fireplace.</em></p></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FgLXKa1rcHbGzBDyxG4If%2Fimage.png?alt=media&amp;token=814a1505-0f6a-4e45-8f8f-4c5c2e697d90" alt="Soft minimalist watercolor of a calico cat sleeping by a fireplace, with a teal sofa and small side table nearby" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>A <mark style="color:green;">soft minimalist watercolor</mark> painting figuring a cat sleeping next to the fireplace.</em></p></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2FKTlGoqHD8jNCr3gy7b6l%2Fimage.png?alt=media&amp;token=015d1edc-4184-41e1-ad72-b0d9e1e3a0b8" alt="Bold pop art illustration of a black-and-white cat curled on a red armchair beside a fireplace and striped wallpaper" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>A <mark style="color:green;">pop art</mark> painting figuring a cat sleeping next to the fireplace.</em></p></td><td></td></tr></tbody></table>

Precision covers position too. Say where each thing sits and in what order, and Ideogram tends to follow:

<figure><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fm4WwDuohJOhW6EK0V1u5%2Fimage.png?alt=media&amp;token=52b93242-0d72-4709-9315-35dedc463f7e" alt="Three jars on a kitchen counter labelled Blueberry Jam, Strawberry Jam and Orange Marmalade, each topped with its fruit" width="375"><figcaption><p>"...The first jar on the left, in the foreground, is a blueberry jam jar, with a blueberry adorning its lid. The second jar, in the middle and slightly recessed, is a jar of strawberry jam... The third jar, on the right and farthest back, is a jar of orange marmalade..."</p></figcaption></figure>

## Describe only what should show

If you describe something, Ideogram tries to show it, even when that breaks the framing you asked for. Below, the goal was a portrait. Mentioning boots, or even just jeans, forced a full-length shot; cutting everything below the waist brought the camera in close.

<table data-view="cards"><thead><tr><th></th><th></th><th></th></tr></thead><tbody><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-d037f97d4b38e7ebeae5f979e12f846b642943e5%2Frefine-portrait-boots.png?alt=media" alt="Full-length photo of a young woman with copper hair in a mustard raincoat, dark jeans and white rain boots outside a flower shop" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>Portrait photo of a young woman with copper hair, wearing a mustard raincoat, wide-leg jeans, and white rain boots, in front of a flower shop.</em></p></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-fa5e8c954bc88e5a67e7a779f10b7bcf9a342cb5%2Frefine-portrait-jeans.png?alt=media" alt="Still a full-length photo of the woman in the mustard raincoat and light wide-leg jeans in front of the flower shop" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>Portrait photo of a young woman with copper hair, wearing a mustard raincoat, wide-leg jeans, </em><del><em><mark style="color:purple;">and white rain boots</mark></em></del><em>, in front of a flower shop.</em></p></td><td></td></tr><tr><td><img src="https://1799634369-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FzjhNby3LLsIikYuvxAJP%2Fuploads%2Fgit-blob-9e31db7c1e5f4034f9a58103297da189874b12dd%2Frefine-portrait-coat.png?alt=media" alt="Close head-and-shoulders portrait of the woman with copper hair in the mustard raincoat, flower buckets behind her" data-size="original"></td><td><p><strong>Prompt:</strong></p><p><em>Portrait photo of a young woman with copper hair, wearing a mustard raincoat, </em><del><em><mark style="color:purple;">wide-leg jeans, and white rain boots</mark></em></del><em>, in front of a flower shop.</em></p></td><td></td></tr></tbody></table>

## Try other words

Some words just don't produce the look you want. Swap in a synonym or describe the idea in plain visual terms:

> **Instead of** "lush jungle", **try** "dense rainforest"
>
> **Instead of** "sad expression", **try** "a face with downturned eyes and a slight frown"

If one swap doesn't help, change the related words around it too; a neighbor may be pulling the other way. A thesaurus or a chat assistant can suggest options.

## Emphasize what matters

When the image is right overall but a key detail comes out small or goes missing, give it more weight.

**Mention it more than once.** Refer to the same subject in the summary, the details and the framing:

> "A product photo of a men's perfume bottle named "Nightlife for men" in a sleek studio setup. The bottle is tall and rectangular with dark glass... The bottle stands upright... The bottle is centered in the frame..."

**Describe it more fully.** Add size, color, texture or placement, and tie it to the rest of the scene. If a person's shoes come out missing or odd, "wearing black leather boots that cast a shadow on the sidewalk" gives Ideogram both the object and where it sits.

## Pick the right tool

Rewording isn't always the fastest fix. Once you have an image you mostly like, edit it instead of rolling the dice again.

| Goal                                              | What to do                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| See what one word changes                         | Fix the [seed](https://docs.ideogram.ai/create/image-settings.md#seed), change one word, generate again              |
| Get more options from a prompt that works         | Generate again, or select **Recreate** on a result to restore its settings                   |
| Keep the look, try a variation                    | **Remix** it in the [Remix app](https://docs.ideogram.ai/edit/remix.md) with a tweaked prompt                        |
| Change one thing in a finished image              | **AI edit** in [Studio](https://docs.ideogram.ai/edit/studio.md), describing the change in words                     |
| Fix a small area (a hand, a face, a word of text) | **Select area** in Studio (paid plans), then describe what should be there                   |
| Show more of the scene or change the shape        | **Reframe** in Studio, or **Extend** (paid plans)                                            |
| Turn a short idea into a full prompt              | Write a few words and let [Magic Prompt](https://docs.ideogram.ai/create/image-settings.md#magic-prompt) expand them |
| Keep your exact wording                           | Turn Magic Prompt off                                                                        |
| Control the layout exactly                        | Write a [JSON prompt](https://docs.ideogram.ai/prompting/json-prompting.md)                                          |

If something keeps going wrong, check [Fix Common Problems](https://docs.ideogram.ai/prompting/fix-common-problems.md).


<!-- SOURCE: https://docs.ideogram.ai/prompting/fix-common-problems.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/fix-common-problems.md).

# Fix Common Problems

What to do when an image shows something you didn't want, ignores part of your prompt, or gets text, framing or style wrong.

Each problem below lists the usual cause and the fix. If you're exploring rather than aiming for one exact image, some of these "mistakes" can lead somewhere interesting.

## Something you said you didn't want shows up

**Why:** Image models don't handle "no", "without" or "not" well. They latch onto the word that follows. "A man without a beard" can give you a beard. A word elsewhere in the prompt, or a detail Magic Prompt added, can also suggest the thing you're trying to avoid.

**Fix:** Describe what you'd see if the thing weren't there.

> **Instead of** "no people in the room", **write** "an empty room with chairs neatly arranged"
>
> **Instead of** "a robot with no eyes", **write** "a robot with a smooth, featureless face"

Then check the rest of the prompt for anything that hints at the unwanted item. If Magic Prompt added it, turn [Magic Prompt](https://docs.ideogram.ai/create/image-settings.md#magic-prompt) off; it can also help in the other direction, since its rewrite often turns a negative into a positive.

On Ideogram 4.5 and 4.0, rephrasing is the only option. On Ideogram 3.0 you can also list what to avoid in the [Negative prompt](https://docs.ideogram.ai/create/image-settings.md#negative-prompt) setting.

## Extra objects, people or text appear

**Why:** A vague or crowded prompt leaves gaps, and Ideogram fills them. Magic Prompt can add details too.

**Fix:** Cut ideas you don't need and describe the main subject more fully. Use positive phrasing ("an empty field" rather than "no one around"). Turn Magic Prompt off if it keeps adding things.

## Text is misspelled, broken or missing

**Why:** The text isn't marked clearly, there's too much of it, or the scene around it is busy. Non-Latin scripts, such as Arabic or Chinese, render less reliably than English.

**Fix:** Put the exact words in quotes, mention them early, and keep them short. Fix a wrong letter afterward with **Select area** in [Studio](https://docs.ideogram.ai/edit/studio.md) (paid plans), or add long copy as a [text layer](https://docs.ideogram.ai/edit/text-and-layers.md) in Studio. Keep non-Latin text to a word or two. See [Text in Images](https://docs.ideogram.ai/prompting/text-in-images.md).

## Faces, hands or limbs look distorted

**Why:** When a person is small in the frame, there are too few pixels for fine detail. This is a limit of the model more than of your wording.

**Fix:** Move closer with framing words such as "close-up" or "portrait", and name the details you care about (hands, eyes). If you like the rest of the image, use **Select area** in Studio (paid plans) on the flawed part instead of starting over.

## The subject is cropped, or too far away

**Why:** Ideogram fills the whole frame, so the [aspect ratio](https://docs.ideogram.ai/create/image-settings.md#aspect-ratio-and-size) shapes the framing. "A woman walking on a busy city street sidewalk" may show her head to toe at 1:2 and from the waist up at 2:1.

**Fix:**

* Name the framing: "full body", "head and shoulders", "wide view".
* For a full figure, describe what's near the feet (shoes, the sidewalk). For a close-up, describe the face and upper body only.
* Make the setting the main subject for a wider shot; make the person the main subject for a tighter one.
* Pick an aspect ratio that suits the subject, or use **Reframe** or **Extend** (paid) in Studio on an image you already have.
* For exact placement on 4.5 or 4.0, use a [JSON prompt](https://docs.ideogram.ai/prompting/json-prompting.md).

## Ideogram ignores part of your prompt

**Why:** The detail comes late in a long prompt, gets little weight, or conflicts with something else.

**Fix:** Move it earlier, describe it in more detail, or mention it more than once. See [Emphasize what matters](https://docs.ideogram.ai/prompting/refine-and-iterate.md#emphasize-what-matters).

## The prompt contradicts itself

**Why:** Conflicting details, such as "a minimalist sculpture with fine and intricate details", force Ideogram to pick one or blend both badly.

**Fix:** Choose one direction.

> **Instead of** "a clean, empty room cluttered with artifacts", **try** "a clean, empty room with plain white walls and a single wooden chair" **or** "a room filled with ancient artifacts on simple white pedestals"

## The style or mood is off

**Why:** Words like "beautiful", "cool", "artistic" or "modern" don't point to anything Ideogram can draw.

**Fix:** Name a medium, technique or movement, and describe what you'd see.

> **Instead of** "a modern painting of a landscape", **try** "an impressionist painting of a rolling countryside with thick brushstrokes and pastel tones"
>
> **Instead of** "a beautiful dress", **try** "a red satin evening gown with intricate lace details"

On 4.5 and 4.0, set style and color in the prompt; you can give exact hex colors in a [JSON prompt](https://docs.ideogram.ai/prompting/json-prompting.md). On Ideogram 3.0 you can also use style presets, style reference images and color palettes.

## The emotion or idea doesn't come through

**Why:** Abstract ideas like "hope" or "regret" have no single look, so results vary widely.

**Fix:** Tie the idea to something visible: a pose, an expression, an object, a place.

> **Instead of** "a symbol of hope", **try** "a single flower blooming through a crack in the concrete"
>
> **Instead of** "an old man lost in regret", **try** "an old man sitting alone on a park bench, staring down at a faded photo in his hands"

## A word or phrase isn't working

**Why:** Some words aren't tied to a clear look, or other words nearby pull against them.

**Fix:** Try a more visual synonym, or change the related words together. See [Try other words](https://docs.ideogram.ai/prompting/refine-and-iterate.md#try-other-words).

***

If you're still stuck, strip the prompt back to its core and rebuild it one detail at a time, as in [Build up in steps](https://docs.ideogram.ai/prompting/refine-and-iterate.md#build-up-in-steps).


<!-- SOURCE: https://docs.ideogram.ai/prompting/vocabulary.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/vocabulary.md).

# Vocabulary Reference

Words that help you describe camera angles, people and colors precisely

Ideogram draws what you name. A precise word ("worm's-eye view", "cranberry red", "toddler") gets you closer than a general one ("from below", "red", "young child"). These lists collect words that work well in prompts:

* [Angle of View and Perspective](https://docs.ideogram.ai/prompting/vocabulary/angle-of-view-and-perspective.md): camera positions and how a subject faces the viewer.
* [Describing Age and Life Stage](https://docs.ideogram.ai/prompting/vocabulary/describing-age-and-life-stage.md): life-stage terms and modifiers, which work better than numeric ages.
* [Describing Body Type](https://docs.ideogram.ai/prompting/vocabulary/describing-body-type.md): neutral words for build and shape.
* [Describing Skin Tones](https://docs.ideogram.ai/prompting/vocabulary/describing-skin-tones.md): depth, undertone and finish.
* [Memory Colors for Naming Color Nuances](https://docs.ideogram.ai/prompting/vocabulary/memory-colors-for-naming-color-nuances.md): everyday color names, grouped by base color.

To steer colors with hex codes, use a [JSON prompt](https://docs.ideogram.ai/prompting/json-prompting.md) on 4.5 or 4.0, or a Custom [Color palette](https://docs.ideogram.ai/create/image-settings.md#color-palette) on 3.0.


<!-- SOURCE: https://docs.ideogram.ai/prompting/vocabulary/angle-of-view-and-perspective.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/vocabulary/angle-of-view-and-perspective.md).

# Angle of View and Perspective

In image generation, the angle from which a scene or subject is viewed can dramatically affect the composition, storytelling, and overall feel of the result. This appendix outlines a variety of useful terms for describing point of view — both for the scene as a whole, and for individual people, animals, or objects within it.

#### 📸 Scene-level perspective (Camera-to-environment angle)

These terms describe how the viewer is positioned relative to the entire scene or environment.

* **Bird’s-eye view** — looking down from high above (also: aerial view, top-down perspective)
* **Worm’s-eye view** — looking up from ground level (also: low-ground view)
* **Overhead view** — directly above the subject (similar to: bird’s-eye, top-down)
* **Aerial view** — wide view from a high altitude (drone-style)
* **Isometric view** — angled top-down with parallel lines and no distortion (also: game map view, simulated 3D)
* **Wide-angle view** — expansive field of vision (also: cinematic wide shot, landscape framing)
* **Establishing shot** — broad scene-setting view (also: intro frame, scene overview)
* **Panoramic view** — ultra-wide horizontal framing (also: 360° view, landscape sweep)
* **Side view** — looking across from the left or right (also: lateral view, profile of the scene)
* **Tilted angle (Dutch angle)** — slanted horizon (also: skewed angle, off-kilter frame)
* **Point of view (POV)** — from a character’s visual perspective (also: first-person view)
* **Over-the-shoulder** — behind a subject, viewing what they see (also: behind POV)
* **Distant view** — subject seen from afar (also: far-shot, wide establishing)

#### 👁️ Subject-level perspective (Viewing a person, animal, or object)

These terms describe how the subject itself is being viewed or framed in the composition.

* **Front view** — directly facing the subject (also: straight-on view)
* **Side profile** — side of the face or body (also: profile view, lateral angle)
* **Back view** — viewing the subject from behind (also: rear angle)
* **Three-quarter view** — angled between front and side (also: partial side view)
* **Close-up** — tightly framed (also: portrait crop, detail view)
* **Extreme close-up** — single facial feature or small detail (e.g., just eyes, lips, hand)
* **Full-body shot** — head to toe (also: wide crop of the subject)
* **Headshot** — upper torso and face (also: bust shot, portrait frame)
* **Low angle** — looking up (also: heroic angle, upward shot)
* **High angle** — looking down (also: downward shot, overhead crop)
* **Eye-level** — neutral, straight-on framing (also: natural perspective)
* **Overhead angle** — directly above the subject (like from a drone or ceiling)
* **Underside view** — from beneath the subject (also: under-angle, worm’s/ant's perspective)
* **Behind-the-subject** — subject turned away (also: back-facing composition)
* **Obscured or cropped view** — subject partially hidden or off-frame (also: partial view)

You can combine these terms with emotional cues, lens types, or action words for even more control.\
For example:

> * *Aerial view of a foggy village*
> * *Over-the-shoulder shot of a warrior facing the horizon*
> * *Three-quarter close-up of a woman smiling*
> * *Low-angle view of a tree glowing in moonlight*
> * *Full-body front view of a seated child*

#### Be careful with conflicting perspective elements

When using angle or point-of-view terms in your prompts, make sure the other details you describe make sense from that same viewpoint. Since the AI tries to include everything you mention, asking for something that wouldn’t normally be visible from a specific angle can confuse it — and may cause the model to ignore the perspective altogether.

**Examples of possible contradictions:**

> * *A top-down bird’s-eye view of a city \[…] the sky is filled with fluffy white clouds.*
> * *A rear view of a man walking away \[…] he is smiling at the camera.*
> * *An extreme close-up of a woman’s face \[…] she is wearing a knee-length red dress.*


<!-- SOURCE: https://docs.ideogram.ai/prompting/vocabulary/describing-age-and-life-stage.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/vocabulary/describing-age-and-life-stage.md).

# Describing Age and Life Stage

When prompting for people, describing age clearly can shape the outcome just as much as physical features or clothing. However, many AIs don’t reliably interpret numeric ages (e.g., “a 7-year-old girl”) with visual accuracy. Instead of specifying numbers, it’s more effective to use descriptive, life stage–based terms. This appendix breaks those terms into two parts per group: age terms and modifiers to help communicate the age and physical traits more accurately.

You can mix and match across each category for more controlled image generation.

#### **Infant (0–1)**

* **Age terms**\
  Newborn, baby, infant
* **Modifiers**\
  Chubby, swaddled, soft-cheeked, sleeping, wide-eyed, tiny limbs, smooth skin, round cheeks

#### **Toddler (1–3)**

* **Age terms**\
  Toddler, young toddler, early walker
* **Modifiers**\
  Baby-faced, wobbling, curly-haired, pudgy, chubby cheeks, short limbs, clumsy

#### **Young Child (4–7)**

* **Age terms**\
  Young child, small child, preschooler, kindergartener
* **Modifiers**\
  Playful, curious, big-eyed, tousled hair, toothy smile, energetic, innocent expression, round face

#### **Older Child (8–12)**

* **Age terms**\
  Older child, grade-schooler, school-aged kid, preteen
* **Modifiers**\
  Freckled, active, gap-toothed, lean-limbed, lively, transitional build, early maturity

#### **Teenager (13–17)**

* **Age terms**\
  Teenager, adolescent, high school–aged, teenage girl/boy
* **Modifiers**\
  Moody, gangly, early puberty, developing features, serious gaze, youthful but mature, growing taller, soft jawline

#### **Young Adult (18–25)**

* **Age terms**\
  Young adult, college-aged adult, late teen, youthful adult, stylish young man/woman
* **Modifiers**\
  Fresh-faced, smooth-skinned, subtly mature, vibrant, minimal wrinkles, soft features

#### **Adult (26–39)**

* **Age terms**\
  Adult, early 30s adult, adult in their prime, mature young man/woman
* **Modifiers**\
  Refined features, confident, glowing skin, strong jawline, well-groomed, composed, healthy appearance, mature presence

#### **Middle-Aged (40–59)**

* **Age terms**\
  Middle-aged adult, mature adult, adult in their 40s or 50s
* **Modifiers**\
  Distinguished, subtle wrinkles, salt-and-pepper hair, thoughtful expression, experienced face, graceful aging, defined lines, mature elegance, visible signs of life experience

#### **Senior (60–79)**

* **Age terms**\
  Senior, older adult, elderly man/woman, grandparent
* **Modifiers**\
  Silver-haired, deep smile lines, wrinkled skin, wise eyes, cane or reading glasses

#### **Elder / Advanced Age (80+)**

* **Age terms**\
  Elder, aged elder, very old person
* **Modifiers**\
  Frail, thin white hair, deeply wrinkled, slow-moving, hunched posture, fragile yet dignified, timeless expression

#### Age modifiers by life stage

These descriptive modifiers can enhance the emotional tone, physical appearance, or personality of a character based on their age group. While some are flexible, most are naturally suited to a specific stage of life.

* **Youth-oriented modifiers:**\
  Youthful, baby-faced, radiant with youth, fresh-faced, full of life, soft-featured, innocent-looking
* **Neutral or crossover modifiers:**\
  Mature presence, confident, refined, composed, healthy-looking, well-groomed, graceful
* **Age-related or elder-oriented modifiers:**\
  Aging gracefully, time-worn, dignified aging, weathered by time, wise-looking, timeless beauty, deeply lined, gentle presence


<!-- SOURCE: https://docs.ideogram.ai/prompting/vocabulary/describing-body-type.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/vocabulary/describing-body-type.md).

# Describing Body Type

Neutral words for body build and shape to use when you prompt for people.

Body shape matters as much as age or clothing when you prompt for a person. Pick a base term, then add one or two visual details.

## Builds

* **Slim:** slim build, slender, narrow shoulders, long limbs, light frame
* **Average:** average build, medium frame, everyday proportions
* **Athletic:** athletic build, toned arms and legs, strong core, swimmer's build, runner's build
* **Muscular:** muscular build, broad shoulders, defined arms and chest, bodybuilder physique
* **Curvy:** curvy figure, full hips, defined waist, hourglass shape
* **Plus-size:** plus-size, full figure, soft and rounded build, wide frame
* **Large:** large build, heavyset, broad and heavy frame
* **Tall and lean:** lanky, tall and thin, wiry frame, long arms and legs
* **Stocky:** stocky build, short and broad, solid frame, wide chest
* **Androgynous:** androgynous build, straight silhouette, minimal curves

## Tips

* **Combine with pose and clothing.** "A plus-size woman in a fitted green dress, standing confidently" gives the model the body, the fit and the attitude.
* **Describe the frame you want shown.** Body details below the waist pull the framing out to a full-length shot. See [Describe only what should show](https://docs.ideogram.ai/prompting/refine-and-iterate.md#describe-only-what-should-show).
* **Keep it specific, not judgmental.** Words like "toned", "broad" or "soft" describe a shape. Loaded words tend to give exaggerated or caricatured results.

## Examples

> * A full-length photo of a stocky older man with a short grey beard, wearing a work jacket
> * A slim dancer with long limbs, mid-leap in a studio
> * A plus-size model in a linen suit, walking down a city street


<!-- SOURCE: https://docs.ideogram.ai/prompting/vocabulary/describing-skin-tones.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/vocabulary/describing-skin-tones.md).

# Describing Skin Tones

Words for skin tone depth, undertone and finish that give more precise results than broad color names.

"Light skin" or "dark skin" leaves a lot to chance. Describe skin with three things: how deep the tone is, its undertone, and how it catches the light.

## Depth

From lightest to deepest:

* **Very light:** porcelain, alabaster, ivory, pale
* **Light:** fair, light beige, cream, peach
* **Light to medium:** beige, sand, light golden, light olive
* **Medium:** golden, olive, honey, tan, warm beige
* **Medium to deep:** bronze, caramel, amber, light brown, copper
* **Deep:** brown, rich brown, mahogany, chestnut, dark brown
* **Very deep:** deep brown, ebony, deep mahogany

## Undertone

Add one of these to pin down the color beneath the surface:

* **Cool:** pink, rosy or blue-ish undertones
* **Warm:** golden, yellow or peach undertones
* **Neutral:** a balance of warm and cool
* **Olive:** a green-gold undertone

## Finish and light

* **Sun-kissed** or **freckled** for skin that shows time outdoors
* **Luminous**, **dewy** or **glowing** for a fresh, lit look
* **Matte** for skin without shine
* **Weathered** for older skin with texture

## Examples

> * A portrait of a woman with deep brown skin and warm undertones, soft window light
> * A fisherman with weathered, sun-kissed olive skin
> * A teenager with fair, freckled skin and cool undertones
> * Two friends laughing, one with light golden skin, the other with rich mahogany skin

Name the tone directly rather than relying on a nationality or ethnicity to imply it. People of every background have a wide range of skin tones.


<!-- SOURCE: https://docs.ideogram.ai/prompting/vocabulary/memory-colors-for-naming-color-nuances.md -->

> For the complete documentation index, see [llms.txt](https://docs.ideogram.ai/llms.txt). Markdown versions of documentation pages are available by appending `.md` to page URLs; this page is available as [Markdown](https://docs.ideogram.ai/prompting/vocabulary/memory-colors-for-naming-color-nuances.md).

# Memory Colors for Naming Color Nuances

When describing colors in a prompt, using specific, visually grounded references is much more effective than generic terms like “red” or “green.” Since Ideogram doesn’t understand numerical color codes (like RGB or hex) unless you use [JSON prompting](https://docs.ideogram.ai/prompting/json-prompting.md) on Ideogram 4.5 or 4.0, or a Custom [Color palette](https://docs.ideogram.ai/create/image-settings.md#color-palette) on 3.0, using familiar, real-world references — often called *memory colors* — helps convey a more accurate color nuance. These references evoke a mental image based on shared visual experience, such as “cherry red” or “sky blue.”

The list below groups memory-based color terms by base color. Each one can help you fine-tune your prompt and get closer to the exact hue you’re aiming for.

**🔴 Red**\
Apple, cherry, cranberry, blood, scarlet, ruby, wine, burgundy, brick, rose, garnet, fire engine, tomato, coral, blush, pomegranate, strawberry, chili, paprika, beet, firelight, jam, sangria, red velvet, ember

**🟠 Orange**\
Pumpkin, tangerine, apricot, rust, amber, carrot, marmalade, clay, copper, burnt orange, squash, paprika, cantaloupe, butternut, saffron, cheddar, flame, tiger, persimmon, ginger, ochre

**🟡 Yellow**\
Lemon, canary, gold, butter, mustard, sunflower, honey, daffodil, marigold, corn, banana, champagne, straw, yolk, dandelion, custard, pineapple, flax, amber glow, maize, goldenrod

**🟢 Green**\
Emerald, olive, sage, forest, moss, mint, jade, chartreuse, pistachio, lime, seafoam, fern, avocado, shamrock, basil, eucalyptus, pear, pickle, ivy, clover, pine, cactus, wasabi, celery

**🔵 Blue**\
Sky, baby blue, robin’s egg, navy, royal, sapphire, denim, indigo, ice blue, slate, teal, powder blue, steel blue, periwinkle, cobalt, storm, glacier, cornflower, ink, horizon, arctic, bluebell, lake, dusk

**🟣 Purple**\
Lavender, plum, violet, eggplant, orchid, grape, amethyst, wine, mauve, lilac, iris, mulberry, blackberry, heather, wisteria, fig, thistle, aubergine, twilight, royal purple, raisin, elderberry

**⚪ White / Off-White**\
Pearl, cream, ivory, alabaster, porcelain, eggshell, chalk, snow, linen, milk, moonlight, lace, frosting, meringue, cloud, rice paper, marble, parchment, vanilla, whipped cream, winter white

**⚫ Black / Gray**\
Charcoal, graphite, slate, ash, onyx, coal, soot, obsidian, lead, pewter, smoke, shadow, ink, iron, flint, steel, raven, gunmetal, stormcloud, tar, night, cinder

**🟤 Brown / Tan**\
Chocolate, coffee, cinnamon, chestnut, caramel, toffee, walnut, sand, taupe, ochre, clay, hazelnut, sepia, sienna, almond, pecan, mocha, maple, cocoa, bronze, dirt, suede, umber

**🩷 Pink**\
Rose, blush, salmon, bubblegum, flamingo, cotton candy, coral, peach, watermelon, raspberry, fuchsia, magenta, strawberry milk, hibiscus, tulip, rose quartz, guava, cherry blossom, lipstick

**🩵 Cyan / Aqua**\
Aqua, turquoise, glacier, lagoon, ocean, seafoam, pool, iceberg, teal, cyan, electric blue, mint, Caribbean, celeste, marine, arctic water, jellyfish glow, frostbite, cerulean, surf, wave

**🌈 Multicolored / Iridescent**\
Oil slick, holographic, rainbow, opal, abalone, prism, pearlized, shimmer, aurora, peacock feather, soap bubble, CD surface, crystal sheen, beetle shell, starlight, dragonfly wing

#### Descriptive modifiers for color

To further refine the color you're describing, you can combine memory-based color terms with descriptive modifiers. These help define the **intensity**, **temperature**, **finish**, or **lighting** of the color — allowing for more control and expressiveness in your prompts.

Here are common categories and examples of useful modifiers:

* **Intensity or Brightness**\
  Pale, light, soft, faint, pastel, bright, deep, vivid, bold, dark, muted, rich, intense
* **Warmth and Temperature**\
  Warm, cool, neutral, icy, frosted, fiery, dusky, earthy
* **Finish or Surface Quality**\
  Glossy, matte, metallic, shimmering, iridescent, pearlescent, velvet-like, translucent, silky, powdery
* **Light and Shadow Modifiers**\
  Sunlit, shadowed, backlit, faded, dimmed, glowing, reflective, foggy, moonlit

**Examples**\
You can combine these with memory-based color terms for added clarity and control:

> * *Powdery sky blue*
> * *Rich cherry red*
> * *Frosted mint green*
> * *Dusky rose pink*
> * *Shimmering sapphire blue*
> * *Matte ivory white*

