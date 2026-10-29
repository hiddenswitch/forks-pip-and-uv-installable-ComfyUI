# FLUX Style, Aesthetics & Text

Reviewed: 2026-10-29. The examples below retain the earlier FLUX.1/2 snapshot.
The [current style and text guide](https://docs.bfl.ml/guides/prompting_unified_style.md)
now includes FLUX 3 Image layouts, reference ordering, and typography examples.
See [FLUX 3](flux3_prompting.md) before transferring API or layout instructions.

Vendored snapshot from: https://docs.bfl.ml/guides/prompting_unified_style

Fetched: 2026-06-03

Note: Original `llms.txt` URL `https://docs.bfl.ml/guides/prompting_guide_t2i_essentials` returned 404; this is the current BFL replacement page covering style, aesthetics, lighting, and text.

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bfl.ml/llms.txt
> Use this file to discover all available pages before exploring further.

# Style, Aesthetics & Text

> Style keywords, photorealistic looks, illustration styles, and how to get readable text in images

export const PromptDisplay = ({prompt}) => {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return <div className="not-prose" style={{
    marginTop: "1rem"
  }}>
      <div style={{
    backgroundColor: "#1a1a1a",
    borderRadius: "1rem",
    padding: "1.25rem 1.5rem",
    display: "flex",
    flexDirection: "column",
    gap: "1rem"
  }}>
        <p style={{
    color: "#e5e5e5",
    fontSize: "1rem",
    lineHeight: 1.6,
    margin: 0,
    fontFamily: "inherit"
  }}>
          {prompt}
        </p>
        <div style={{
    display: "flex",
    justifyContent: "flex-end"
  }}>
          <button onClick={copy} style={{
    backgroundColor: copied ? "#3d8a5b" : "var(--aspen-evergreen, #486A58)",
    color: "#fff",
    border: "none",
    borderRadius: "0.375rem",
    padding: "0.25rem 0.6rem",
    fontSize: "0.7rem",
    fontWeight: 600,
    cursor: "pointer",
    transition: "background-color 0.2s"
  }}>
            {copied ? "Copied!" : "Copy prompt"}
          </button>
        </div>
      </div>
    </div>;
};

## Photorealistic Styles

FLUX generates photorealistic images from natural language. Reference specific eras, cameras, and film stocks for distinctive looks.

| Style              | Key Descriptors                                                                      |
| ------------------ | ------------------------------------------------------------------------------------ |
| **Modern Digital** | "shot on Sony A7IV, clean sharp, high dynamic range"                                 |
| **2000s Digicam**  | "early digital camera, slight noise, flash photography, candid, 2000s digicam style" |
| **80s Vintage**    | "film grain, warm color cast, soft focus, 80s vintage photo"                         |
| **Analog Film**    | "shot on Kodak Portra 400, natural grain, organic colors"                            |

<Tabs>
  <Tab title="Modern Photorealism">
    <Frame>
      ![Tiger cub under banana leaf in rainy jungle](https://cdn.sanity.io/images/2gpum2i6/production/bc6432890df19f127624df9eb47c7d5fdd984a3d-2656x1504.png)
    </Frame>

    <PromptDisplay prompt="Soaking wet tiger cub taking shelter under a banana leaf in the rainy jungle, close up photo" />
  </Tab>

  <Tab title="2000s Digicam">
    <Frame>
      ![Sloth in Bangkok nightlife, digicam style](https://cdn.sanity.io/images/2gpum2i6/production/d02f27fc01e66dca0f812c02678e3502cf664d19-1328x752.png)
    </Frame>

    <PromptDisplay prompt="Sloth out drinking in Bangkok at night in a street full of party folks, 2000s digicam style, people in the background fading" />
  </Tab>

  <Tab title="80s Vintage">
    <Frame>
      ![Baby penguins in trampoline park, 80s vintage](https://cdn.sanity.io/images/2gpum2i6/production/ef2490df716e6b6eeaf7d7bc4e9af0200fa53b52-1328x752.png)
    </Frame>

    <PromptDisplay prompt="A group of baby penguins in a trampoline park, having the time of their lives, 80s vintage photo" />
  </Tab>

  <Tab title="Analog Photography">
    <Frame>
      ![Old faded family portrait](https://cdn.sanity.io/images/2gpum2i6/production/dc7ce17bfc1ec5d526ae33bc975d00459590f9de-1328x752.png)
    </Frame>

    <PromptDisplay prompt="An old faded family portrait photograph from the early 2000s showing a family of five standing stiffly in front of their modest wooden farmhouse" />
  </Tab>
</Tabs>

<Tip>For photorealism, specify camera models, lenses, and film stocks. "Shot on Fujifilm X-T5, 35mm f/1.4" produces more authentic results than just "professional photo."</Tip>

### Camera and Lens Simulation

Be specific about camera settings for authentic results:

```
Shot on Hasselblad X2D, 80mm lens, f/2.8, natural lighting
```

```
Canon 5D Mark IV, 24-70mm at 35mm, golden hour, shallow depth of field
```

## Art Styles & Illustration

FLUX handles a wide range of artistic styles beyond photorealism. Name the style specifically and describe its visual characteristics.

**Style fusion** — Combine two styles with a unifying palette:

<Card title="Style Fusion Example" horizontal img="https://cdn.sanity.io/images/gsvmb6gz/production/f11c404eed10d12b215b7a07e601bf41e7fcc4bb-1392x752.jpg">
  *"Ancient Greek marble statue precision and anatomical detail, infused with cyberpunk neon lighting, holographic overlays, and electric blue/magenta glow effects, set against dark futuristic environments"*
</Card>

**Style + mood annotations** — Add explicit tags at the end of your prompt for consistent aesthetics:

```
[Scene description]. Style: Country chic meets luxury lifestyle editorial.
Mood: Serene, romantic, grounded.
```

```
[Scene description]. Shot on 35mm film (Kodak Portra 400) with shallow
depth of field — subject razor-sharp, background softly blurred.
```

<Columns cols={3}>
  <Frame caption="1990s fashion editorial">
    <img src="https://cdn.sanity.io/images/2gpum2i6/production/ad3f43a52bdf1b82350e58da90d50d0122b8dc14-1440x2048.jpg" alt="1990s editorial" />
  </Frame>

  <Frame caption="Surreal interior">
    <img src="https://cdn.sanity.io/images/2gpum2i6/production/dd1aa1741cc7a7e8149672f0aa65fc326561bbc6-1440x2048.jpg" alt="Surreal interior" />
  </Frame>

  <Frame caption="Golden hour silhouette">
    <img src="https://cdn.sanity.io/images/2gpum2i6/production/a182f770a402a3b318336d6156b5eb1dd8c80cec-1440x2048.jpg" alt="Golden hour silhouette" />
  </Frame>
</Columns>

## Lighting

Lighting has the **greatest single impact** on output quality. Describe it like a photographer would — "good lighting" is not enough.

**What to describe:**

* **Source**: natural, artificial, ambient
* **Quality**: soft, harsh, diffused, direct
* **Direction**: side, back, overhead, fill
* **Temperature**: warm, cool, golden, blue
* **Interaction**: catches, filters, reflects on surfaces

<AccordionGroup>
  <Accordion title="Portrait Lighting">
    **Rembrandt lighting** (45° key light) — triangle of light on the face for dramatic portraits:

    *"Portrait with Rembrandt lighting, key light at 45 degrees, dramatic chiaroscuro effect"*

    **Split lighting** (90° side light) — half-face illuminated for high contrast:

    *"Artistic portrait, split lighting, strong side illumination, dramatic contrast"*

    <Columns cols={2}>
      <Frame caption="Rembrandt lighting">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/ae53f6422efdaf3f51e1e0bd928d7c3952e8a7de-1392x752.jpg" alt="Rembrandt lighting" />
      </Frame>

      <Frame caption="Split lighting">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/284c21b1028f62e395461f7e7e3142f2c7823386-1392x752.png" alt="Split lighting" />
      </Frame>
    </Columns>
  </Accordion>

  <Accordion title="Environmental Light Quality">
    **Window light** = soft, even illumination

    **Golden hour** = warm and soft

    **Blue hour** = cool and moody

    **Overhead artificial** = harsh and dramatic

    <Columns cols={2}>
      <Frame caption="Window Light">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/d95a5129f006cd34e184451a5e813be3f3237409-1392x752.jpg" alt="Window light" />
      </Frame>

      <Frame caption="Golden hour">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/b9635dc280dfaaba1684083b0f8bbb145eeb3ebf-1392x752.jpg" alt="Golden hour" />
      </Frame>

      <Frame caption="Blue hour">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/6fcccf61808d5cb23b8339a52ade9cd7adf435e3-1392x752.jpg" alt="Blue hour" />
      </Frame>

      <Frame caption="Overhead artificial light">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/9487dd7f4c2ac8a16f5a4ce130ffa010fa57f7f2-1392x752.jpg" alt="Overhead artificial light" />
      </Frame>
    </Columns>
  </Accordion>

  <Accordion title="Cinematic Lighting">
    **Chiaroscuro** — high contrast light/shadow for drama:

    *"Film noir detective scene, single practical desk lamp, strong chiaroscuro lighting"*

    **Practical lighting** — visible light sources in scene for realism:

    *"Cyberpunk street scene, neon signs and LED strips providing atmospheric lighting"*

    <Columns cols={2}>
      <Frame caption="Chiaroscuro">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/4ac0d069fa68429339f919d528e7c88e403184d6-1392x752.jpg" alt="Chiaroscuro" />
      </Frame>

      <Frame caption="Practical lighting">
        <img src="https://cdn.sanity.io/images/gsvmb6gz/production/ae31cf4f6cd3d434bddbf7d5659be68fd03c3d91-1392x752.jpg" alt="Practical lighting" />
      </Frame>
    </Columns>
  </Accordion>
</AccordionGroup>

## Text in Images

FLUX handles text well when prompted correctly. Use this three-step approach:

<Steps>
  <Step title="Enclose in Quotation Marks">
    Use quotes for exact text: `"COFFEE SHOP"` or `"Est. 1952"`
  </Step>

  <Step title="Describe Placement">
    Specify where text appears: *"The text 'OPEN' appears in red neon letters above the door"*
  </Step>

  <Step title="Specify Font Style">
    Name the style: *"elegant serif typography"* or *"bold industrial sans-serif lettering"*
  </Step>
</Steps>

<Tabs>
  <Tab title="Neon Sign">
    !['Open' neon sign](https://cdn.sanity.io/images/2gpum2i6/production/324a5024e679ee8a693821fc5bbdedb2c3fde129-1888x1056.png)

    <PromptDisplay prompt="A Entry of a Sushi Restaurant, The text 'OPEN' appears in red neon letters above the door" />
  </Tab>

  <Tab title="Product Advertisement">
    <Frame>
      <img src="https://cdn.sanity.io/images/2gpum2i6/production/7730d0482ee19d0d3207b5cbbb55074c3b66d554-1456x1920.png" alt="Phone advertisement" style={{ maxWidth:"420px",margin:"0 auto",display:"block" }} />
    </Frame>

    <PromptDisplay prompt="Samsung Galaxy S25 Ultra product advertisement, 'Ultra-strong titanium' headline, 'Shielded in a strong titanium frame, your Galaxy S25 Ultra always stays protected' subtext, close-up of phone edge showing titanium frame, dark gradient background, clean minimalist tech aesthetic, professional product photography" />
  </Tab>

  <Tab title="Retro Poster">
    <Frame>
      <img src="https://cdn.sanity.io/images/2gpum2i6/production/6d99bf4508c927ab16b67359b6ff70cdc992b279-1456x1920.png" alt="Groovy retro poster" style={{ maxWidth:"420px",margin:"0 auto",display:"block" }} />
    </Frame>

    <PromptDisplay prompt={'Groovy retro poster with the quote "If you love me let me sleep". Bold 70s typography in deep red and warm pink tones. Cream background and bold orange doodle around the text. Funky layout with playful shadow. Style: bold vintage aesthetic, dopamine decor'} />
  </Tab>

  <Tab title="Magazine Cover">
    <Frame>
      <img src="https://cdn.sanity.io/images/2gpum2i6/production/00ddd4ce8b582891f3b174462dc635dac4e45d46-1456x1920.jpg" alt="Magazine cover" style={{ maxWidth:"420px",margin:"0 auto",display:"block" }} />
    </Frame>

    <PromptDisplay prompt="Women's Health magazine cover, April 2025 issue, 'Spring forward' headline, woman in green outfit sitting on orange blocks, white sneakers, 'Covid: five years on' feature text, '15 skincare habits' callout, professional editorial photography, magazine layout with multiple text elements" />
  </Tab>
</Tabs>

### Text Rendering Tips

* **Front-load text descriptions** for better accuracy
* **Use quotation marks** around exact text you want rendered
* **Describe color and effects**: "red neon letters", "gold serif lettering", "chalk on blackboard"
* **Use hex codes** for brand-precise colors: *"The logo text 'ACME' in color #FF5733"*
* **Keep text short** — long strings are harder to render accurately
* **Specify font character**: serif = traditional/formal, sans-serif = modern, script = elegant, display = bold/impactful

### Typography Styles Reference

| Style             | Effect                 | Example                                                  |
| ----------------- | ---------------------- | -------------------------------------------------------- |
| **3D text**       | Dimensional, impactful | "raised chrome letters with realistic metal reflections" |
| **Neon effects**  | Atmospheric, glowing   | "glowing neon text with electric blue light"             |
| **Vintage signs** | Authentic, weathered   | "weathered painted text with chipped paint and rust"     |
| **Environmental** | Integrated into scene  | "carved directly into the ancient stone wall"            |
| **Object-based**  | Printed on props       | "printed on a newspaper being read by the character"     |
