# FLUX Prompt Reference

Reviewed: 2026-10-29. Camera, lighting and composition vocabulary below is
retained from the earlier snapshot. The [current reference](https://docs.bfl.ml/guides/prompting_unified_reference.md)
also covers [FLUX 3](flux3_prompting.md). Its current FLUX.2 reference limits
distinguish API requests (eight for max/pro/flex, four for Klein), Playground
(up to ten), and dev inference (recommended six). Pro also limits combined
input/output pixels. Check the selected endpoint; these are not universal
local sampler limits.

Vendored snapshot from: https://docs.bfl.ml/guides/prompting_unified_reference

Fetched: 2026-06-03

Note: Original `llms.txt` URL `https://docs.bfl.ml/guides/prompting_guide_t2i_advanced` returned 404; this is the current BFL replacement page.

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bfl.ml/llms.txt
> Use this file to discover all available pages before exploring further.

# Prompt Reference

> Cheat sheet with camera terms, lighting and style keywords, and ready-to-use example prompts

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

## Camera & Lens Cheat Sheet

| Term                | Effect                                            |
| ------------------- | ------------------------------------------------- |
| **f/1.4 – f/2.8**   | Blurry background (shallow depth of field)        |
| **f/8 – f/16**      | Everything sharp (deep depth of field)            |
| **24mm**            | Wide angle — shows more of the scene              |
| **35mm**            | Natural, documentary-style perspective            |
| **50mm**            | Eye-level, neutral perspective                    |
| **85mm**            | Portrait-ideal, slight background compression     |
| **135mm+**          | Telephoto — strong background compression         |
| **ISO 100**         | Clean image, low noise                            |
| **ISO 1600–3200**   | Brighter but grainy — useful for film-style looks |
| **Macro lens**      | Extreme close-up detail                           |
| **Anamorphic lens** | Widescreen cinematic look with oval bokeh         |

<PromptDisplay prompt="Shot on Hasselblad X2D, 80mm lens, f/2.8, natural lighting" />

## Lighting Keywords

| Term                     | Effect                                                       |
| ------------------------ | ------------------------------------------------------------ |
| **Golden hour**          | Warm, soft, flattering — just after sunrise or before sunset |
| **Blue hour**            | Cool, moody — just before sunrise or after sunset            |
| **Overcast**             | Flat, even, shadow-free — great for product shots            |
| **Rembrandt lighting**   | Dramatic triangle of light on the face                       |
| **Split lighting**       | High contrast, half-face illuminated                         |
| **Chiaroscuro**          | Strong light/shadow drama                                    |
| **Backlit / rim light**  | Subject glowing at the edges                                 |
| **Soft box / key light** | Studio, controlled, even                                     |
| **Practical lighting**   | Light sources visible in the scene (lamps, neon, fire)       |
| **Diffused light**       | Soft, wrap-around, minimal shadows                           |
| **Harsh direct light**   | Strong shadows, high contrast                                |

**Example lighting phrases:**

<PromptDisplay prompt="soft, diffused natural light filtering through sheer curtains" />

<PromptDisplay prompt="dramatic side lighting creating deep shadows and highlights" />

<PromptDisplay prompt="golden hour backlighting with lens flare" />

<PromptDisplay prompt="overcast light creating even, shadow-free illumination" />

## Style Keywords

| Category         | Keywords                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Photographic** | "shot on Kodak Portra 400", "35mm film", "IMAX camera", "Sony A7IV", "Hasselblad X2D", "Canon 5D"                  |
| **Cinematic**    | "cinematic", "anamorphic lens flare", "teal and orange color grading", "film noir", "Roger Deakins cinematography" |
| **Artistic**     | "oil painting", "watercolor", "pencil sketch", "impasto texture", "Art Nouveau", "Bauhaus"                         |
| **Digital art**  | "concept art", "matte painting", "octane render", "unreal engine", "stylized 3D"                                   |
| **Illustration** | "flat design", "vector illustration", "comic art", "anime style", "graphic novel", "whimsical"                     |
| **Vintage**      | "80s vintage photo", "2000s digicam", "VHS aesthetic", "polaroid", "sepia tone"                                    |

## Composition Techniques

| Technique                        | When to Use                             | Example Phrase                                        |
| -------------------------------- | --------------------------------------- | ----------------------------------------------------- |
| **Rule of thirds**               | Natural, balanced framing               | "composed using rule of thirds"                       |
| **Leading lines**                | Guide the eye through the image         | "diagonal lines leading to main entrance"             |
| **Foreground/background layers** | Add depth and dimension                 | "strong foreground boulder, background mountains"     |
| **Low angle (worm's eye)**       | Make subjects powerful and dominant     | "low angle worm's eye view, dramatic diagonal lines"  |
| **High angle (bird's eye)**      | Show patterns and spatial relationships | "bird's eye view, geometric patterns of city blocks"  |
| **Dutch angle**                  | Tension and psychological unease        | "dutch angle, psychological tension"                  |
| **Symmetrical**                  | Formal, balanced, architectural         | "perfectly symmetrical composition"                   |
| **Negative space**               | Minimal, focused, product               | "minimalist composition with generous negative space" |

## Model-Specific Quick Reference

<AccordionGroup>
  <Accordion title="FLUX.2 [pro] & [max]">
    * No negative prompts supported
    * Excellent typography — use quotation marks for exact text
    * HEX color codes for brand-precise color matching: `"in color #FF5733"`
    * JSON structured prompts supported for production workflows
    * Reference specific camera models for authentic photorealistic looks

    See: [JSON Prompting →](https://docs.bfl.ml/guides/usecases_t2i_json_prompting) · [Typography & Design →](https://docs.bfl.ml/guides/usecases_t2i_typography_design) · [HEX Color Prompting →](https://docs.bfl.ml/guides/usecases_t2i_hex_color_prompting)
  </Accordion>

  <Accordion title="FLUX.2 [klein]">
    * What you write is exactly what the model receives
    * Write in prose, not keyword lists — describe scenes like a novelist
    * Lighting descriptions have the highest single impact on output quality
    * Supports image editing with single and multi-reference inputs
    * Add `Style: [style]. Mood: [mood].` at the end for consistent aesthetics

    See: [Single-Reference Editing →](https://docs.bfl.ml/guides/prompting_editing_single_reference) · [Multi-Reference Editing →](https://docs.bfl.ml/guides/prompting_editing_multi_reference)
  </Accordion>

  <Accordion title="FLUX.1 Kontext">
    * Specify what should **change** — the input image provides all other visual context
    * Use quotation marks for text editing: `Replace 'joy' with 'BFL'`
    * Be explicit about preservation: *"while maintaining the same facial features and hairstyle"*
    * Prefer specific verbs: "change the clothes" over "transform the person"
    * For multiple changes, add as many explicit details as possible
  </Accordion>
</AccordionGroup>
