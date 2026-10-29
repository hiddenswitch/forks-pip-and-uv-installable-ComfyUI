# FLUX Image Editing Prompting

Reviewed: 2026-10-29. This preserves the earlier FLUX.1 Kontext / FLUX.2 editing
snapshot. The general `prompting_editing_overview` source now describes FLUX 3.
For Kontext use the [dedicated official editing page](https://docs.bfl.ml/kontext/kontext_image_editing.md)
and [Kontext dev card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
For FLUX 3 use [its guide](flux3_prompting.md). Keep each model's reference-image
and request-field contract separate.

Vendored snapshot from: https://docs.bfl.ml/guides/prompting_editing_overview

Fetched: 2026-06-03

Note: Original `llms.txt` URL `https://docs.bfl.ml/guides/prompting_guide_kontext_i2i` returned 404; this is the current BFL replacement page.

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bfl.ml/llms.txt
> Use this file to discover all available pages before exploring further.

# Image Editing

> Overview of image editing with FLUX.2 — swap backgrounds, replace objects, transfer styles, and combine multi-reference images in natural language.

export const MultiRefGrid = ({inputs = [], result = {}, prompt = ""}) => {
  const [hoveredInput, setHoveredInput] = useState(-1);
  const colors = [{
    bg: "rgba(99,182,137,0.22)",
    border: "#63b689",
    text: "#63b689"
  }, {
    bg: "rgba(130,170,255,0.22)",
    border: "#82aaff",
    text: "#82aaff"
  }, {
    bg: "rgba(255,180,107,0.22)",
    border: "#ffb46b",
    text: "#ffb46b"
  }, {
    bg: "rgba(199,140,230,0.22)",
    border: "#c78ce6",
    text: "#c78ce6"
  }, {
    bg: "rgba(255,130,130,0.22)",
    border: "#ff8282",
    text: "#ff8282"
  }, {
    bg: "rgba(130,220,220,0.22)",
    border: "#82dcdc",
    text: "#82dcdc"
  }, {
    bg: "rgba(220,200,120,0.22)",
    border: "#dcc878",
    text: "#dcc878"
  }, {
    bg: "rgba(200,160,180,0.22)",
    border: "#c8a0b4",
    text: "#c8a0b4"
  }];
  const renderPrompt = () => {
    if (!prompt) return null;
    const parts = [];
    let lastIndex = 0;
    const re = /\b(image\s+(\d))\b/gi;
    let m;
    while ((m = re.exec(prompt)) !== null) {
      if (m.index > lastIndex) parts.push({
        text: prompt.slice(lastIndex, m.index),
        idx: -1
      });
      parts.push({
        text: m[1],
        idx: parseInt(m[2], 10) - 1
      });
      lastIndex = m.index + m[0].length;
    }
    if (lastIndex < prompt.length) parts.push({
      text: prompt.slice(lastIndex),
      idx: -1
    });
    return parts.map((p, i) => {
      if (p.idx >= 0 && p.idx < colors.length) {
        const c = colors[p.idx];
        return <span key={i} style={{
          color: c.text,
          fontWeight: 700,
          padding: "1px 5px",
          borderRadius: "3px",
          background: hoveredInput === p.idx ? c.bg : "transparent",
          transition: "background 200ms"
        }}>
            {p.text}
          </span>;
      }
      return <span key={i}>{p.text}</span>;
    });
  };
  const getGridCols = () => {
    const n = inputs.length;
    if (n <= 3) return n;
    if (n === 4) return 2;
    if (n <= 6) return 3;
    return 4;
  };
  const gridCols = getGridCols();
  return <div className="not-prose" style={{
    borderRadius: "0.75rem",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(0,0,0,0.35)"
  }}>

      {}
      <div style={{
    display: "grid",
    gridTemplateColumns: "repeat(" + gridCols + ", 1fr)",
    gap: "3px",
    padding: "3px",
    background: "rgba(0,0,0,0.3)"
  }}>
        {inputs.map((img, idx) => {
    const c = colors[idx % colors.length];
    const isHovered = hoveredInput === idx;
    return <div key={idx} onMouseEnter={() => setHoveredInput(idx)} onMouseLeave={() => setHoveredInput(-1)} style={{
      position: "relative",
      overflow: "hidden",
      cursor: "default",
      aspectRatio: "4/3",
      outline: isHovered ? "2px solid " + c.border : "2px solid transparent",
      outlineOffset: "-2px",
      transition: "outline-color 200ms ease",
      borderRadius: "4px"
    }}>
              <img src={img.src} alt={img.label || "Input " + (idx + 1)} style={{
      display: "block",
      width: "100%",
      height: "100%",
      objectFit: "cover",
      pointerEvents: "none"
    }} />
              <div style={{
      position: "absolute",
      top: "4px",
      left: "4px",
      width: "22px",
      height: "22px",
      borderRadius: "50%",
      background: c.border,
      color: "#000",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: "0.65rem",
      fontWeight: 800,
      boxShadow: "0 1px 4px rgba(0,0,0,0.5)"
    }}>
                {idx + 1}
              </div>
            </div>;
  })}
      </div>

      {}
      <div style={{
    display: "flex",
    justifyContent: "center",
    padding: "4px 0",
    background: "rgba(0,0,0,0.3)"
  }}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M12 5v14M6 13l6 6 6-6" stroke="rgba(255,255,255,0.3)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {}
      <div style={{
    position: "relative",
    padding: "0 3px 3px",
    background: "rgba(0,0,0,0.3)"
  }}>
        <img src={result.src} alt={result.label || "Result"} style={{
    display: "block",
    width: "100%",
    height: "auto",
    borderRadius: "4px",
    pointerEvents: "none"
  }} />
        <div style={{
    position: "absolute",
    top: "6px",
    right: "9px",
    padding: "3px 10px",
    borderRadius: "5px",
    background: "rgba(0,0,0,0.6)",
    backdropFilter: "blur(4px)",
    color: "#fff",
    fontSize: "0.65rem",
    fontWeight: 600
  }}>
          Result
        </div>
      </div>

      {}
      {prompt && <div style={{
    padding: "0.4rem 0.5rem 0.5rem"
  }}>
          <div style={{
    padding: "0.4rem 0.6rem",
    borderRadius: "6px",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    fontFamily: "monospace",
    fontSize: "0.68rem",
    lineHeight: 1.5,
    color: "rgba(255,255,255,0.7)"
  }}>
            <span style={{
    color: "rgba(255,255,255,0.35)",
    fontSize: "0.6rem",
    marginRight: "5px"
  }}>prompt:</span>
            {renderPrompt()}
          </div>
        </div>}
    </div>;
};

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

export const ImageComparisonSlider = ({beforeImage, afterImage, beforeLabel = "Before", afterLabel = "After", height = "500px", objectFit = "cover", garmentImage, garmentLabel = "Garment reference"}) => {
  const [position, setPosition] = useState(50);
  const dialogRef = useRef(null);
  const openLightbox = () => {
    if (dialogRef.current && typeof dialogRef.current.showModal === "function") {
      dialogRef.current.showModal();
    }
  };
  const closeLightbox = () => {
    if (dialogRef.current && typeof dialogRef.current.close === "function") {
      dialogRef.current.close();
    }
  };
  const getPosition = (e, container) => {
    const rect = container.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const x = clientX - rect.left;
    return Math.max(0, Math.min(100, x / rect.width * 100));
  };
  const onPointerDown = e => {
    e.preventDefault();
    e.stopPropagation();
    const container = e.currentTarget;
    setPosition(getPosition(e, container));
    const onMove = ev => {
      ev.preventDefault();
      setPosition(getPosition(ev, container));
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove, {
      passive: false
    });
    window.addEventListener("touchend", onUp);
  };
  return <div className="not-prose" style={{
    borderRadius: "1rem",
    overflow: "hidden",
    height,
    width: "100%"
  }}>
      <div onMouseDown={onPointerDown} onTouchStart={onPointerDown} onClick={e => {
    e.preventDefault();
    e.stopPropagation();
  }} style={{
    position: "relative",
    width: "100%",
    height,
    overflow: "hidden",
    cursor: "ew-resize",
    userSelect: "none",
    WebkitUserSelect: "none"
  }}>
        {}
        <img src={afterImage} alt={afterLabel} draggable={false} style={{
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    objectFit,
    pointerEvents: "none"
  }} />

        {}
        <div style={{
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    clipPath: `inset(0 ${100 - position}% 0 0)`
  }}>
          <img src={beforeImage} alt={beforeLabel} draggable={false} style={{
    display: "block",
    width: "100%",
    height: "100%",
    objectFit,
    pointerEvents: "none"
  }} />
        </div>

        {}
        <div style={{
    position: "absolute",
    top: 0,
    left: `${position}%`,
    transform: "translateX(-50%)",
    width: "3px",
    height: "100%",
    background: "rgba(255,255,255,0.85)",
    pointerEvents: "none",
    zIndex: 2
  }} />

        {}
        <div style={{
    position: "absolute",
    top: "50%",
    left: `${position}%`,
    transform: "translate(-50%, -50%)",
    width: "44px",
    height: "44px",
    borderRadius: "50%",
    background: "rgba(255,255,255,0.95)",
    border: "2px solid rgba(0,0,0,0.15)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "6px",
    zIndex: 3,
    pointerEvents: "none",
    boxShadow: "0 2px 8px rgba(0,0,0,0.3)"
  }}>
          {}
          <svg width="10" height="14" viewBox="0 0 10 14" fill="none" style={{
    marginRight: "-2px"
  }}>
            <path d="M8 1L2 7L8 13" stroke="#333" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          {}
          <svg width="10" height="14" viewBox="0 0 10 14" fill="none" style={{
    marginLeft: "-2px"
  }}>
            <path d="M2 1L8 7L2 13" stroke="#333" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        {}
        <div style={{
    position: "absolute",
    top: "12px",
    left: "12px",
    padding: "4px 10px",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.55)",
    backdropFilter: "blur(4px)",
    color: "#fff",
    fontSize: "0.75rem",
    fontWeight: 600,
    letterSpacing: "0.02em",
    pointerEvents: "none",
    zIndex: 4
  }}>
          {beforeLabel}
        </div>
        <div style={{
    position: "absolute",
    top: "12px",
    right: "12px",
    padding: "4px 10px",
    borderRadius: "6px",
    background: "rgba(0,0,0,0.55)",
    backdropFilter: "blur(4px)",
    color: "#fff",
    fontSize: "0.75rem",
    fontWeight: 600,
    letterSpacing: "0.02em",
    pointerEvents: "none",
    zIndex: 4
  }}>
          {afterLabel}
        </div>

        {}
        {garmentImage && <div onMouseDown={e => e.stopPropagation()} onTouchStart={e => e.stopPropagation()} onClick={e => {
    e.preventDefault();
    e.stopPropagation();
    openLightbox();
  }} title={`${garmentLabel} — click to enlarge`} style={{
    position: "absolute",
    bottom: "12px",
    right: "12px",
    height: "140px",
    maxWidth: "40%",
    borderRadius: "8px",
    overflow: "hidden",
    cursor: "zoom-in",
    background: "rgba(255,255,255,0.95)",
    border: "2px solid rgba(255,255,255,0.9)",
    boxShadow: "0 2px 12px rgba(0,0,0,0.45)",
    zIndex: 5,
    display: "flex",
    alignItems: "center",
    justifyContent: "center"
  }}>
            <img src={garmentImage} alt={garmentLabel} draggable={false} style={{
    height: "100%",
    width: "auto",
    objectFit: "contain",
    pointerEvents: "none"
  }} />
          </div>}
      </div>

      {}
      {garmentImage && <dialog ref={dialogRef} onClick={closeLightbox} onClose={closeLightbox} style={{
    padding: 0,
    border: "none",
    background: "transparent",
    maxWidth: "100vw",
    maxHeight: "100vh",
    width: "100vw",
    height: "100vh",
    overflow: "hidden"
  }}>
          <div onClick={closeLightbox} style={{
    width: "100vw",
    height: "100vh",
    background: "rgba(0,0,0,0.88)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "zoom-out",
    padding: "40px",
    boxSizing: "border-box"
  }}>
            <img src={garmentImage} alt={garmentLabel} onClick={e => e.stopPropagation()} style={{
    maxWidth: "90vw",
    maxHeight: "90vh",
    objectFit: "contain",
    borderRadius: "8px",
    boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
    cursor: "default"
  }} />
          </div>
        </dialog>}
    </div>;
};

export const EditingShowcase = () => {
  const [activeId, setActiveId] = useState("starting");
  const categories = [{
    id: "starting",
    title: "Starting Image",
    description: "The original image before any edits.",
    prompt: "A lone wolf standing on a rocky outcropping, bathed in golden light. The wind ruffles its thick grey fur as it gazes across a vast wilderness landscape.",
    image: "https://cdn.sanity.io/images/2gpum2i6/production/03cb44b883709a79500c46d8db8e0d0fc932413d-1440x1024.png"
  }, {
    id: "character",
    title: "Change the character",
    description: "Replace or transform people and creatures in your scene.",
    prompt: "Replace the wolf with a large silver fox, keeping the same pose \u2014 head tilted up, howling. The fox has a thick, bushy tail and piercing amber eyes. It stands in the same spot on the factory floor.",
    image: "https://cdn.sanity.io/images/2gpum2i6/production/da47586dc36c5a66fbeba00d6f124bf9dee6a2c9-1440x1024.png"
  }, {
    id: "composition",
    title: "Adjust the Composition",
    description: "Reframe and restructure your image.",
    prompt: "Zoom out to reveal more of the abandoned factory interior. Show towering rusted machinery on both sides, a collapsed roof section letting in dramatic shafts of light, and debris scattered across a vast concrete floor.",
    image: "https://cdn.sanity.io/images/2gpum2i6/production/ca941af1f2c7eddc09c50937b7f3f381fa686e42-1440x1024.png"
  }, {
    id: "action",
    title: "Alter the Action",
    description: "Change what the subject is doing.",
    prompt: "The wolf is now lying down peacefully on the factory floor, resting its head on its front paws. Eyes half-closed, calm and relaxed. A small bird has landed on its back.",
    image: "https://cdn.sanity.io/images/2gpum2i6/production/798b153d0e8228f1db3974cdfbeeb88701bc5d54-1440x1024.png"
  }, {
    id: "setting",
    title: "Swap the Setting",
    description: "Transform the environment, as much or as little as you like.",
    prompt: "Change the setting from a natural rock outcropping to have the wolf on the floor of a cavernous, abandoned factory. Shafts of light from shattered skylights pierce the dusty gloom, and illuminate silent, colossal machinery in the background.",
    image: "https://cdn.sanity.io/images/2gpum2i6/production/d4b701dab8e201072f6726175f1b9381b77ae769-1440x1024.png"
  }];
  const active = categories.find(c => c.id === activeId) || categories[0];
  return <div className="not-prose" style={{
    borderRadius: "1rem",
    overflow: "hidden",
    border: "1px solid rgba(255,255,255,0.1)",
    background: "#111210"
  }}>
      <div style={{
    position: "relative",
    aspectRatio: "16/9",
    overflow: "hidden"
  }}>
        <img src={active.image} alt={active.title} style={{
    display: "block",
    width: "100%",
    height: "100%",
    objectFit: "cover"
  }} />
        <div style={{
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    padding: "2rem 1.5rem 1.25rem",
    background: "linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.85) 100%)"
  }}>
          <p style={{
    margin: 0,
    color: "rgba(255,255,255,0.88)",
    fontFamily: "monospace",
    fontSize: "0.82rem",
    lineHeight: 1.65,
    maxWidth: "48rem"
  }}>
            {active.prompt}
          </p>
        </div>
      </div>
      <div style={{
    display: "grid",
    gridTemplateColumns: "repeat(5, 1fr)",
    gap: "0"
  }}>
        {categories.map(cat => {
    const isActive = cat.id === activeId;
    return <button key={cat.id} onClick={() => setActiveId(cat.id)} style={{
      display: "block",
      width: "100%",
      padding: "1rem 1.25rem",
      border: "none",
      borderRight: "1px solid rgba(255,255,255,0.08)",
      background: isActive ? "rgba(255,255,255,0.06)" : "transparent",
      textAlign: "left",
      cursor: "pointer",
      color: "#f5f5f5",
      borderTop: isActive ? "2px solid rgba(255,255,255,0.5)" : "2px solid transparent"
    }}>
              <div style={{
      fontWeight: 600,
      fontSize: "0.9rem",
      color: isActive ? "#fff" : "rgba(255,255,255,0.6)"
    }}>
                {cat.title}
              </div>
            </button>;
  })}
      </div>
    </div>;
};

FLUX.2 brings powerful image editing capabilities across the entire model family. Describe what you want changed in natural language — swap backgrounds, replace objects, transfer styles, adjust lighting — and FLUX.2 makes it happen while maintaining photorealism.

All **FLUX.2** variants support multi-reference image editing, allowing you to combine elements from multiple source images into a single coherent result.

<EditingShowcase />

## Reference Images per Model

| Model                  | Reference Images (API) | Reference Images (Playground) |
| ---------------------- | ---------------------- | ----------------------------- |
| **FLUX.2 \[max]**      | Up to **8**            | Up to **10**                  |
| **FLUX.2 \[pro]**      | Up to **8**            | Up to **10**                  |
| **FLUX.2 \[flex]**     | Up to **8**            | Up to **10**                  |
| **FLUX.2 \[klein] 9B** | Up to **4**            | —                             |
| **FLUX.2 \[klein] 4B** | Up to **4**            | —                             |
| **FLUX.2 \[dev]**      | Recommended max **6**  | —                             |

<Tip>
  More reference images means more control. Use multiple inputs to maintain character consistency, combine furniture from different photos, or transfer styles — all in a single generation.
</Tip>

## Single Editing Example

<ImageComparisonSlider beforeImage="https://cdn.sanity.io/images/2gpum2i6/production/ca563be7241d46f06868a7aae876098ce8520588-1440x960.png" afterImage="https://cdn.sanity.io/images/2gpum2i6/production/b200442e944e6f1f58a91443dd82e64f8eb753c4-1549x1033.jpg" beforeLabel="Original" afterLabel="Edited" height="500px" />

<PromptDisplay prompt="Change it to Night" />

## Multi-Reference Example

<MultiRefGrid
  inputs={[
{ src: "https://cdn.sanity.io/images/2gpum2i6/production/669126258fdc53965be6d8168180b298755d5db1-1125x750.png", label: "Ice Skates" },
{ src: "https://cdn.sanity.io/images/2gpum2i6/production/5c022629eb46a126547ea685a5bedb145288875c-1125x750.png", label: "Location" },
{ src: "https://cdn.sanity.io/images/2gpum2i6/production/c00b05340b6dc9807e492efa7cd85277d4fc7201-500x331.png", label: "Decorations" },
]}
  result={{ src: "https://cdn.sanity.io/images/2gpum2i6/production/31acfcff514f52c13abb3a7327d4d74aeb35103c-1120x736.png" }}
  prompt="Create a vintage image taken with a Kodak camera, with heavy grain and slight light smudges. Use Image 2 as the location. Insert only the ice skates from Image 1 into Image 2, with the decorations and evening lighting vibe from Image 3. Add more people skating on the ice."
/>

<CardGroup cols={2}>
  <Card title="FLUX.2 Editing" icon="wand-magic-sparkles" href="https://docs.bfl.ml/flux_2/flux2_image_editing">
    Full editing documentation with API examples, parameters, and multi-reference techniques.
  </Card>

  <Card title="Editing Prompting Guide" icon="pen-to-square" href="https://docs.bfl.ml/guides/prompting_editing_single_reference">
    Learn how to write effective prompts for image editing workflows.
  </Card>
</CardGroup>
