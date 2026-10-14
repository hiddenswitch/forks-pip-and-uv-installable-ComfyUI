# FLUX Negative Prompts

Vendored snapshot from: https://docs.bfl.ml/guides/prompting_guide_t2i_negative

Fetched: 2026-06-03

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.bfl.ml/llms.txt
> Use this file to discover all available pages before exploring further.

# Working Without Negative Prompts

> How to use positive framing instead of negative prompts in FLUX. Replace exclusions with specific visual descriptions.

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

FLUX models don't support negative prompts. Even if they could process negatives, AI models generally struggle with negation. When you write "a person without glasses," the model focuses on the word "glasses" and often generates exactly what you're trying to avoid.

## Turn Negatives Into Positives

When you want to exclude something from your image, resist the urge to write what you don't want. Instead, describe the **positive visual opposite** of what you're trying to avoid.

Ask yourself: **"If this thing wasn't there, what would I see instead?"**

## The Replacement Strategy

When you catch yourself writing negative phrases, use this mental process:

1. **Identify the unwanted element**: "no crowds"
2. **Ask what replaces it**: "What would be there instead?"
3. **Describe the positive**: "peaceful solitude" or "empty spaces"

**Common replacements**:

* "no people" → "empty", "deserted", "solitary"
* "without clothes" → "bare skin", "natural form"
* "no colors" → "monochrome", "black and white", "grayscale"
* "no text" → "clean surfaces", "unmarked", "blank"
* "no modern elements" → "traditional", "historical", "period-accurate"

## Advanced Positive Framing

**For atmosphere**: Instead of "not dark," use "brightly lit" or "sun-drenched"

**For emotions**: Instead of "not sad," use "joyful" or "content"

**For actions**: Instead of "not running," use "walking peacefully" or "standing still"

**For quantities**: Instead of "not many," use "few", "single", or "minimal"

### Practical Examples

<AccordionGroup>
  <Accordion title="Context & Setting">
    1. **Instead of**: "a street with no cars"\
       **Write**: "a quiet pedestrian walkway with cobblestones"

    2. **Instead of**: "a landscape without buildings"\
       **Write**: "pristine wilderness with untouched natural terrain"

    3. **Instead of**: "a room with no furniture"\
       **Write**: "a spacious empty room with polished wooden floors"
  </Accordion>

  <Accordion title="Character & Portrait">
    4. **Instead of**: "a person without a hat"\
       **Write**: "a person with natural hair flowing freely"

    5. **Instead of**: "a portrait with no glasses"\
       **Write**: "a portrait showing clear, unobstructed eyes"
  </Accordion>

  <Accordion title="Objects & Props">
    6. **Instead of**: "a table without food"\
       **Write**: "a clean marble table set with elegant empty plates"
  </Accordion>
</AccordionGroup>

<Tip>
  When building prompts with positive alternatives, maintain the [Subject + Action + Style + Context framework](https://docs.bfl.ml/guides/prompting_guide_t2i_fundamentals#basic-prompt-structure) for best results.
</Tip>

## Advanced Positive Techniques

<AccordionGroup>
  <Accordion title="Compositional Control">
    Instead of describing what shouldn't be in frame, describe what fills the space:

    **Weak**: "portrait with no background distractions"
    **Strong**: "portrait with smooth gradient background transitioning from deep blue to black"
  </Accordion>

  <Accordion title="Style Specifications">
    Instead of avoiding certain artistic elements, specify the desired aesthetic:

    **Weak**: "not too realistic"
    **Strong**: "stylized illustration with simplified forms and bold color blocks"
  </Accordion>

  <Accordion title="Mood and Atmosphere">
    Frame emotional tone positively:

    **Weak**: "not scary or threatening"\
    **Strong**: "peaceful, welcoming, and warm atmosphere with soft golden lighting"
  </Accordion>
</AccordionGroup>

## When Positive Alternatives Don't Work

If you're still getting unwanted elements despite positive framing:

1. **Be more specific** about what you do want in that space
2. **Front-load the positive description** following [word order principles](https://docs.bfl.ml/guides/prompting_guide_t2i_fundamentals#word-order-importance)
3. **Add more descriptive detail** to strengthen the positive alternative
4. **Use environmental context** to make the positive element more natural

**Example progression**:

* Basic: "A beach"
* Enhanced: "Empty beach with palm trees and gentle waves"
* Detailed: "Empty beach with palm trees and gentle waves, golden sunset, IMAX-quality look"

<Columns cols={3}>
  <div>
    <Frame caption="A beach">
      <img src="https://cdn.sanity.io/images/gsvmb6gz/production/4fc2c5a86463f31bca29cb1b4ee502b5d713ccfe-1392x752.jpg" alt="Basic" />
    </Frame>

    <PromptDisplay prompt="A beach" />
  </div>

  <div>
    <Frame caption="Empty beach with palm trees and gentle waves">
      <img src="https://cdn.sanity.io/images/gsvmb6gz/production/ca548a2e74e7372b485c9d9ba72914487d196d60-1392x752.jpg" alt="Enhanced" />
    </Frame>

    <PromptDisplay prompt="Empty beach with palm trees and gentle waves" />
  </div>

  <div>
    <Frame caption="Empty beach with palm trees and gentle waves, golden sunset, IMAX-quality look">
      <img src="https://cdn.sanity.io/images/gsvmb6gz/production/ca91ec2d49372239d56bb13b18a8d50bc1180305-1392x752.jpg" alt="Detailed" />
    </Frame>

    <PromptDisplay prompt="Empty beach with palm trees and gentle waves, golden sunset, IMAX-quality look" />
  </div>
</Columns>

<Tip>
  Think visually about what you want to see, not what to avoid.
</Tip>
