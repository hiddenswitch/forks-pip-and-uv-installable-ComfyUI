# Kandinsky 6.0 Video prompting

Official sources, checked 2026-10-29:

- [Text-to-audio-video prompt expansion](https://github.com/kandinskylab/kandinsky-6/blob/main/comfyui/kandinsky6/beautifier_prompts/t2av_system.txt)
- [Image-to-audio-video prompt expansion](https://github.com/kandinskylab/kandinsky-6/blob/main/comfyui/kandinsky6/beautifier_prompts/i2av_system.txt)
- [Official generation examples](https://github.com/kandinskylab/kandinsky-6/blob/main/docs/quick_start.md#generate)

These are the authors' shipped beautifier instructions, not a separate prose
handbook. Their expansion format is one 180–300-word paragraph in the request's
language: framing, one camera sentence, appearance, chronological five-second
action, lighting and style. Give movement a direction and use visible details.
The word budget belongs to this beautifier format, not a tokenizer limit.

Preserve requested dialogue verbatim inside `<S>…<E>`, identifying the speaker.
Keep ambient sound and music in a final `<AUDCAP>…<ENDAUDCAP>` sentence. Silent
characters should have closed mouths; animal noises are not dialogue.

For image-to-video, treat the reference as the first frame: preserve its visible
objects, colors and lettering. Newly requested subjects enter during the clip.
Avoid inventing depth behind a flat illustration. Read the distinct source for
each task before expanding a prompt.
