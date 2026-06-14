# SCAIL Auto Extend

A ComfyUI custom node that generates SCAIL-2 videos of any length in a single queue. It automatically splits generation into chunks (the model's limit is 81 frames), anchors each chunk on the last frames of the previous one, color-matches to prevent drift, and stitches the result — no manual extension sections, bypassing, or frame math.

## Examples

📥 **[Download the example workflow — `SCAIL Auto Extend V3.json`](https://github.com/Brobert-in-aus/scail-auto-extend/raw/main/SCAIL%20Auto%20Extend%20V3.json)** (right-click → Save As, or drag it into ComfyUI).

https://github.com/user-attachments/assets/ee4ea6c3-a1ca-47ce-9fe0-537cb69b431f

https://github.com/user-attachments/assets/c3a18739-da55-408e-8903-51757cc6530e

https://github.com/user-attachments/assets/59d71f7b-8892-4457-87cd-9a94043ba287

## What it does

SCAIL-2 generates at most 81 frames per pass. Longer videos require extension passes where the first 5 frames are anchored to the last 5 frames of the previous chunk, so each extension only contributes 76 new frames — and the final chunk length depends on the input video. Doing this by hand means duplicated node sections, manual bypassing, and recalculating the last chunk for every video.

The **SCAIL Auto Extend Sampler** node does the whole loop internally at runtime:

1. Reads the pose video length (controlled as usual by your video loader's force_rate / skip / frame cap), trimmed to the nearest 4n+1 frames.
2. Plans the chunks: 81, then 81 (76 new + 5 overlap) repeating, with an automatically sized final chunk. E.g. 197 frames → 81 + 81 + 45.
3. For each chunk: builds the SCAIL-2 conditioning, samples, decodes, and (optionally) Reinhard-LAB color-matches the new frames to the last frame of the previous chunk.
4. Stitches everything and outputs the full image batch plus a frame count.

It calls ComfyUI's own `WanSCAILToVideo`, `SamplerCustom`, and `ColorTransfer` implementations internally, so output is identical to the equivalent hand-built chain.

## Requirements

- ComfyUI with the SCAIL-2 nodes (merged June 2026 — update to a recent ComfyUI)
- SCAIL-2 models: https://huggingface.co/Comfy-Org/SCAIL-2/tree/main/diffusion_models

The bundled example workflows additionally use:

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (video load/combine)
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (resize, Set/Get, model loader)
- [ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) (person tracking for the colored masks)
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) (GetImageSize+)
- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) (background removal on the reference image — V3 workflow)

> **RMBG is optional and swappable.** Removing the reference image's background and padding it to the video's aspect ratio helps the model produce cleaner replacements, but you can replace the RMBG node with any background-removal node you prefer, feed a reference image whose background is already removed, or skip background removal entirely and see how it goes.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/Brobert-in-aus/scail-auto-extend
```

Restart ComfyUI. The nodes appear as **SCAIL Auto Extend Sampler** (`sampling/video`), plus **SCAIL-2 Identity Tracker**, **SCAIL-2 Identity Seeder** and **SCAIL-2 Multi-Reference (experimental)** (`conditioning/video_models/scail`) for multi-person work.

## Usage

Load the included workflow: [`SCAIL Auto Extend V3.json`](https://github.com/Brobert-in-aus/scail-auto-extend/raw/main/SCAIL%20Auto%20Extend%20V3.json) (direct download — right-click → Save As, or drag the file into ComfyUI) — input video + reference image in, finished video out.

Or wire the node into your own workflow in place of your sampler section:

| Input | Connect from |
|---|---|
| model | your model chain (e.g. ModelSamplingSD3) |
| positive / negative | CLIPTextEncode |
| vae | VAELoader |
| sampler / sigmas | KSamplerSelect / BasicScheduler |
| pose_video | your (resized) driving video |
| pose_video_mask | SCAIL2ColoredMask → pose_video_mask |
| reference_image | reference image |
| reference_image_mask | SCAIL2ColoredMask → reference_image_mask |
| clip_vision_output | CLIPVisionEncode |
| width / height | generation resolution |

Outputs: `images` (stitched batch → VHS_VideoCombine) and `frame_count`.

### Options

| Option | Default | Description |
|---|---|---|
| chunk_length | 81 | Max frames per chunk (model limit). Must be 4n+1. |
| overlap | 5 | Anchor frames carried between chunks. SCAIL-2 was trained at 5. |
| seed_mode | increment | `increment`: chunk i uses noise_seed + i. `fixed`: same seed every chunk. |
| color_transfer | true | Reinhard-LAB match of each chunk to the previous chunk's last frame (fights color drift). |
| pose_strength / pose_start / pose_end | 1.0 / 0.0 / 1.0 | Pose conditioning strength and active step range. |
| replacement_mode | true | SCAIL-2 replacement vs animation mode (must match your mask setup). |

### Notes

- Total length is driven by however many frames reach `pose_video` — cap or trim at your video loader. Input is trimmed to the nearest 4n+1 frames (loses at most 3).
- Progress is reported per chunk; the console logs the chunk plan, e.g. `[SCAIL Auto Extend] 197 pose frames -> 197 output frames, 3 chunk(s): [81, 81, 45]`.
- Interrupting cancels cleanly between/during chunks.

## Multi-person replacement

Replace several people in the driving video, each with a different character from a single composited reference image. Three helper nodes (`conditioning/video_models/scail`):

- **SCAIL-2 Identity Tracker** — an interactive canvas that outputs `ref_track_data` + `driving_track_data` for **Create SCAIL-2 Colored Mask**. It's an output node, so it gets a ▶ play button (step 2 below). Per side (Reference / Driving tabs):
  - **Box** mode (default, most reliable): drag a box per person; boxes are selectable, draggable and resizable.
  - **Point** mode: click = new identity, **Shift+click** adds a positive point, **Alt+click** a negative one. Use 2+ points to capture a whole person — a single click tends to grab only the part under it (a shirt, a face).
  - **Right-click** removes a box / a single point (or the identity if it was its last point); **Delete** removes the selected one.
  - Optional **text detection**: wire a `CLIPTextEncode` ("person") into `reference_conditioning` and/or `driving_conditioning`. With `auto_detect` on, text *adds* identities beyond your boxes — up to 6 on the reference, up to the reference box count on the driving side (caps are ceilings, not quotas; it only adds people it actually detects). With no boxes at all, this reproduces the V1 text-only flow.
  - With `auto_detect` **off** and fewer reference than driving subjects, a warning appears below the preview (some driving people would have no reference to map to).
- **SCAIL-2 Identity Seeder** — a headless variant that takes point/box coordinates and outputs per-person masks for `SAM3_VideoTrack`'s `initial_mask`.
- **SCAIL-2 Multi-Reference (experimental)** — feeds each identity as a separate reference frame. It binds by colour (position-independent) but the model blends appearances, so it's not recommended for quality — see [Findings](#findings).

The [`SCAIL Auto Extend V3.json`](https://github.com/Brobert-in-aus/scail-auto-extend/raw/main/SCAIL%20Auto%20Extend%20V3.json) workflow wires the Identity Tracker end to end.

### Workflow (Identity Tracker)

1. Feed the **processed** reference image (background removed + padded to the video's aspect ratio) into `reference_image`, the resized pose video into `pose_video`, a SAM3 model into `sam3_model`. Optionally wire `CLIPTextEncode` into `reference_conditioning` / `driving_conditioning` for text detection.
2. **Prepare the canvas (partial execution).** Press the node's **▶ play button** — *Queue Selected Output Nodes* runs the graph **only up to this node**, rendering the reference and driving frames onto the canvas without running the sampler. (Background removal / padding must sit upstream so the canvas shows the exact pixels the model will see, and your masks line up.)
3. Mark each person on the **Reference** tab, then the **Driving** tab — boxes (recommended) or multi-point identities. Marker order = colour order.
4. On **Create SCAIL-2 Colored Mask**, set **`sort_by = left_to_right`** so both sides colour by position (see [Findings](#findings) for why).
5. Queue the workflow normally.

See [Findings](#findings) for how identity mapping actually behaves and where it breaks.

## License

MIT

## Findings

Notes from working out how SCAIL-2 handles multi-identity replacement, recorded so they aren't re-discovered the hard way.

- **Routing is position-first, not colour-first.** With a single composited reference frame, the model assigns reference characters to driving people by **spatial position**; the colour mask's real job is *temporal consistency* (pinning identities frame to frame), not initial assignment. So control who-becomes-whom by **ordering the reference composite left-to-right** to match the driving people, and set `SCAIL2ColoredMask sort_by = left_to_right` so both sides colour by position and the two signals agree. (Mechanism: the reference is concatenated as one extra frame, the colour mask is an additive signal that loses to RoPE positional encoding, and only one reference latent is ever used.)
- **You can't force colour over position.** Rearranging the reference to break the spatial correspondence (e.g. stacking characters vertically instead of in a row) does **not** override position-first routing — tested, no effect.
- **Multi-reference binds by colour but blends appearance.** Feeding each identity as its own reference frame (the experimental Multi-Reference node) *does* make binding colour-driven and position-independent — but the model can't keep the appearances apart and blends them across characters. Correct routing, degraded fidelity; matches the model card's "not optimized for multi-reference." Kept as a documented dead end.
- **Max 6 identities.** The model was trained on a fixed 6-colour palette; a 7th wraps and collides.
- **Constant subject count.** The model expects the driving people present to correspond to the reference. People **entering or leaving mid-shot** produce artifacts — it tries to realise all reference identities from the start, cramming/hallucinating. Split the clip at the entrance/exit and render each segment with a reference of only the people present, then concatenate.
- **Crossings & occlusion are the weak point.** Identity is held over time only by the per-frame tracked colour mask (the reference is static), so when people cross or one passes in front of another it can break in two places: the model's position-first routing momentarily points each person at the *other's* reference region (swap / bleed / flicker), and SAM3's tracker can swap the colour assignment as the masks overlap. Footage where people stay in their lanes is far more reliable. To tell which layer failed, preview `pose_video_mask` through the crossing — swapped colours there mean the tracker; a clean mask but a swapped output means the model.
- **Resolution must be divisible by 32.** Width and height should be multiples of **32**, not just 16. The pose/mask conditioning runs at half resolution, so its latent is `dim/16`; with the model's patch size of 2 that has to be even → `dim` divisible by 32. Out-of-spec dimensions get **circular**-padded (`common_dit.pad_to_patch_size`), which wraps the top edge of the frame onto the bottom — the symptom is the bottom ~8–16 px echoing the top. The trap is a multiple of 16-but-not-32: the main latent is clean but the half-res pose latent is odd, so it only shows up at some resolutions. Set your resize node's `divisible_by` to 32 (the bundled workflow does).
- **Framerate.** SCAIL-2 is a Wan-2.1 model trained at 16 fps. For smoother output, generate at 16 and **interpolate** (e.g. FILM VFI), setting Video Combine's `frame_rate` to the interpolated rate — cheaper than, and truer to the model's cadence than, raising `force_rate` in the loader (which generates proportionally more frames and pushes the model off 16 fps).
