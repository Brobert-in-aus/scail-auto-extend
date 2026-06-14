# SCAIL Auto Extend

A ComfyUI custom node that generates SCAIL-2 videos of any length in a single queue. It automatically splits generation into chunks (the model's limit is 81 frames), anchors each chunk on the last frames of the previous one, color-matches to prevent drift, and stitches the result — no manual extension sections, bypassing, or frame math.

## Examples

https://github.com/user-attachments/assets/ee4ea6c3-a1ca-47ce-9fe0-537cb69b431f

https://github.com/user-attachments/assets/c3a18739-da55-408e-8903-51757cc6530e

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
- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) (background removal on the reference image — V2 workflow)

> **RMBG is optional and swappable.** Removing the reference image's background and padding it to the video's aspect ratio helps the model produce cleaner replacements, but you can replace the RMBG node with any background-removal node you prefer, feed a reference image whose background is already removed, or skip background removal entirely and see how it goes.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/Brobert-in-aus/scail-auto-extend
```

Restart ComfyUI. The nodes appear as **SCAIL Auto Extend Sampler** (`sampling/video`), plus **SCAIL-2 Identity Tracker** and **SCAIL-2 Identity Seeder** (`conditioning/video_models/scail`) for multi-person work.

## Usage

Load the included workflow: [`SCAIL Auto Extend.json`](https://github.com/Brobert-in-aus/scail-auto-extend/raw/main/SCAIL%20Auto%20Extend.json) (direct download — right-click → Save As, or drag the file into ComfyUI) — input video + reference image in, finished video out.

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

Two helper nodes handle replacing several people in the driving video, each with a different character from a single composited reference image:

- **SCAIL-2 Identity Tracker** (`conditioning/video_models/scail`) — an interactive canvas. Draw a **box** (default) or point per person on the reference image and the driving video; it outputs `ref_track_data` + `driving_track_data` for **Create SCAIL-2 Colored Mask**. Boxes are markedly more reliable than points.
- **SCAIL-2 Identity Seeder** — a headless variant that takes point/box coordinates as inputs and outputs per-person masks for `SAM3_VideoTrack`'s `initial_mask`.

The [`SCAIL Auto Extend V2.json`](https://github.com/Brobert-in-aus/scail-auto-extend/raw/main/SCAIL%20Auto%20Extend%20V2.json) workflow wires the Identity Tracker end to end.

### Workflow (Identity Tracker)

1. Feed the **processed** reference image (background removed + padded to the video's aspect ratio) into `reference_image`, the resized pose video into `pose_video`, a SAM3 model into `sam3_model`, and optionally a `CLIPTextEncode("person")` into `detect_conditioning` (to auto-detect people who enter later).
2. **Prepare the canvas (partial execution).** Press the node's **▶ play button** — this runs *Queue Selected Output Nodes*, executing the graph **only up to this node**. With no boxes drawn yet it just renders the reference and driving frames onto the canvas, without running the sampler. (This is why background removal/padding must sit upstream: the canvas then shows the exact pixels the model will see, so your masks line up.)
3. Draw a **box per person** on the **Reference** tab, then switch to **Driving** and box each person there. Right-click a box to delete; use Undo/Clear as needed.
4. On **Create SCAIL-2 Colored Mask**, set **`sort_by = left_to_right`** (see below).
5. Queue the workflow normally.

### How identity mapping works (important)

SCAIL-2 assigns reference characters to driving people **by spatial position, not by mask colour** — the reference is a single composited frame and the model routes position-first. So:

- **Control who-becomes-whom by ordering the reference composite left-to-right** to match the driving people. Want a character on the middle person? Put them in the middle of the reference lineup.
- The coloured masks' real job is **temporal consistency** — pinning each identity as people move, cross, or occlude. `sort_by = left_to_right` colours both sides by horizontal position so the two signals agree.

### Limitations

- **Max 6 identities.** The model was trained on a fixed 6-colour palette; a 7th wraps and collides.
- **Constant subject count.** The model expects the driving people present to correspond to the reference. A clip where people **enter or leave mid-shot** (e.g. starts with one person, two walk in) produces artifacts — the model tries to realise all reference identities from the start, cramming/hallucinating. Work around it by splitting the clip at the entrance/exit and rendering each segment with a reference containing only the people present, then concatenating.

## License

MIT
