"""SCAIL Auto Extend — single node that generates a full-length SCAIL-2 video
by looping chunks internally (81 frames, then 76-new/5-overlap extensions),
replacing the manually-bypassed extension sections of the SCAIL Extend workflow.

Wraps the core WanSCAILToVideo / SamplerCustom / ColorTransfer node logic.
"""

import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths


def _plan_chunks(n_frames, chunk_len, overlap):
    """Trim n_frames to 4n+1, return list of chunk lengths (all 4n+1).
    Coverage = lengths[0] + sum(L - overlap for L in lengths[1:]) == n_eff."""
    n_eff = ((n_frames - 1) // 4) * 4 + 1
    if n_eff <= chunk_len:
        return n_eff, [n_eff]
    step = chunk_len - overlap
    k = math.ceil((n_eff - chunk_len) / step)
    final_len = n_eff - step * k
    return n_eff, [chunk_len] * k + [final_len]


class SCAILAutoExtend:
    DESCRIPTION = (
        "Generates the full video in one go: samples the first chunk, then "
        "automatically loops as many extension chunks as the pose video needs, "
        "anchoring each on the last frames of the previous chunk, and stitches "
        "the result. Replaces the manual extension sections."
    )
    CATEGORY = "sampling/video"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "pose_video": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 896, "min": 32, "max": 8192, "step": 32}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                       "control_after_generate": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "chunk_length": ("INT", {"default": 81, "min": 9, "max": 1024, "step": 4,
                                         "tooltip": "Max frames per chunk (model limit 81). Must be 4n+1."}),
                "overlap": ("INT", {"default": 5, "min": 1, "max": 81, "step": 4,
                                    "tooltip": "Frames from the previous chunk used as anchor. SCAIL-2 trained at 5."}),
                "seed_mode": (["increment", "fixed"], {"default": "increment",
                              "tooltip": "increment: chunk i uses noise_seed+i. fixed: same seed every chunk."}),
                "color_transfer": ("BOOLEAN", {"default": True,
                                   "tooltip": "Reinhard LAB color match of each extension chunk to the last frame of the previous chunk (fights drift)."}),
            },
            "optional": {
                "pose_video_mask": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "replacement_mode": ("BOOLEAN", {"default": True}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_noise": ("BOOLEAN", {"default": True}),
            },
        }

    def generate(self, model, positive, negative, vae, sampler, sigmas, pose_video,
                 width, height, noise_seed, cfg, chunk_length, overlap, seed_mode,
                 color_transfer, pose_video_mask=None, reference_image=None,
                 reference_image_mask=None, clip_vision_output=None,
                 replacement_mode=True, pose_strength=1.0, pose_start=0.0,
                 pose_end=1.0, add_noise=True):
        # imported here so a missing/changed core module gives a clear error at run time
        from comfy_extras.nodes_scail import WanSCAILToVideo
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        from comfy_extras.nodes_post_processing import ColorTransfer

        chunk_length = ((chunk_length - 1) // 4) * 4 + 1
        if overlap % 4 != 1:
            overlap = max(1, ((overlap - 1) // 4) * 4 + 1)
        if chunk_length - overlap < 4:
            raise ValueError(f"chunk_length ({chunk_length}) must exceed overlap "
                             f"({overlap}) by at least 4.")

        n_input = pose_video.shape[0]
        n_eff, lengths = _plan_chunks(n_input, chunk_length, overlap)
        print(f"[SCAIL Auto Extend] {n_input} pose frames -> {n_eff} output frames, "
              f"{len(lengths)} chunk(s): {lengths}")

        pbar = comfy.utils.ProgressBar(len(lengths))
        chunks = []          # stitched contributions
        prev_frames = None   # full frames of previous chunk's contribution
        offset = 0

        for i, length in enumerate(lengths):
            comfy.model_management.throw_exception_if_processing_interrupted()
            seed = noise_seed + i if seed_mode == "increment" else noise_seed

            cond = WanSCAILToVideo.execute(
                positive=positive, negative=negative, vae=vae,
                width=width, height=height, length=length, batch_size=1,
                pose_strength=pose_strength, pose_start=pose_start, pose_end=pose_end,
                video_frame_offset=offset, previous_frame_count=overlap,
                replacement_mode=replacement_mode,
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pose_video, pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=prev_frames,
            )
            pos_c, neg_c, latent, offset = cond.args

            sampled = SamplerCustom.execute(
                model=model, add_noise=add_noise, noise_seed=seed, cfg=cfg,
                positive=pos_c, negative=neg_c, sampler=sampler, sigmas=sigmas,
                latent_image=latent,
            )
            denoised = sampled.args[1]  # denoised_output

            images = vae.decode(denoised["samples"])
            if images.ndim == 5:
                images = images.reshape(-1, *images.shape[-3:])

            if i == 0:
                contrib = images
            else:
                contrib = images[overlap:]
                if color_transfer and prev_frames is not None:
                    contrib = ColorTransfer.execute(
                        image_target=contrib,
                        image_ref=prev_frames[-1:],
                        method="reinhard_lab",
                        source_stats={"source_stats": "per_frame"},
                        strength=1.0,
                    ).args[0]

            chunks.append(contrib)
            prev_frames = contrib
            pbar.update(1)
            print(f"[SCAIL Auto Extend] chunk {i + 1}/{len(lengths)} done "
                  f"({length} frames, offset now {offset})")

        out = torch.cat([c.to(chunks[0].device, dtype=chunks[0].dtype) for c in chunks], dim=0)
        return (out, out.shape[0])


class SCAIL2IdentitySeeder:
    """Produce one binary mask per person from explicit point or box prompts, so
    each subject becomes a distinct tracked object (and thus a distinct colour) in
    SCAIL-2 multi-person workflows.

    Why this exists: SAM3_VideoTrack's auto-detection runs mask NMS using an
    IoU+IoM overlap test at a fixed 0.5 threshold, which collapses close or
    overlapping people into a single object — and its object roster is seeded from
    the first frame, so late-appearing people fail the same overlap gate. Neither
    constant is exposed, so detection_threshold can't fix it. Feeding explicit
    per-object masks into SAM3_VideoTrack's `initial_mask` (with conditioning left
    disconnected) bypasses detection entirely: the tracker propagates exactly the
    masks you seed, one colour each.

    Output: MASK of shape [N_people, H, W]. Wire into SAM3_VideoTrack.initial_mask.
    Feed `image` at the SAME resolution the tracker will run on (the resized
    reference image, or the resized pose video's first frame).
    """

    DESCRIPTION = (
        "One mask per person from point or box prompts -> SAM3_VideoTrack.initial_mask. "
        "Guarantees one tracked object (one colour) per subject, bypassing the "
        "auto-detector's overlap-merging. Leave SAM3_VideoTrack conditioning disconnected."
    )
    CATEGORY = "conditioning/video_models/scail"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("initial_masks",)
    FUNCTION = "seed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SAM3 model (same one feeding SAM3_VideoTrack)."}),
                "image": ("IMAGE", {"tooltip": "Frame to segment, at the resolution the tracker runs on (resized reference image, or pose video's first frame)."}),
                "mode": (["points", "boxes"], {"default": "points",
                          "tooltip": "points: one positive click per person (from a PointsEditor). boxes: one bounding box per person."}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5,
                                      "tooltip": "SAM decoder refinement passes per object (0 = raw prompt mask)."}),
            },
            "optional": {
                "points": ("STRING", {"default": "", "multiline": True,
                            "tooltip": "points mode: JSON list, one positive point per person, e.g. [{\"x\":120,\"y\":210},{\"x\":480,\"y\":205}]. Order sets identity order."}),
                "bboxes": ("BOUNDING_BOX", {"tooltip": "boxes mode: one bounding box per person. Order sets identity order."}),
            },
        }

    def seed(self, model, image, mode, refine_iterations, points="", bboxes=None):
        B, H, W, C = image.shape
        image_in = comfy.utils.common_upscale(
            image[:1, ..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model
        frame = image_in.to(device=device, dtype=dtype)

        def _refine(mask_logit):
            for _ in range(max(0, refine_iterations - 1)):
                mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
            mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
            return (mask[0] > 0).float()  # [1, H, W]

        masks = []
        if mode == "points":
            pts = json.loads(points) if points.strip() else []
            if not pts:
                raise ValueError("SCAIL-2 Identity Seeder (points mode): provide at least one point "
                                 "(one per person) in `points`.")
            for p in pts:
                coords = torch.tensor([[[p["x"] / W * 1008, p["y"] / H * 1008]]], dtype=dtype, device=device)
                labels = torch.ones((1, 1), dtype=torch.int32, device=device)
                mask_logit = sam3_model.forward_segment(
                    frame, point_inputs={"point_coords": coords, "point_labels": labels})
                masks.append(_refine(mask_logit))
        else:  # boxes
            box_list = bboxes if isinstance(bboxes, list) else ([bboxes] if bboxes else [])
            if not box_list:
                raise ValueError("SCAIL-2 Identity Seeder (boxes mode): provide one bounding box "
                                 "per person in `bboxes`.")
            for d in box_list:
                x1 = d["x"] / W * 1008
                y1 = d["y"] / H * 1008
                x2 = (d["x"] + d["width"]) / W * 1008
                y2 = (d["y"] + d["height"]) / H * 1008
                sam_box = torch.tensor([[[x1, y1], [x2, y2]]], device=device, dtype=dtype)
                mask_logit = sam3_model.forward_segment(frame, box_inputs=sam_box)
                masks.append(_refine(mask_logit))

        out = torch.cat(masks, dim=0).to(comfy.model_management.intermediate_device())  # [N_people, H, W]
        return (out,)


def _sam3_segment(sam3, image, markers, refine_iterations, device, dtype):
    """Run SAM3 segment per marker on the first frame of `image` (B,H,W,C).
    markers: list of {"type":"point","x","y"} or {"type":"box","x","y","w","h"}
    in `image` pixel coords. Returns [N, H, W] float masks at native H,W, or None."""
    if not markers:
        return None
    B, H, W, C = image.shape
    frame = comfy.utils.common_upscale(
        image[:1, ..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)

    def _refine(mask_logit):
        for _ in range(max(0, refine_iterations - 1)):
            mask_logit = sam3.forward_segment(frame, mask_inputs=mask_logit)
        mask = F.interpolate(mask_logit, size=(H, W), mode="bilinear", align_corners=False)
        return (mask[0] > 0).float()  # [1, H, W]

    masks = []
    for m in markers:
        if m.get("type") == "box":
            x1 = m["x"] / W * 1008
            y1 = m["y"] / H * 1008
            x2 = (m["x"] + m["w"]) / W * 1008
            y2 = (m["y"] + m["h"]) / H * 1008
            sam_box = torch.tensor([[[x1, y1], [x2, y2]]], device=device, dtype=dtype)
            mask_logit = sam3.forward_segment(frame, box_inputs=sam_box)
        else:  # point
            coords = torch.tensor([[[m["x"] / W * 1008, m["y"] / H * 1008]]], dtype=dtype, device=device)
            labels = torch.ones((1, 1), dtype=torch.int32, device=device)
            mask_logit = sam3.forward_segment(frame, point_inputs={"point_coords": coords, "point_labels": labels})
        masks.append(_refine(mask_logit))
    return torch.cat(masks, dim=0)  # [N, H, W]


class SCAIL2IdentityTracker:
    """Canvas-driven multi-person seeding + dual SAM3 tracking for SCAIL-2.

    Turns a *processed* reference image and a driving video into the two
    SAM3_TRACK_DATA bundles SCAIL2ColoredMask consumes, with identities seeded by
    points/boxes you draw on an in-node canvas (placement order = colour order).
    Optional auto-detection appends late arrivals on the driving side.

    Because it's an OUTPUT_NODE, ComfyUI shows a play button on it: press it with no
    markers drawn and it only renders the two frames to the canvas (no tracking, no
    sampler) thanks to partial execution. Draw your points/boxes, then run normally.

    Feed the reference image AFTER background-removal/padding so masks match the
    pixels the model sees. Keep SCAIL2ColoredMask sort_by = "none" so the order you
    draw is preserved.
    """

    DESCRIPTION = (
        "Draw ordered points/boxes per person on the reference image and driving "
        "video, get ref_track_data + driving_track_data out. Play button previews "
        "the frames without tracking (partial execution). Auto-detect adds latecomers."
    )
    CATEGORY = "conditioning/video_models/scail"
    RETURN_TYPES = ("SAM3_TRACK_DATA", "SAM3_TRACK_DATA", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ref_track_data", "driving_track_data", "reference_image", "pose_video")
    FUNCTION = "track"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("MODEL", {"tooltip": "SAM3 model (e.g. from CheckpointLoaderSimple)."}),
                "reference_image": ("IMAGE", {"tooltip": "Processed reference (post background-removal + padding), at model resolution."}),
                "pose_video": ("IMAGE", {"tooltip": "Driving/pose video frames, at the resolution fed to the sampler."}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5,
                                      "tooltip": "SAM decoder refinement passes per seed."}),
                "auto_detect": ("BOOLEAN", {"default": True,
                                "tooltip": "Driving side: append late-arriving people via text detection (needs detect_conditioning). Order may differ from seeds."}),
                "detection_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Score threshold for auto-detected latecomers."}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 64,
                                    "tooltip": "Run auto-detection every N frames."}),
                "markers": ("STRING", {"default": "{}", "multiline": True,
                            "tooltip": "Canvas markers (JSON). Managed by the node's canvas widget."}),
            },
            "optional": {
                "detect_conditioning": ("CONDITIONING", {"tooltip": "CLIPTextEncode (e.g. 'person') used only for driving-side auto-detect."}),
            },
        }

    def _empty_track(self, N, H, W):
        return {"packed_masks": None, "orig_size": (H, W), "n_frames": int(N), "scores": []}

    def _track_side(self, sam3, images, seed_masks, text_prompts, device, dtype,
                    detection_threshold, max_objects, detect_interval):
        N, H, W, C = images.shape
        if seed_masks is None and text_prompts is None:
            return self._empty_track(N, H, W)
        frames_in = images[..., :3].movedim(-1, 1)
        init_masks = seed_masks.unsqueeze(1).to(device=device, dtype=dtype) if seed_masks is not None else None
        pbar = comfy.utils.ProgressBar(N)
        result = sam3.forward_video(
            images=frames_in, initial_masks=init_masks, pbar=pbar, text_prompts=text_prompts,
            new_det_thresh=detection_threshold, max_objects=max_objects,
            detect_interval=detect_interval, target_device=device, target_dtype=dtype)
        result["orig_size"] = (H, W)
        return result

    def _save_preview(self, img_hwc, prefix):
        arr = (img_hwc.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr, "RGB")
        fname = f"{prefix}_{random.randint(0, 0xffffffff):08x}.png"
        tdir = folder_paths.get_temp_directory()
        os.makedirs(tdir, exist_ok=True)
        pil.save(os.path.join(tdir, fname), compress_level=4)
        return {"filename": fname, "subfolder": "", "type": "temp"}

    def track(self, sam3_model, reference_image, pose_video, refine_iterations, auto_detect,
              detection_threshold, detect_interval, markers, detect_conditioning=None):
        try:
            data = json.loads(markers) if markers and markers.strip() else {}
        except json.JSONDecodeError:
            data = {}
        ref_markers = data.get("reference", []) or []
        drv_markers = data.get("driving", []) or []

        previews = {
            "reference_preview": [self._save_preview(reference_image[0], "scail_ref")],
            "driving_preview": [self._save_preview(pose_video[0], "scail_drv")],
        }

        # No markers yet -> this is a play-button preview pass. Emit frames, skip tracking.
        if not ref_markers and not drv_markers:
            ref_td = self._empty_track(reference_image.shape[0], reference_image.shape[1], reference_image.shape[2])
            drv_td = self._empty_track(pose_video.shape[0], pose_video.shape[1], pose_video.shape[2])
            return {"ui": previews, "result": (ref_td, drv_td, reference_image, pose_video)}

        comfy.model_management.load_model_gpu(sam3_model)
        device = comfy.model_management.get_torch_device()
        dtype = sam3_model.model.get_dtype()
        sam3 = sam3_model.model.diffusion_model

        ref_seed = _sam3_segment(sam3, reference_image, ref_markers, refine_iterations, device, dtype)
        drv_seed = _sam3_segment(sam3, pose_video, drv_markers, refine_iterations, device, dtype)

        text_prompts = None
        if auto_detect and detect_conditioning is not None and len(detect_conditioning) > 0:
            from comfy_extras.nodes_sam3 import _extract_text_prompts
            text_prompts = [(emb, m) for emb, m, _ in _extract_text_prompts(detect_conditioning, device, dtype)]

        # Reference is a controlled still: seeds only, never auto-detect.
        ref_td = self._track_side(sam3, reference_image, ref_seed, None, device, dtype,
                                  detection_threshold, 0, detect_interval)
        # Driving cap = reference identity count, bounded by the 6-colour palette ceiling
        # (you can't map more distinct identities than you have references). Seeds are never
        # dropped by this; it only limits auto-detected latecomers.
        ref_count = int(ref_seed.shape[0]) if ref_seed is not None else 0
        drv_max = min(ref_count, 6) if ref_count > 0 else 6
        # Driving: seeds + optional auto-detected latecomers.
        drv_td = self._track_side(sam3, pose_video, drv_seed, text_prompts, device, dtype,
                                  detection_threshold, drv_max, detect_interval)

        return {"ui": previews, "result": (ref_td, drv_td, reference_image, pose_video)}


NODE_CLASS_MAPPINGS = {
    "SCAILAutoExtend": SCAILAutoExtend,
    "SCAIL2IdentitySeeder": SCAIL2IdentitySeeder,
    "SCAIL2IdentityTracker": SCAIL2IdentityTracker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILAutoExtend": "SCAIL Auto Extend Sampler",
    "SCAIL2IdentitySeeder": "SCAIL-2 Identity Seeder",
    "SCAIL2IdentityTracker": "SCAIL-2 Identity Tracker",
}

WEB_DIRECTORY = "./web"
