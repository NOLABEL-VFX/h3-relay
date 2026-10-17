"""Reusable H3 timeline adapters around native ComfyUI nodes.

Native StartLoop/EndLoop own iteration. Models, reference encoding, guides,
sampling, VAE encoding/decoding and video output stay in the user's graph.
"""
from __future__ import annotations

import json
import logging

import torch
import torch.nn.functional as F

from .video_prep import LabsH3PrepareVideo
from .windowed_edit import HISTORY, WINDOWS, plan_windows, validate_settings
from .person_remover import window_starts

LOG = logging.getLogger(__name__)
CONTRACT = "h3-relay-window-v2"


def pad_frames(tensor, length):
    if len(tensor) < 1:
        raise ValueError("Cannot pad an empty window")
    tensor = tensor[:length]
    if len(tensor) < length:
        tensor = torch.cat((tensor, tensor[-1:].expand(length-len(tensor), *tensor.shape[1:])))
    return tensor


def slice_audio(audio, start, end, pad=True):
    if audio is None:
        return None
    if start < 0 or end < start:
        raise ValueError("Invalid source audio interval")
    rate = int(audio["sample_rate"])
    first, last = round(start*rate/24), round(end*rate/24)
    wave = audio["waveform"][..., first:last]
    if pad and wave.shape[-1] < last-first:
        wave = F.pad(wave, (0, last-first-wave.shape[-1]))
    return {"sample_rate": rate, "waveform": wave}


def plan_segment(source_frames, history_frames=0):
    """An explicit app segment whose source includes its supplied history."""
    if type(source_frames) is not int or type(history_frames) is not int or history_frames not in HISTORY:
        raise ValueError("Invalid segment frame counts")
    target = source_frames-history_frames
    needed = target+bool(history_frames)
    sizes = [v for v in WINDOWS if v >= needed and v > history_frames]
    if target < 1 or not sizes:
        raise ValueError("The explicit segment must fit a single diffusion pass")
    h, size = history_frames, sizes[0]
    return [{"index": 0, "source_start": h-1 if h else 0,
             "source_end_exclusive": source_frames, "window_frames": size,
             "padded_tail_frames": size-needed, "history_frames": h,
             "history_start": 0 if h else None, "history_end_exclusive": h if h else None,
             "discarded_overlap": int(bool(h)), "delivered_frames": target}]


def pool_edit_mask(mask, latent_shape):
    """Union source pixels over H3 temporal spans and 32-pixel DiT cells."""
    _, channels, t, h, w = latent_shape
    spans = [(1, 4, 4, 4, 4)[i % 5] for i in range(t)]
    if channels != 24 or tuple(mask.shape) != (sum(spans), h*16, w*16):
        raise ValueError(f"H3 mask/latent timeline mismatch: {tuple(mask.shape)} vs {latent_shape}")
    if h % 2 or w % 2:
        raise ValueError("H3 edit masks require dimensions divisible by 32")
    if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("Edit masks must contain finite values between zero and one")
    rows, offset = [], 0
    for span in spans:
        rows.append(mask[offset:offset+span].amax(dim=0))
        offset += span
    pooled = F.max_pool2d(torch.stack(rows)[:, None].float(), 32, 32)[:, 0]
    return pooled.repeat_interleave(2, -2).repeat_interleave(2, -1)[None, None]


class H3RelayPrepareSource(LabsH3PrepareVideo):
    SOURCE_EXACT = True
    CATEGORY = "H3 Relay/timeline"

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        spec["required"]["resolution"][1]["tooltip"] = "Preserve aligned sources within the pixel budget; otherwise fit the H3 canvas."
        spec["optional"]["duration_seconds"][1]["tooltip"] = "Keep the requested source duration. Model padding is excluded from the final output. Zero keeps the full source."
        spec["optional"]["missing_audio"] = (["silence", "none"], {"default": "silence"})
        return spec

    def prepare(self, video, resolution, duration_seconds=0., missing_audio="silence"):
        values = list(super().prepare(video, resolution, duration_seconds))
        if values[1] is None and missing_audio == "silence":
            values[1] = {"waveform": torch.zeros(1, 2, round(values[5]/24*48000)), "sample_rate": 48000}
            values[7] += "\nNo source audio: silent conditioning reference."
        return tuple(values)


class H3RelayWindowPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_frames": ("INT", {"default": 120, "min": 1}),
            "window_frames": ([str(v) for v in WINDOWS], {"default": "124"}),
            "history_frames": ([str(v) for v in HISTORY], {"default": "18"}),
            "seed": ("INT", {"default": 194731, "min": 0, "max": 2**64-1}),
            "schedule": (["adaptive", "removal_v1", "segment"], {"default": "adaptive"}),
            "seed_mode": (["fixed", "increment"], {"default": "fixed"}),
        }, "optional": {"seed_overrides": ("STRING", {"default": "{}"}),
                         "initial_history": ("IMAGE",)}}
    RETURN_TYPES = ("H3_RELAY_WINDOW", "STRING")
    RETURN_NAMES = ("windows", "timeline")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "plan"
    CATEGORY = "H3 Relay/timeline"

    def plan(self, source_frames, window_frames, history_frames, seed,
             schedule="adaptive", seed_mode="fixed", seed_overrides="{}", initial_history=None):
        size, history = validate_settings(window_frames, history_frames)
        initial = 0 if initial_history is None else len(initial_history)
        if initial and schedule != "segment":
            raise ValueError("External history requires the explicit segment schedule")
        if schedule == "segment":
            entries = plan_segment(source_frames, initial)
        elif schedule == "removal_v1":
            if history != 18:
                raise ValueError("The removal_v1 schedule requires 18 history frames")
            entries, covered = [], 0
            for i, start in enumerate(window_starts(source_frames, size)):
                end = min(source_frames, start+size)
                h = 18 if i else 0
                entries.append({"index": i, "source_start": start, "source_end_exclusive": end,
                    "window_frames": size, "history_frames": h,
                    "history_start": start-h+1 if h else None,
                    "history_end_exclusive": start+1 if h else None,
                    "padded_tail_frames": size-(end-start), "discarded_overlap": covered-start,
                    "delivered_frames": end-covered})
                covered = end
        else:
            entries = plan_windows(source_frames, size, history)
        overrides = json.loads(seed_overrides)
        if not isinstance(overrides, dict) or any(str(k) not in {str(e["index"]) for e in entries} for k in overrides):
            raise ValueError("Seed overrides must map existing window indices to seeds")
        for entry in entries:
            value = overrides.get(str(entry["index"]), (seed+(entry["index"] if seed_mode == "increment" else 0)) % 2**64)
            if type(value) is not int or not 0 <= value < 2**64:
                raise ValueError("Window seed must be an unsigned 64-bit integer")
            entry.update(contract=CONTRACT, source_frames=source_frames, seed=value,
                         initial_history_frames=initial, schedule=schedule)
        return entries, json.dumps({"contract": CONTRACT, "fps": 24, "windows": entries}, indent=2)


class H3RelayWindowInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",), "window": ("H3_RELAY_WINDOW",),
            "reference_history": ("BOOLEAN", {"default": True}),
            "history_audio_mode": (["source", "generated", "none"], {"default": "source"})},
            "optional": {"previous": ("H3_RELAY_WINDOW_STATE",), "source_audio": ("AUDIO",),
                "mask": ("MASK",), "anchor": ("IMAGE",), "initial_history": ("IMAGE",),
                "initial_history_audio": ("AUDIO",)}}
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "IMAGE", "AUDIO", "IMAGE", "AUDIO", "INT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("source_window", "edit_mask", "target_audio", "history", "history_audio",
                    "reference_video", "reference_audio", "history_frames", "conditioning_length",
                    "target_frames", "seed", "anchor")
    FUNCTION = "select"
    CATEGORY = "H3 Relay/timeline"

    def select(self, source, window, reference_history=True, history_audio_mode="source",
               previous=None, source_audio=None, mask=None, anchor=None, initial_history=None,
               initial_history_audio=None):
        if window.get("contract") != CONTRACT:
            raise ValueError("Unsupported H3 window contract")
        start, size, hist = (window[k] for k in ("source_start", "window_frames", "history_frames"))
        count = window["source_frames"]
        if not 1 <= count <= len(source) or not 0 <= start < count:
            raise ValueError("Source does not cover the planned timeline")
        if source.shape[-1] != 3 or source.shape[1] % 32 or source.shape[2] % 32:
            raise ValueError("H3 source must be RGB with dimensions divisible by 32")
        if mask is not None and (len(mask) < count or mask.shape[1:] != source.shape[1:3]):
            raise ValueError("Mask must cover the source at matching dimensions")
        if anchor is None:
            anchor = source[:1]
        else:
            import comfy.utils
            anchor = comfy.utils.common_upscale(anchor[:1].movedim(-1, 1), source.shape[2], source.shape[1],
                                                "lanczos", "disabled").movedim(1, -1)
        frames = previous["frames"] if previous is not None else initial_history
        audio = previous.get("audio") if previous is not None else initial_history_audio
        prior, prior_audio = anchor, None
        if hist:
            if frames is None or start < hist-1 or start >= len(frames):
                raise ValueError("Missing generated history at the exact source boundary")
            prior = frames[start-hist+1:start+1]
            if prior.shape[1:] != source.shape[1:]:
                raise ValueError("Generated history does not match the source canvas")
            anchor = prior[-1:]
            selected_audio = source_audio if history_audio_mode == "source" else audio
            if history_audio_mode != "none":
                prior_audio = slice_audio(selected_audio, start-hist+1, start+1)
        past = max(0, hist-1) if reference_history else 0
        begin, end = start-past, min(start+size, count)
        if begin < 0:
            raise ValueError("Reference history precedes the source")
        edit_mask = pad_frames(mask[start:end], size) if mask is not None else None
        return (pad_frames(source[start:end], size), edit_mask, slice_audio(source_audio, start, start+size),
                prior, prior_audio, pad_frames(source[begin:end], size+past),
                slice_audio(source_audio, begin, start+size), hist, size+max(0, hist-1),
                size, window["seed"], anchor)


class H3RelayApplyHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "latent": ("LATENT",),
            "vae": ("VAE",), "audio_vae": ("VAE",), "frames": ("IMAGE",), "history_frames": ("INT",),
            "first_frame_guide": ("BOOLEAN", {"default": True}),
            "target_audio_guide": ("BOOLEAN", {"default": True})},
            "optional": {"audio": ("AUDIO",)}}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/conditioning"

    def apply(self, positive, latent, vae, audio_vae, frames, history_frames,
              first_frame_guide=True, target_audio_guide=True, audio=None):
        if history_frames not in HISTORY:
            raise ValueError("History must be 0, 1, or 17n+1 frames")
        if history_frames >= 18:
            from .vendor.context_loop.sliding_context import apply_sliding_history
            if audio is not None:
                import comfy.audio
                rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
                audio = {"waveform": comfy.audio.resample(audio["waveform"], audio["sample_rate"], rate), "sample_rate": rate}
            positive, latent, trim = apply_sliding_history(positive, vae, latent, frames,
                history_frames, "disabled", audio_vae=audio_vae, previous_audio=audio)
            if trim != 1:
                raise RuntimeError("H3 continuation boundary contract changed")
            if target_audio_guide:
                clean = []
                for embedding, metadata in positive:
                    copied = dict(metadata)
                    copied["minimax_keyframes"] = [k for k in copied.get("minimax_keyframes", [])
                        if not (k.get("anchor") == "first" and "audio_latent" in k)]
                    clean.append([embedding, copied])
                positive = clean
        elif history_frames or first_frame_guide:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, vae=vae,
                                               image=frames[-1:], frame_idx=0)[0]
        return positive, latent


class H3RelayApplyEditMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "source_latent": ("LATENT",), "mask": ("MASK",)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/conditioning"

    def apply(self, latent, source_latent, mask):
        from comfy.nested_tensor import NestedTensor
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2:
            raise ValueError("Expected the native H3 video/audio latent pair")
        video, audio = samples.tensors
        source = source_latent["samples"]
        if source.shape != video.shape:
            raise ValueError("Source VAE shape does not match the H3 target")
        video_mask = pool_edit_mask(mask, video.shape)
        return ({**latent, "samples": NestedTensor([source, audio]),
                 "noise_mask": NestedTensor([video_mask, torch.ones_like(audio)])},)


class H3RelayAppendWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "window": ("H3_RELAY_WINDOW",)},
                "optional": {"previous": ("H3_RELAY_WINDOW_STATE",), "audio": ("AUDIO",),
                             "initial_history": ("IMAGE",), "initial_history_audio": ("AUDIO",)}}
    RETURN_TYPES = ("H3_RELAY_WINDOW_STATE",)
    FUNCTION = "append"
    CATEGORY = "H3 Relay/timeline"

    def append(self, frames, window, previous=None, audio=None, initial_history=None, initial_history_audio=None):
        start, end, size = (window[k] for k in ("source_start", "source_end_exclusive", "window_frames"))
        prefix = previous["frames"] if previous is not None else initial_history
        retained = len(prefix) if prefix is not None else 0
        overlap = window["discarded_overlap"]
        if retained != start+overlap or not 0 <= overlap < size:
            raise ValueError("Window assembly would create a gap or duplicate")
        if len(frames) == size-1:
            LOG.warning("H3 decoded one frame short; holding its last frame to preserve timing")
            frames = torch.cat((frames, frames[-1:]))
        if len(frames) != size:
            raise ValueError(f"Expected {size} decoded frames, received {len(frames)}")
        frames = frames[overlap:end-start].cpu()
        if prefix is not None:
            frames = torch.cat((prefix.cpu(), frames))
        assembled_audio = None
        if audio is not None:
            rate = int(audio["sample_rate"])
            origin = round(start*rate/24)
            n = round((start+size)*rate/24)-origin
            wave = audio["waveform"][..., :n].cpu()
            wave = F.pad(wave, (0, max(0, n-wave.shape[-1])))
            wave = wave[..., round((start+overlap)*rate/24)-origin:round(end*rate/24)-origin]
            prior_audio = previous.get("audio") if previous is not None else initial_history_audio
            if retained:
                if prior_audio is None or prior_audio["sample_rate"] != rate:
                    raise ValueError("Missing matching generated audio history")
                wave = torch.cat((prior_audio["waveform"].cpu(), wave), -1)
            if wave.shape[-1] != round(end*rate/24):
                raise ValueError("Generated audio does not match the source clock")
            assembled_audio = {"sample_rate": rate, "waveform": wave}
        records = previous["windows"] if previous is not None else []
        return ({"contract": CONTRACT, "frames": frames, "audio": assembled_audio,
                 "windows": [*records, window], "source_frames": window["source_frames"],
                 "initial_history_frames": window["initial_history_frames"]},)


class H3RelayFinishWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": ("H3_RELAY_WINDOW_STATE",),
            "audio_mode": (["source", "generated", "none"], {"default": "source"})},
            "optional": {"source_audio": ("AUDIO",)}}
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "audio", "fps", "report")
    FUNCTION = "finish"
    CATEGORY = "H3 Relay/timeline"

    def finish(self, state, audio_mode="source", source_audio=None):
        if state["contract"] != CONTRACT or len(state["frames"]) != state["source_frames"]:
            raise ValueError("Completed windows do not cover the exact source timeline")
        history = state["initial_history_frames"]
        frames = state["frames"][history:]
        audio = source_audio if audio_mode == "source" else state["audio"] if audio_mode == "generated" else None
        audio = slice_audio(audio, history, len(state["frames"]))
        report = {"contract": CONTRACT, "fps": 24, "source_frames": len(frames),
                  "delivered_frames": len(frames), "history_frames": history,
                  "audio_mode": audio_mode, "body_composited": False, "complete": True,
                  "windows": state["windows"]}
        return frames, audio, 24., json.dumps(report, indent=2)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (H3RelayPrepareSource, H3RelayWindowPlan,
    H3RelayWindowInputs, H3RelayApplyHistory, H3RelayApplyEditMask, H3RelayAppendWindow, H3RelayFinishWindows)}
NODE_DISPLAY_NAME_MAPPINGS = dict(zip(NODE_CLASS_MAPPINGS, (
    "H3 · Prepare Source", "H3 · Plan Windows", "H3 · Window Inputs", "H3 · Apply History",
    "H3 · Apply Edit Mask", "H3 · Append Window", "H3 · Finish Windows")))
