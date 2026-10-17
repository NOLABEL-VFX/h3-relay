"""Portable continuation artifacts that native SaveVideo does not encode.

Ordinary deliverables use native CreateVideo/SaveVideo. These adapters retain
the app's 16-bit RGB history master and reviewed NPZ history input contract.
"""
from pathlib import Path
import hashlib
import subprocess

import numpy as np
import torch

from .planning import HISTORY


def input_path(name):
    import folder_paths
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("History file must be inside ComfyUI input")
    return path


class H3RelayLoadHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"history_file": ("STRING", {"default": "history.npz"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "H3 Relay/history files"

    @classmethod
    def IS_CHANGED(cls, history_file):
        return hashlib.sha256(input_path(history_file).read_bytes()).hexdigest()

    def load(self, history_file):
        with np.load(input_path(history_file), allow_pickle=False) as archive:
            frames = archive["frames"].astype(np.float32)
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) not in HISTORY or not len(frames):
            raise ValueError("History must contain 1 or 17n+1 RGB frames")
        if frames.shape[1] % 32 or frames.shape[2] % 32 or not np.isfinite(frames).all():
            raise ValueError("History requires a finite, 32-pixel-aligned canvas")
        if frames.min() < 0 or frames.max() > 1:
            raise ValueError("History values must be in the range zero to one")
        return (torch.from_numpy(frames),)


class H3RelaySaveHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "h3-relay/history"})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3 Relay/history files"

    def save(self, frames, filename_prefix):
        import folder_paths
        import json
        root = Path(folder_paths.get_output_directory()).resolve()
        directory, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(root), frames.shape[2], frames.shape[1])
        path = (Path(directory) / f"{filename}_{counter:05}_history.mkv").resolve()
        if not path.is_relative_to(root):
            raise ValueError("History output must remain inside ComfyUI output")
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", "24", "-i", "pipe:0",
            "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp16le", str(path)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in frames:
                pixels = frame.float().clamp(0, 1).mul(65535).round().cpu().numpy().astype("<u2")
                process.stdin.write(pixels.tobytes())
        finally:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Continuation master encoding failed")
        descriptor = {"filename": path.name, "subfolder": subfolder, "type": "output",
            "pixel_format": "rgb48le", "frames": len(frames), "fps": 24,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return {"result": (json.dumps(descriptor),), "ui": {"h3_relay_history": [descriptor]}}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3RelayLoadHistory, H3RelaySaveHistory)}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RelayLoadHistory": "H3 · Load Continuation History",
                             "H3RelaySaveHistory": "H3 · Save Continuation Master"}
