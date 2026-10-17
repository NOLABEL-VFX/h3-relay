"""Optional durable prefix review, independent of H3 sampling implementation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import removal_cache, removal_previews, removal_inputs
from .toolkit import H3RelayAppendWindow


def graph_identity(graph):
    """Conservative identity of graph parameters and referenced local file versions.

    Review choices do not change generation configuration. Everything else does;
    this deliberately prefers invalidation over reusing incompatible history.
    """
    import folder_paths
    fields = {"unet_name": "diffusion_models", "clip_name": "text_encoders",
              "vae_name": "vae", "lora_name": "loras", "ckpt_name": "checkpoints"}
    result = {}
    for key, node in sorted(graph.items()):
        inputs = {k: v for k, v in node.get("inputs", {}).items() if k != "window_controls"}
        files = {}
        for field, value in inputs.items():
            if not isinstance(value, str):
                continue
            path = None
            if field in fields:
                path = folder_paths.get_full_path(fields[field], value)
            elif field in ("image", "file", "mask_file"):
                path = folder_paths.get_annotated_filepath(value)
            if path and Path(path).is_file():
                stat = Path(path).stat()
                files[field] = [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        result[key] = {"class_type": node["class_type"], "inputs": inputs, "files": files}
    return result


class H3RelayReviewWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"windows": ("H3_RELAY_WINDOW",), "source": ("IMAGE", {"lazy": True}),
                "anchor": ("IMAGE",), "window_controls": ("STRING", {"default": "{}"})},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "execution_prompt": "PROMPT"}}
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False, False, False, False)
    RETURN_TYPES = ("H3_RELAY_WINDOW", "H3_RELAY_WINDOW_STATE", "INT", "H3_RELAY_WINDOWS", "IMAGE")
    RETURN_NAMES = ("pending_list", "restored_prefix", "count", "windows", "prepared_source")
    FUNCTION = "review"
    CATEGORY = "H3 Relay/timeline"

    @classmethod
    def IS_CHANGED(cls, execution_prompt=None, unique_id=None, **kwargs):
        graph = execution_prompt[0] if execution_prompt else {}
        node_id = unique_id[0] if unique_id else ""
        return removal_inputs.signature(graph, node_id, 0, ("source",)) or float("nan")

    def check_lazy_status(self, source=None, unique_id=None, extra_pnginfo=None,
                          execution_prompt=None, **kwargs):
        graph = execution_prompt[0] if execution_prompt else {}
        node_id = unique_id[0] if unique_id else ""
        scope = removal_cache.scope_key(node_id, extra_pnginfo[0] if extra_pnginfo else {})
        key = removal_inputs.signature(graph, node_id, 0, ("source",))
        if removal_inputs.available(scope, key):
            return []
        return ["source"] if source is None or source[0] is None else []

    def review(self, windows, source, anchor, window_controls, unique_id=None,
               extra_pnginfo=None, execution_prompt=None):
        first = windows[0]
        if first["initial_history_frames"]:
            raise ValueError("Standalone checkpoint review does not accept external segment history")
        node_id = unique_id[0] if unique_id else ""
        extra = extra_pnginfo[0] if extra_pnginfo else {}
        graph = execution_prompt[0] if execution_prompt else {}
        scope = removal_cache.scope_key(node_id, extra)
        prepared_key = removal_inputs.signature(graph, node_id, 0, ("source",))
        if removal_inputs.available(scope, prepared_key):
            source = [removal_inputs.load(scope, prepared_key)]
        elif prepared_key:
            removal_inputs.save(scope, prepared_key, source[0])
        identity = {"version": 2, "source": removal_cache.tensor_hash(source[0]),
                    "anchor": removal_cache.tensor_hash(anchor[0]), "windows": windows,
                    "implementation": hashlib.sha256(Path(__file__).read_bytes() +
                        Path(__file__).with_name("toolkit.py").read_bytes()).hexdigest(),
                    "graph": graph_identity(graph)}
        config = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        controls, reset = removal_cache.controls_for_configuration(window_controls[0], scope, config)
        chosen = json.loads(controls)
        for entry in windows:
            chosen.setdefault("windows", {}).setdefault(str(entry["index"]), {}).setdefault("seed", str(entry["seed"]))
        controls = json.dumps(chosen)
        # Window seeds are separately editable; temporal metadata stays invariant.
        metadata = [{k: v for k, v in entry.items() if k != "seed"} for entry in windows]
        starts = [entry["source_start"] for entry in windows]
        choices = removal_cache.plan(controls, scope, config, starts, first["window_frames"],
            first["source_frames"], first["seed"], derive_seeds=first["schedule"] == "removal_v1",
            window_plan=metadata)
        if all(choice["record_id"] for choice in choices):
            raise ValueError("Unlock at least the final window to render; completed output remains saved")
        run = removal_previews.begin_run(node_id, extra, starts, first["window_frames"],
            first["source_frames"], first["seed"], config_key=config, controls_reset=reset,
            window_plan=metadata, window_policy=first["schedule"])
        state, pending = None, []
        for entry, choice, temporal in zip(windows, choices, metadata):
            entry = {**entry, "seed": choice["seed"]}
            if choice["record_id"]:
                frames, audio = removal_cache.load(scope, choice["record_id"], config)
                state, = H3RelayAppendWindow().append(frames, entry, state, audio)
                state["records"].append(choice["record_id"])
                record = removal_cache.metadata(scope, choice["record_id"])
                removal_previews.publish_window(run, entry["index"], frames, entry["source_start"],
                    entry["discarded_overlap"], record=record, reused=True)
            else:
                pending.append({**entry, "checkpoint": {"scope": scope, "config_key": config,
                                "run_id": run, "window_metadata": temporal}})
        return {"result": (pending, state, len(pending), {"windows": pending}, source[0]),
                "ui": {"h3_removal_run": [run]}}


NODE_CLASS_MAPPINGS = {"H3RelayReviewWindows": H3RelayReviewWindows}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RelayReviewWindows": "H3 · Review Windows"}
