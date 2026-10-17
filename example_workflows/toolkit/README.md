# Native H3 workflow toolkit

Open the canvas JSON in ComfyUI. The matching `.api.json` files are ordinary
API prompts. Both graphs use native Start Loop / End Loop and were exported
and reloaded through the actual frontend, with their inputs and links checked.

| Workflow | Canvas | API |
| --- | --- | --- |
| Character Swap V1.5 | [Open](H3-Character-Swap.json) | [API](H3-Character-Swap.api.json) |
| Person Removal V1 | [Open](H3-Person-Removal.json) | [API](H3-Person-Removal.api.json) |

## Canvas guide

Both canvases use numbered, colored groups for inputs, models, SAM masks,
window planning, conditioning, sampling and output. The left-hand notes include
clickable model downloads, a ComfyUI folder tree, setup instructions, resolution
references and window/history guidance. Green nodes are inputs, blue nodes are
Relay helpers and red nodes save output. The automatic resolution guide follows
Prepare Source's strict pixel caps; the manual size table is separate.

## Runtime and models

Install H3 Relay in the ComfyUI environment running the graph. No separate
H3 Character Swap node folder is required. The qualified runtime is ComfyUI
`79be670e2d9be63e238785af307369d2b9039ed1`, frontend 1.53.6, Python 3.12,
PyTorch 2.11.0 + CUDA 12.8 and comfy-kitchen 0.2.35 on Linux/NVIDIA. ComfyUI
0.37.0 is the minimum schema version. FFmpeg and FFprobe must be on PATH.

Both workflows need the following files, relative to ComfyUI's model directory:

| Model | Destination |
| --- | --- |
| [SAM 3.1](https://huggingface.co/Comfy-Org/sam3.1) | `checkpoints/sam3.1_multiplex_fp16.safetensors` |
| [H3 Ref2VA INT8](https://huggingface.co/Comfy-Org/MiniMax-H3) | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| [H3 Qwen encoder](https://huggingface.co/Comfy-Org/MiniMax-H3) | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| [H3 video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3) | `vae/minimax_h3_video_vae_fp16.safetensors` |
| [H3 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3) | `vae/minimax_h3_audio_vae_fp32.safetensors` |

Character Swap additionally uses [FL2V Turbo 8-step](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
at `loras/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` and
[Character Swap V1.5](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
at `loras/h3-v2-challenge-eval-20261013/v2-1000.safetensors`.
Download `h3_character_swap_v1.5.safetensors` and save it at that graph path,
or select its actual filename in the LoRA node. Its SHA-256 is
`8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f`.
V1 remains selectable as an alternative; it is not the default.

Person Removal uses [Person Remover V1](https://huggingface.co/akatz-ai/MiniMax-H3-Person-Remover-LoRA)
at `loras/H3-Person-Remover-V1.safetensors`, without the Turbo adapter.
Its SHA-256 is `b01dc9fe888c2f4e455a8eb3b4878dfd4038ee0e2320a7b186c7935e65c93ba5`.
All model licenses remain upstream; this package does not redistribute weights.

## Character Swap

1. Upload one continuous source shot and a character reference. Prepare Source
   uses a 24 fps, 32-pixel-aligned canvas. Duration zero preserves the full clip.
2. Describe the selected person to SAM, review the expanded mask, and write the
   character and performance directions in the native reference node.
3. Keep the Turbo adapter at strength 1, eight steps, normal scheduler,
   `er_sde` and CFG 1. The adaptive planner preserves source-exact delivery.
4. Review the saved raw generation, including identity, limbs, occlusions and
   every continuation boundary. Output is not pasted back through the mask.

The default canvas has no explicit scene anchor. The canonical Python recipe
accepts `use_anchor=True`; the app exposes reviewed anchors and saved masks.
Source video and reference audio share the same H-1 history extent. Final
audio comes from the source, trimmed to exactly the delivered frame count.

## Person Removal

1. Upload a 24 fps source whose dimensions are divisible by 32 and a clean
   version of its first frame at exactly the same size.
2. Describe the person to SAM. Native threshold, square dilation and green
   compositing produce the guide. Check that the person is fully covered.
3. Keep the removal_v1 planner, 18-frame history, 12 steps, simple scheduler,
   `er_sde` and CFG 1. The default window is 22 frames.
4. Review each completed window in Review Windows. Change a window's seed and
   click Reroll to retain the exact preceding windows and regenerate from there.
   Save the workflow to retain its review identity across browser restarts.

Review checkpoints and prepared source pixels live in ComfyUI's output
`__h3_removal_cache` directory. Preserve it when moving a saved workflow.
Changed media, models or graph settings invalidate incompatible checkpoints.
Generated audio is retained internally for history; the default final video
is silent. Connect source audio to Create Video if it should be included.

## Memory and app integration

Text Stage Memory releases inactive VAEs around native Qwen encoding.
Sampling Workspace supplies the qualified activation estimate and releases
the inactive diffusion model before native decoding. Both policies preserve
ComfyUI's explicit high-VRAM setting. On the tested 24 GB cards full residency
exceeded physical capacity; normal offloading completed the same generation
settings with optimized kernels enabled.

The app uses the same pinned recipe and node package. Optional history-file
adapters retain 16-bit FFV1 continuation masters for independent segment jobs.
See [the migration contract](../../docs/toolkit-v2.md) for inputs and outputs.
