"""Native-node graph recipes shared by standalone workflows and app adapters.

This module has no ComfyUI or torch imports. An app may vendor it at the same
revision as its installed Relay pack; its output is a normal API prompt.
"""
from __future__ import annotations

def node(kind, title=None, **inputs):
    return {"class_type": kind, "inputs": inputs, "_meta": {"title": title or kind}}


def models(removal=False):
    graph = {
        "model": node("UNETLoader", unet_name="minimax_h3_ref2va_pruned_int8_convrot.safetensors", weight_dtype="default"),
        "shift": node("MiniMaxH3SigmaShift", model=["model", 0], shift_video=12., shift_audio=3.),
        "attention": node("ModelAttentionBackend", model=["shift", 0], attention="comfy kitchen attention"),
        "text": node("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax", device="default"),
        "text_device": node("SelectCLIPDevice", clip=["text", 0], device="gpu:0"),
        "text_memory": node("H3RelayTextMemory", clip=["text_device", 0]),
        "video_vae": node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "audio_vae": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
    }
    if not removal:
        graph["turbo"] = node("LoraLoaderModelOnly", model=["attention", 0],
            lora_name="minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", strength_model=1.)
    graph["adapter"] = node("LoraLoaderModelOnly", model=["attention" if removal else "turbo", 0],
        lora_name="H3-Person-Remover-V1.safetensors" if removal else "h3-v2-challenge-eval-20261013/v2-1000.safetensors",
        strength_model=1.)
    return graph


def sam(graph, frames, count, removal=False):
    graph.update({
        "sam": node("CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors"),
        "subject": node("CLIPTextEncode", clip=["sam", 1], text="man in gray shirt" if removal else "person"),
        "first_frame": node("ImageFromBatch", image=frames, batch_index=0, length=1),
        "detect": node("SAM3_Detect", model=["sam", 0], image=["first_frame", 0], conditioning=["subject", 0],
            threshold=.4 if removal else .35, refine_iterations=2, individual_masks=not removal),
    })
    initial_mask = ["detect", 0]
    if not removal:
        graph["select"] = node("H3RelaySelectMask", masks=initial_mask, selection="largest", index=0)
        initial_mask = ["select", 0]
    graph["track"] = node("SAM3_VideoTrack", model=["sam", 0], images=frames, initial_mask=initial_mask,
        conditioning=["subject", 0], detection_threshold=.5, max_objects=1, detect_interval=12 if removal else 1)
    graph["mask"] = node("SAM3_TrackToMask", track_data=["track", 0], object_indices="")
    if not removal:
        graph["recover"] = node("H3RelayRecoverMasks", masks=["mask", 0], images=frames, source_frames=count,
            model=["sam", 0], conditioning=["subject", 0], selection=["select", 1], recover_empty_frames=True)
        graph["expand_mask"] = node("H3RelayExpandMask", masks=["recover", 0], images=frames,
            source_frames=count, expansion_px=64)
        graph["mask_preview"] = node("PreviewImage", images=["expand_mask", 1])
        return ["expand_mask", 0]
    graph["binary_mask"] = node("ThresholdMask", mask=["mask", 0], value=.5)
    graph["grow_mask"] = node("GrowMask", mask=["binary_mask", 0], expand=5, tapered_corners=False)
    return ["grow_mask", 0]


def loop(graph, frames, count, audio=None, removal=False, use_anchor=False, use_mask=True, audio_guide=True, fps=24.):
    graph["plan"] = node("H3RelayWindowPlan", "Plan source-aligned windows", source_frames=count,
        window_frames="22" if removal else "124", history_frames="18", seed=904234 if removal else 904239,
        schedule="removal_v1" if removal else "adaptive", seed_mode="increment" if removal else "fixed",
        seed_overrides="{}")
    graph["start"] = node("StartLoop", "Native loop · one H3 window", mode="simple",
                         **{"mode.num_iterations": ["plan", 2]}, cache_iterations=False)
    inputs = dict(source=frames, window=["plan", 3], iteration_index=["start", 0], previous=["start", 4],
                  reference_history=not removal, fps=fps, resize_anchor=not removal,
                  history_audio_mode="generated" if removal else "source" if audio_guide else "none")
    if audio is not None:
        inputs["source_audio"] = audio
    if use_anchor or removal:
        inputs["anchor"] = ["anchor", 0]
    if use_mask and not removal:
        inputs["mask"] = ["expand_mask", 0]
    graph["window"] = node("H3RelayWindowInputs", **inputs)
    refs = {"ref_videos.ref_video_0": ["window", 5]}
    if removal:
        refs["ref_images.ref_image_0"] = ["window", 11]
    else:
        refs["ref_images.ref_image_0"] = ["reference", 0]
        if use_anchor:
            refs["ref_images.ref_image_1"] = ["window", 11]
        refs.update(vae=["video_vae", 0], audio_vae=["audio_vae", 0])
        if audio_guide:
            refs["ref_audios.ref_audio_0"] = ["window", 6]
    prompt = ("Remove the green-masked person and reconstruct the background. Preserve the rest of the video, including its camera motion and frame timing."
              if removal else "Replace the selected person with the reference character. Preserve source performance, scene and audio timing.")
    graph["reference_conditioning"] = node("MiniMaxH3ReferenceToVideo", "Native H3 reference conditioning",
        clip=["text_memory", 0], prompt=prompt, width=["dimensions", 0], height=["dimensions", 1],
        length=["window", 8], ref_image_size="match", **refs)
    history_args = dict(positive=["reference_conditioning", 0], latent=["reference_conditioning", 1],
        vae=["video_vae", 0], audio_vae=["audio_vae", 0], frames=["window", 3], history_frames=["window", 7],
        first_frame_guide=use_anchor and not removal, target_audio_guide=audio_guide and not removal)
    if audio_guide or removal:
        history_args["audio"] = ["window", 4]
    graph["history"] = node("H3RelayApplyHistory", **history_args)
    positive, latent = ["history", 0], ["history", 1]
    if use_mask and not removal:
        graph["encode_source"] = node("VAEEncode", pixels=["window", 0], vae=["video_vae", 0])
        graph["edit_mask"] = node("H3RelayApplyEditMask", latent=latent, source_latent=["encode_source", 0], mask=["window", 1])
        latent = ["edit_mask", 0]
    if removal or audio_guide:
        guide = dict(positive=positive, latent=latent, frame_idx=0)
        guide.update({"vae": ["video_vae", 0], "image": ["window", 0]} if removal else
                     {"audio_vae": ["audio_vae", 0], "audio": ["window", 2]})
        graph["guide"] = node("MiniMaxH3AddGuide", "Native target guide", **guide)
        positive = ["guide", 0]
    graph["sampling_memory"] = node("H3RelaySamplingMemory", model=["adapter", 0],
        latent=latent, conditioning=positive, activation_memory_gib=4. if removal else 17.)
    sampling_model = ["sampling_memory", 0]
    graph.update({
        "noise": node("RandomNoise", noise_seed=["window", 10]),
        "guider": node("CFGGuider", model=sampling_model, positive=positive, negative=positive, cfg=1.),
        "sampler": node("KSamplerSelect", sampler_name="er_sde"),
        "schedule": node("BasicScheduler", model=["adapter", 0], scheduler="simple" if removal else "normal",
                         steps=12 if removal else 8, denoise=1.),
        "sample": node("SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0],
            sampler=["sampler", 0], sigmas=["schedule", 0], latent_image=latent),
        "separate": node("LTXVSeparateAVLatent", av_latent=["sample", 0]),
        "decode": node("VAEDecode", samples=["separate", 0], vae=["video_vae", 0]),
        "append": node("H3RelayAppendWindow", frames=["decode", 0], window=["window", 12], previous=["start", 4]),
        "end": node("EndLoop", output_value=["append", 0], next_iteration_value=["append", 0], accumulate=False),
        "finish": node("H3RelayFinishWindows", state=["end", 0], audio_mode="none" if removal else "source"),
        "video": node("CreateVideo", images=["finish", 0], fps=["finish", 2]),
        "save": node("SaveVideo", video=["video", 0], filename_prefix="h3-relay/person-removal" if removal else "h3-relay/character-swap",
                     format="mp4", **{"format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16}),
    })
    if removal:
        graph["review"] = node("H3RelayReviewWindows", windows=["plan", 0], source=frames,
                               anchor=["anchor", 0], window_controls="{}")
        graph["start"]["inputs"]["mode.num_iterations"] = ["review", 2]
        graph["start"]["inputs"]["initial_iteration_value"] = ["review", 1]
        graph["window"]["inputs"]["window"] = ["review", 3]
        graph["window"]["inputs"]["source"] = ["review", 4]
        graph["decode_audio"] = node("VAEDecodeAudio", samples=["separate", 1], vae=["audio_vae", 0])
        graph["append"]["inputs"]["audio"] = ["decode_audio", 0]
    else:
        graph["finish"]["inputs"]["source_audio"] = audio
        graph["video"]["inputs"]["audio"] = ["finish", 1]


def character_swap_graph(use_anchor=False, use_mask=True, audio_guide=True, saved_mask=False):
    graph = models()
    graph.update({
        "source": node("LoadVideo", file="source.mp4"),
        "prepare": node("H3RelayPrepareSource", video=["source", 0], resolution="768", duration_seconds=0., missing_audio="silence"),
        "dimensions": node("GetImageSize", image=["prepare", 0]),
        "reference": node("LoadImage", image="reference.png"),
    })
    if use_anchor:
        graph["anchor"] = node("LoadImage", image="anchor.png")
    if use_mask:
        if saved_mask:
            graph["expand_mask"] = node("H3RelayLoadMask", mask_file="person-mask.npz", images=["prepare", 0],
                                         source_frames=["prepare", 5], expansion_px=64)
        else:
            sam(graph, ["prepare", 0], ["prepare", 5])
    loop(graph, ["prepare", 0], ["prepare", 5], ["prepare", 1],
         use_anchor=use_anchor, use_mask=use_mask, audio_guide=audio_guide)
    return graph


def person_removal_graph():
    graph = models(removal=True)
    graph.update({
        "source": node("LoadVideo", file="source.mp4"),
        "components": node("GetVideoComponents", video=["source", 0]),
        "dimensions": node("GetImageSize", image=["components", 0]),
        "anchor": node("LoadImage", image="clean-first-frame.png"),
        "green": node("EmptyImage", width=["dimensions", 0], height=["dimensions", 1], batch_size=1, color=65280),
    })
    mask = sam(graph, ["components", 0], ["dimensions", 2], removal=True)
    graph["green_guide"] = node("ImageCompositeMasked", destination=["components", 0], source=["green", 0],
                                 mask=mask, x=0, y=0, resize_source=False)
    loop(graph, ["green_guide", 0], ["dimensions", 2], removal=True, fps=["components", 2])
    return graph
