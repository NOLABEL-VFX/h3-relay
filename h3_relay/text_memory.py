"""A scoped residency policy; native H3 still tokenizes and encodes references."""
import logging


def staged_clip(clip, management):
    # Respect an explicit no-offloading launch. Normal residency can release
    # inactive VAEs before Qwen's activation peak, then Qwen before diffusion.
    if management.vram_state == management.VRAMState.HIGH_VRAM:
        return clip
    result = clip.clone()
    encode = result.encode_from_tokens_scheduled

    def staged_encode(*args, **kwargs):
        management.unload_all_models()
        logging.info("H3 text stage: released inactive models before native encoding")
        try:
            return encode(*args, **kwargs)
        finally:
            management.unload_all_models()
            logging.info("H3 text stage: released text encoder after native encoding")

    result.encode_from_tokens_scheduled = staged_encode
    return result


class H3RelayTextMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/Runtime"
    DESCRIPTION = "Release inactive models around native text encoding. Respects ComfyUI high-VRAM mode."

    def apply(self, clip):
        import comfy.model_management as management
        return (staged_clip(clip, management),)


NODE_CLASS_MAPPINGS = {"H3RelayTextMemory": H3RelayTextMemory}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RelayTextMemory": "H3 Text Stage Memory"}


def sampling_workspace(model, activation_memory_gib, management, wrapper_type):
    if management.vram_state != management.VRAMState.HIGH_VRAM:
        management.unload_all_models()
    patched = model.clone()
    native = model.get_model_object("memory_required")

    def memory_required(input_shape, cond_shapes=None):
        return max(native(input_shape, cond_shapes or {}), activation_memory_gib * 2**30)

    def scoped(executor, *args, **kwargs):
        base = patched.model
        previous = base.memory_required
        base.memory_required = memory_required
        try:
            return executor(*args, **kwargs)
        finally:
            base.memory_required = previous
            if management.vram_state != management.VRAMState.HIGH_VRAM:
                management.unload_all_models()
                logging.info("H3 sampling stage: released inactive model before native decoding")

    patched.add_wrapper(wrapper_type, scoped)
    return patched


class H3RelaySamplingMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "activation_memory_gib": ("FLOAT", {"default": 17., "min": 0., "max": 128., "step": .5,
                    "tooltip": "Estimated sampling workspace used by native model placement; not a GPU memory cap."})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/Runtime"
    DESCRIPTION = "Supply H3's measured activation workspace to native sampling after guides are encoded."

    def apply(self, model, latent, conditioning, activation_memory_gib=17.):
        import comfy.model_management as management
        from comfy.patcher_extension import WrappersMP
        return (sampling_workspace(model, activation_memory_gib, management, WrappersMP.OUTER_SAMPLE),)


NODE_CLASS_MAPPINGS["H3RelaySamplingMemory"] = H3RelaySamplingMemory
NODE_DISPLAY_NAME_MAPPINGS["H3RelaySamplingMemory"] = "H3 Sampling Workspace"
