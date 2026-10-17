"""Optional mask policies and app file adapters built around native SAM nodes.

Derived from the published H3 Character Swap 0.1 worker (GPL-3.0).
Detection, tracking, and ordinary mask operations remain native ComfyUI nodes.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
import folder_paths
import comfy.model_management
import comfy.utils


def expand_masks(raw, radius):
    if not 0 <= radius <= 256:raise ValueError('Expansion must be between 0 and 256 pixels')
    result=[]
    for mask in raw:
        binary=mask>0.5
        result.append((distance_transform_edt(~binary)<=radius) if radius and binary.any() else binary)
    return np.stack(result).astype(np.float32)


class SavedPersonMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'mask_file':('STRING',{'default':'person-mask.npz'}),'images':('IMAGE',),
            'source_frames':('INT',{'default':124,'min':1}),
            'expansion_px':('INT',{'default':64,'min':0,'max':256,'tooltip':'Outward Euclidean radius at the prepared video resolution. No automatic anchor-based expansion.'})}}
    RETURN_TYPES=('MASK','MASK','STRING')
    RETURN_NAMES=('expanded_person_mask','raw_person_mask','provenance')
    FUNCTION='load'
    CATEGORY='H3 Relay/mask tools'
    @classmethod
    def IS_CHANGED(cls,mask_file,**kwargs):
        return hashlib.sha256((Path(folder_paths.get_input_directory())/mask_file).read_bytes()).hexdigest()
    def load(self,mask_file,images,source_frames,expansion_px):
        root=Path(folder_paths.get_input_directory()).resolve();p=(root/mask_file).resolve()
        if not p.is_relative_to(root):raise ValueError('Mask must be inside ComfyUI input')
        with np.load(p,allow_pickle=False) as z:
            raw=z['mask']
            if 'present' in z and not z['present'].all():raise ValueError('Saved mask has unreviewed missing frames')
        if raw.ndim!=3 or len(raw)!=source_frames or len(images)<source_frames:raise ValueError('Mask and real source timeline must match exactly')
        if not np.isfinite(raw).all():raise ValueError('Mask contains non-finite values')
        h,w=images.shape[1:3]
        if raw.shape[1:]!=(h,w):
            raw=np.stack([np.asarray(Image.fromarray((x>.5).astype(np.uint8)*255).resize((w,h),Image.Resampling.NEAREST))>127 for x in raw])
        else:raw=raw>.5
        expanded=expand_masks(raw,expansion_px)
        raw=raw.astype(np.float32)
        if len(images)>source_frames:
            n=len(images)-source_frames
            expanded=np.concatenate([expanded,np.repeat(expanded[-1:],n,0)])
            raw=np.concatenate([raw,np.repeat(raw[-1:],n,0)])
        provenance=json.dumps({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_frames':source_frames,'prepared_frames':len(images),'dimensions':[w,h],'expansion_px':expansion_px,'metric':'euclidean_outward_radius','anchor_auto_expansion':False})
        return(torch.from_numpy(expanded),torch.from_numpy(raw),provenance)


class SaveMaskNPZ:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'masks':('MASK',),'filename_prefix':('STRING',{'default':'person-tail'}),'frames':('INT',{'default':5,'min':1})}}
    RETURN_TYPES=()
    FUNCTION='save'
    OUTPUT_NODE=True
    CATEGORY='H3 Relay/mask tools'
    def save(self,masks,filename_prefix,frames):
        root=Path(folder_paths.get_output_directory()).resolve();p=(root/(filename_prefix+'.npz')).resolve()
        if not p.is_relative_to(root):raise ValueError('Output outside output directory')
        if len(masks)<frames:raise ValueError('Insufficient mask frames')
        p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,mask=(masks[:frames].cpu().numpy()>.5))
        return {'ui':{'mask_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':frames}]}}


class SelectSubject:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",),
            "selection": (["largest", "leftmost", "rightmost", "index"],),
            "index": ("INT", {"default": 0, "min": 0})}}

    RETURN_TYPES = ("MASK", "H3_SUBJECT_SELECTION")
    RETURN_NAMES = ("selected_mask", "selection")
    FUNCTION = "select"
    CATEGORY = "H3 Relay/mask tools"

    def select(self, masks, selection, index):
        areas = masks.flatten(1).sum(1)
        valid = torch.where(areas > 4)[0].tolist()
        if not valid:
            raise ValueError("SAM found no subject. Adjust the subject text and inspect the mask preview.")
        if selection == "largest":
            chosen = max(valid, key=lambda i: areas[i].item())
        elif selection in {"leftmost", "rightmost"}:
            ordered = sorted(valid, key=lambda i: float(torch.where(masks[i] > 0.5)[1].float().mean()))
            chosen = ordered[0 if selection == "leftmost" else -1]
        else:
            if not 0 <= index < len(valid):
                raise ValueError(f"Subject index {index} exceeds {len(valid)} detections.")
            chosen = valid[index]
        return masks[chosen:chosen + 1], {"selection": selection, "index": index}


class ExpandMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",), "images": ("IMAGE",),
            "source_frames": ("INT", {"default": 120, "min": 1}),
            "expansion_px": ("INT", {"default": 64, "min": 0, "max": 256,
                "tooltip": "Outward Euclidean radius in prepared-video pixels; identical to the app's mask expansion."})}}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("generation_mask", "mask_overlay")
    FUNCTION = "expand"
    CATEGORY = "H3 Relay/mask tools"

    def expand(self, masks, images, source_frames, expansion_px):
        if not 1 <= source_frames <= len(images) or len(masks) < source_frames:
            raise ValueError("Masks must cover the complete source timeline.")
        if masks.shape[1:] != images.shape[1:3]:
            raise ValueError("Masks and prepared video dimensions differ.")
        raw = masks[:source_frames].detach().cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError("Mask contains non-finite values.")
        missing = np.flatnonzero((raw > .5).sum(axis=(1, 2)) == 0).tolist()
        if missing:
            raise ValueError(f"SAM lost the selected person at frames {missing[:20]}. Correct the selection or split the source shot before generating.")
        expanded = torch.from_numpy(expand_masks(raw, expansion_px))
        if len(images) > source_frames:
            expanded = torch.cat([expanded, expanded[-1:].expand(len(images) - source_frames, -1, -1)])
        overlay = torch.where(expanded[..., None] > 0,
            images.cpu() * .65 + torch.tensor([.1, .9, .4]) * .35, images.cpu())
        return expanded, overlay


class RecoverMasks:
    """Repeat the app's per-frame SAM recovery only where tracking is empty."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",), "images": ("IMAGE",),
            "model": ("MODEL",), "conditioning": ("CONDITIONING",),
            "selection": ("H3_SUBJECT_SELECTION",), "source_frames": ("INT",),
            "recover_empty_frames": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("recovered_masks", "report")
    FUNCTION = "recover"
    CATEGORY = "H3 Relay/mask tools"

    def recover(self, masks, images, model, conditioning, selection, source_frames, recover_empty_frames):
        import json
        import comfy.model_management
        from comfy_extras.nodes_sam3 import SAM3_Detect
        if len(masks) < source_frames or len(images) < source_frames:
            raise ValueError("Tracking must cover the complete source timeline")
        result = masks.clone().cpu()
        empty = torch.where((result[:source_frames] > .5).flatten(1).sum(1) == 0)[0].tolist()
        recovered = []
        if recover_empty_frames:
            for frame in empty:
                comfy.model_management.throw_exception_if_processing_interrupted()
                detection = SAM3_Detect.execute(model=model, image=images[frame:frame+1],
                    conditioning=conditioning, threshold=.35, refine_iterations=2, individual_masks=True)[0]
                result[frame] = SelectSubject().select(detection, **selection)[0][0].cpu()
                recovered.append(frame)
        report = json.dumps({"empty_tracking_frames": empty, "recovered_frames": recovered})
        return {"result": (result, report), "ui": {"text": [report]}}


class RefinePersonMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "positive_coords": ("STRING", {"default": "[]"}),
                "negative_coords": ("STRING", {"default": "[]"}),
                "prior_file": ("STRING", {"default": ""}),
                "frame": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "refine"
    CATEGORY = "H3 Relay/mask tools"

    @classmethod
    def IS_CHANGED(cls, prior_file, **kwargs):
        if not prior_file:
            return "no-prior"
        return hashlib.sha256(cls.prior_path(prior_file).read_bytes()).hexdigest()

    @staticmethod
    def prior_path(name):
        root = Path(folder_paths.get_input_directory()).resolve()
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError("Prior mask must be a staged input")
        return p

    def refine(self, model, image, positive_coords, negative_coords, prior_file, frame):
        if len(image) != 1:
            raise ValueError("Refine exactly one correction frame")
        _, h, w, _ = image.shape
        pos, neg = json.loads(positive_coords), json.loads(negative_coords)
        if not pos:
            raise ValueError(
                "An include point or derived prior-mask anchor is required"
            )
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam = model.model.diffusion_model
        pixels = comfy.utils.common_upscale(
            image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled"
        ).to(device=device, dtype=dtype)
        points = {
            "point_coords": torch.tensor(
                [[[p["x"] / w * 1008, p["y"] / h * 1008] for p in pos + neg]],
                dtype=dtype,
                device=device,
            ),
            "point_labels": torch.tensor(
                [[1] * len(pos) + [0] * len(neg)], dtype=torch.int32, device=device
            ),
        }
        logits = None
        if prior_file:
            with np.load(self.prior_path(prior_file), allow_pickle=False) as z:
                raw = z["mask"]
                if raw.ndim != 3 or frame >= len(raw):
                    raise ValueError("Prior mask frame is outside the saved timeline")
                prior = raw[frame] > 0.5
            if prior.any():
                prior = np.asarray(
                    Image.fromarray(prior).resize((w, h), Image.Resampling.NEAREST)
                ).copy()
                logits = (
                    torch.from_numpy(prior).to(device=device, dtype=dtype)[None, None]
                    * 12
                    - 6
                )
                logits = F.interpolate(
                    logits, size=(1008, 1008), mode="bilinear", align_corners=False
                )
        # Keep the user's positive/negative constraints in both refinement passes.
        for _ in range(2):
            logits = sam.forward_segment(
                pixels, point_inputs=points, mask_inputs=logits
            )
        mask = (
            F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)[0]
            > 0
        )
        return (mask.float().cpu(),)


NODE_CLASS_MAPPINGS = {
    "H3RelaySelectMask": SelectSubject,
    "H3RelayExpandMask": ExpandMask,
    "H3RelayRecoverMasks": RecoverMasks,
    "H3RelayLoadMask": SavedPersonMask,
    "H3RelaySaveMask": SaveMaskNPZ,
    "H3RelayRefineMask": RefinePersonMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RelaySelectMask": "Select Mask by Area or Position",
    "H3RelayExpandMask": "Prepare Mask Sequence (Euclidean Radius)",
    "H3RelayRecoverMasks": "Recover Empty SAM Frames",
    "H3RelayLoadMask": "Load Reviewed Mask Sequence",
    "H3RelaySaveMask": "Save Mask Sequence",
    "H3RelayRefineMask": "Refine Mask with Prior and Points",
}
