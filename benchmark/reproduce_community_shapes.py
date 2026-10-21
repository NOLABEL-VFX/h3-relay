"""Probe reported packing failures with actual ComfyUI code, optionally a real VAE.

The diffusion model is deliberately tiny and has no attention blocks: this tests
conditioning assembly and forward packing, not generation quality or a full sampler.
Run GPU mode only through the worker's normal GPU admission mechanism.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfyui-root', required=True, type=Path)
parser.add_argument('--audio-vae', type=Path)
parser.add_argument('--video-vae', type=Path)
parser.add_argument('--out', required=True, type=Path)
args = parser.parse_args()
if args.video_vae and not args.audio_vae:
    parser.error('--video-vae requires --audio-vae')
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(args.comfyui_root), str(root / 'tests')]
sys.argv = [sys.argv[0]] + (['--highvram', '--reserve-vram', '0'] if args.audio_vae else ['--cpu'])
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.model_base as model_base
import comfy.ldm.minimax.model as mm
import comfy.ops
import comfy.sd
import comfy.utils
import module_fixture

context = importlib.import_module(module_fixture.PACKAGE + '.vendor.context_loop.nodes')
sliding = importlib.import_module(module_fixture.PACKAGE + '.vendor.context_loop.sliding_context')

result = {
    'comfyui_revision': subprocess.check_output(['git', '-C', str(args.comfyui_root), 'rev-parse', 'HEAD'], text=True).strip(),
    'torch': torch.__version__,
    'scope': 'real core packing and extra_conds; tiny zero-block diffusion model; optional real audio VAE; no full sampler',
    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (root / 'h3_relay/vendor/context_loop/nodes.py', root / 'benchmark/reproduce_community_shapes.py')},
    'audio': [], 'packing': [],
}


def save():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')


def payload_counts(model, payload):
    layout = payload['layout']
    video = model._cond_video_rows(payload, 'cpu')
    audio = model._cond_audio_rows(payload, 'cpu')
    return {
        'expected_video_condition_rows': int((~layout.img_update).sum()),
        'actual_video_condition_rows': 0 if video is None else len(video),
        'expected_audio_condition_rows': int((~layout.audio_update).sum()),
        'actual_audio_condition_rows': 0 if audio is None else len(audio),
        'total_rows': layout.seq_len,
    }


@torch.inference_mode()
def main():
    if args.audio_vae:
        state = comfy.utils.load_torch_file(str(args.audio_vae), safe_load=True)
        vae = comfy.sd.VAE(sd=state, device=torch.device('cuda'))
        del state
        with args.audio_vae.open('rb') as source:
            result['audio_vae_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        # 10.3 seconds produces precisely the reported 412 steps at 40 Hz.
        time = torch.arange(329600, dtype=torch.float32) / 32000
        mono = (0.1 * torch.sin(2 * torch.pi * 220 * time))[None, None]
        stereo = torch.cat((mono, mono * 0.5), dim=1)
        for label, waveform in [('mono', mono), ('stereo', stereo), ('silent-mono', torch.zeros_like(mono))]:
            latent, steps = context._encode_tail_audio(vae, {'waveform': waveform, 'sample_rate': 32000}, 10.3)
            rows = mm.pack_audio(latent)
            record = {'case': label, 'input_shape': list(waveform.shape), 'latent_shape': list(latent.shape),
                      'packed_rows': len(rows), 'steps': steps, 'finite': bool(torch.isfinite(latent).all())}
            result['audio'].append(record)
            assert list(latent.shape) == [1, 32, 2, 412], record
            assert record['finite']
            print(json.dumps(record), flush=True)
            save()
        result['gpu_peak_bytes'] = torch.cuda.max_memory_allocated()

    # Exercise actual core extra_conds and _forward, with tiny initialized weights
    # and no expensive attention/learned generation. Keep reported spatial size.
    model = mm.MiniMaxH3Model(hidden_size=24, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=24, ffn_hidden_size=48, text_dim=24,
        timestep_input_dim=16, time_embed_hidden_size=24, time_embed_dim=24,
        rope_inv_freq_len=4, dtype=torch.float32, device='cpu', operations=comfy.ops.disable_weight_init)
    for parameter in model.parameters():
        parameter.zero_()
    model.rope.inv_freq.zero_()
    base = model_base.MiniMaxH3.__new__(model_base.MiniMaxH3)
    torch.nn.Module.__init__(base)
    base.diffusion_model = model
    base.concat_keys = ()
    base.latent_shapes = None
    base.manual_cast_dtype = None
    embedding = torch.zeros(1, 22, 24)
    video = torch.zeros(1, 24, 37, 46, 84)
    audio = torch.zeros(1, 32, 2, 207)
    still = torch.zeros(1, 24, 1, 46, 84)
    history = torch.zeros(1, 24, 5, 46, 84)
    audio_ref = torch.zeros(1, 32, 2, 412)
    refs = [{'kind': 'image', 'latent': still, 'latent_h': 46, 'latent_w': 84},
            {'kind': 'audio', 'audio_latent': audio_ref, 'ref_audio_t': 412}]

    def run_case(label, keyframes, ref_list, streams=None, mutate=None):
        streams = streams or [video, audio]
        payload = base.extra_conds(cross_attn=embedding, device='cpu',
            latent_shapes=[list(t.shape) for t in streams], minimax_keyframes=keyframes,
            minimax_refs=ref_list)['minimax_payload'].cond
        if mutate:
            mutate(payload)
        record = {'case': label, **payload_counts(model, payload)}
        try:
            out = model._forward(streams, torch.tensor([500.]), embedding,
                                 transformer_options={}, minimax_payload=payload)
            record['forward'] = 'passed'
            record['output_shapes'] = [list(t.shape) for t in out]
            assert all(torch.isfinite(t).all() for t in out)
        except Exception as exc:
            record['forward'] = 'failed'
            record['error'] = str(exc)
            record['traceback'] = traceback.format_exc()
        result['packing'].append(record)
        print(json.dumps({k:v for k,v in record.items() if k != 'traceback'}), flush=True)
        save()
        return record

    assert run_case('native-first-window-ref-image-audio', [], refs)['forward'] == 'passed'
    first = [{'resolved_frame_index': 0, 'latent': still}]
    assert run_case('native-boundary-and-refs', first, refs)['forward'] == 'passed'
    video_ref = {'kind': 'video', 'latent': video, 'latent_t': 37, 'latent_h': 46,
                 'latent_w': 84, 'ref_audio_t': 0}
    assert run_case('native-image-video-audio-refs', first, refs + [video_ref])['forward'] == 'passed'
    # Positive diagnostic controls: deliberately broken payloads show whether
    # the harness reaches the same kinds of failing assignment.
    mono_ref = dict(refs[1], audio_latent=audio_ref[:, :, :1])
    assert run_case('injected-mono-latent', first, [refs[0], mono_ref])['forward'] == 'failed'
    broken = run_case('injected-missing-boundary', first, refs,
                     mutate=lambda p: p.update(cond_video_latents=p['cond_video_latents'][1:]))
    assert broken['expected_video_condition_rows'] - broken['actual_video_condition_rows'] == 966

    sliding.require_sliding_history_support()
    history_guides = [{'anchor': 'history', 'latent': history},
                      {'anchor': 'first', 'resolved_frame_index': 0, 'latent': still},
                      {'anchor': 'history', 'audio_latent': audio_ref[..., :28]},
                      {'anchor': 'first', 'resolved_frame_index': 0, 'audio_latent': audio_ref[..., :2]}]
    shifted = sliding._shift_target_guides([[embedding, {'minimax_keyframes': first, 'minimax_refs': refs}]], 17, 18, history_guides)
    target = [video[:, :, :32], audio[..., :178]]
    assert run_case('relay-continuation-history-boundary-refs', shifted[0][1]['minimax_keyframes'], refs, target)['forward'] == 'passed'
    if args.video_vae:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo
        toolkit = module_fixture.load('toolkit')
        state = comfy.utils.load_torch_file(str(args.video_vae), safe_load=True)
        video_vae = comfy.sd.VAE(sd=state, device=torch.device('cuda'))
        del state
        with args.video_vae.open('rb') as source:
            result['video_vae_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        # Synthetic constant footage; real VAE weights, published conditioning
        # nodes and Relay history path. Text embeddings alone are a fixture.
        frames = torch.full((22, 736, 1344, 3), 0.25)
        sound = {'waveform': mono, 'sample_rate': 32000}
        class ClipFixture:
            def tokenize(self, *a, **kw): return None
            def encode_from_tokens_scheduled(self, tokens): return [[embedding, {}]]
        cond, empty = MiniMaxH3ReferenceToVideo.execute(
            clip=ClipFixture(), prompt='shape probe', width=1344, height=736, length=124,
            vae=video_vae, audio_vae=vae, ref_images={'ref_image_0': frames[:1]},
            ref_videos={'ref_video_0': frames}, ref_audios={'ref_audio_0': sound})
        for history_count in (0, 18):
            guided, target_latent = toolkit.H3RelayApplyHistory().apply(
                cond, empty, video_vae, vae, frames[-18:], history_count,
                first_frame_guide=False, target_audio_guide=True, audio=sound)
            guided = MiniMaxH3AddGuide.execute(positive=guided, latent=target_latent,
                frame_idx=0, audio_vae=vae, audio=sound)[0]
            metadata = guided[0][1]
            record = run_case('real-vaes-published-nodes-history-%d' % history_count,
                metadata.get('minimax_keyframes', []), metadata.get('minimax_refs', []),
                list(target_latent['samples'].unbind()))
            assert record['forward'] == 'passed'
        result['gpu_peak_bytes'] = torch.cuda.max_memory_allocated()
    result['completed'] = True
    save()


try:
    main()
except Exception:
    result['fatal_error'] = traceback.format_exc()
    save()
    raise
