"""Apply the published workflow's grouped canvas and setup-note conventions.

This edits presentation metadata only. API graphs and generation settings stay
unchanged; export and reload the canvas in ComfyUI when reviewing changes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H3 = 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/'
SWAP = 'https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA'
REMOVE = 'https://huggingface.co/akatz-ai/MiniMax-H3-Person-Remover-LoRA'


def organize(path):
    graph = json.loads(path.read_text())
    removal = 'Removal' in path.name
    nodes = {str(n['id']): n for n in graph['nodes'] if n['type'] != 'MarkdownNote'}
    groups = []
    assigned = set()

    def section(title, columns, x, y, widths, color):
        members = []
        cursor = x
        for keys, width in zip(columns, widths):
            top = y
            for key in keys:
                if key not in nodes:
                    continue
                node = nodes[key]
                node['pos'] = [cursor, top]
                height = node['size'][1]
                if key in {'source', 'reference', 'anchor', 'mask_preview'}:
                    height = max(height, 330)
                if key == 'reference_conditioning':
                    height = 520
                if key == 'review':
                    height = 800
                node['size'] = [max(width, node['size'][0]), height]
                node['color'] = '#235044' if key in {'source', 'reference', 'anchor', 'subject', 'prepare'} else '#603d40' if key == 'save' else '#28465c' if node['type'].startswith('H3Relay') else '#303844'
                node['bgcolor'] = '#1e252d'
                top += height + 76
                members.append(node)
                assigned.add(key)
            cursor += width + 60
        left = min(n['pos'][0] for n in members) - 28
        top = min(n['pos'][1] for n in members) - 78
        right = max(n['pos'][0] + n['size'][0] for n in members) + 28
        bottom = max(n['pos'][1] + n['size'][1] for n in members) + 30
        groups.append({'id':len(groups)+1, 'title':title, 'bounding':[left, top, right-left, bottom-top], 'color':color, 'font_size':26, 'flags':{}})
        return bottom

    input_bottom = section('01 · User inputs', [['source', 'prepare', 'components', 'dimensions', 'reference', 'anchor']], 0, 100, [380], '#2b737b')
    model_bottom = section('02 · H3 models & LoRA', [['model', 'shift', 'attention', 'turbo', 'adapter'], ['text', 'text_device', 'text_memory', 'video_vae', 'audio_vae']], 480, 100, [370, 370], '#554477')
    section('04 · H3 Relay · window planning', [['plan', 'start', 'window']], 1380, 100, [400], '#396b89')
    section('05 · Reference conditioning & guides', [['reference_conditioning', 'history', 'encode_source', 'edit_mask', 'guide']], 1880, 100, [450], '#3f596e')
    section('06 · Sampling & decoding', [['noise', 'sampler', 'schedule', 'sampling_memory'], ['guider', 'sample', 'separate', 'decode', 'decode_audio']], 2430, 100, [320, 320], '#746345')
    section('07 · H3 Relay · assembly & output', [['append', 'end', 'finish', 'video', 'save', 'review']], 3230, 100, [550], '#3e6e56')
    section('03 · SAM · person selection & masks', [['sam', 'subject', 'first_frame', 'detect'], ['select', 'track', 'mask', 'recover', 'binary_mask'], ['expand_mask', 'grow_mask', 'green', 'green_guide', 'mask_preview']], 0, max(input_bottom, model_bottom)+110, [400, 360, 400], '#75694b')
    assert assigned == set(nodes), sorted(set(nodes)-assigned)

    files = [
        ('diffusion_models', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', H3+'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors'),
        ('text_encoders', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', H3+'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'),
        ('vae', 'minimax_h3_video_vae_fp16.safetensors', H3+'vae/minimax_h3_video_vae_fp16.safetensors'),
        ('vae', 'minimax_h3_audio_vae_fp32.safetensors', H3+'vae/minimax_h3_audio_vae_fp32.safetensors'),
        ('checkpoints', 'sam3.1_multiplex_fp16.safetensors', 'https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors'),
    ]
    if removal:
        files.append(('loras', 'H3-Person-Remover-V1.safetensors', REMOVE+'/resolve/main/H3-Person-Remover-V1.safetensors'))
    else:
        files += [('loras', 'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors', 'https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'), ('loras', 'h3_character_swap_v1.5.safetensors', SWAP+'/resolve/main/h3_character_swap_v1.5.safetensors')]
    links = '# Model Links\n\nDownload each file into the named folder under **ComfyUI/models/**.\n\n'
    for folder in dict.fromkeys(row[0] for row in files):
        links += '**'+folder+'/**\n\n'+''.join('- ['+filename+']('+url+')\n' for category, filename, url in files if category == folder)+'\n'
    links += ('**Person Remover V1:** strength 1.0. No Turbo LoRA.\n' if removal else '**Character Swap V1.5 + FL2V Turbo 8-step:** both strengths 1.0.\n')
    links += '\nRefresh model lists after downloading. Models retain their upstream licenses.'
    tree = '# Model Storage Location\n\n```text\nComfyUI/\n└── models/\n'
    folders = list(dict.fromkeys(row[0] for row in files))
    for i, folder in enumerate(folders):
        last = i == len(folders)-1
        tree += '    '+('└── ' if last else '├── ')+folder+'/\n'
        names = [f for category, f, _ in files if category == folder]
        for j, filename in enumerate(names):
            tree += '    '+('    ' if last else '│   ')+('└── ' if j == len(names)-1 else '├── ')+filename+'\n'
    tree += '```\n\nSelect these files in the matching loaders. An existing shared model directory also works when ComfyUI is configured to use it.\n'
    if not removal:
        tree += '\n**V1.5 filename:** the saved LoRA loader uses `h3-v2-challenge-eval-20261013/v2-1000.safetensors`, the validation alias for V1.5. Select your downloaded `h3_character_swap_v1.5.safetensors` in that loader, or keep the same subfolder/alias. Both names refer to the same weights.\n'
    tree += '\n## Custom nodes\n\nInstall [H3 Relay](https://github.com/akatz-ai/h3-relay) under `ComfyUI/custom_nodes/h3-relay/`. This canvas requires the **Relay 2.0 toolkit**. Use a current ComfyUI with native H3, SAM3 and Start Loop / End Loop support; restart after installing nodes. FFmpeg and FFprobe must be on PATH.\n\nBlue nodes are H3 Relay helpers; the surrounding groups show where they fit. Green nodes are user inputs. Red nodes save output.'
    intro = '# '+('Person Remover V1' if removal else 'Character Swap V1.5')+' · H3 Relay 2.0\n\n'
    intro += ('1. Upload a **24 fps source video** and a **clean version of source frame zero**, with exactly matching dimensions and framing.\n2. In **SAM person description**, identify the person to remove. Check that the green guide covers the whole person.\n3. Describe the background to reconstruct in **H3 prompt**.\n4. Run and review the completed windows. Reroll controls are in **Review Windows** at the output.\n\n**Defaults:** 22-frame windows, 18-frame history; 12 steps, `er_sde` / `simple`, CFG 1, sigma shift 12 / 3.\n\nThe clean first frame is a visual reference, not a pixel lock. Removing a person visually does not remove their voice.' if removal else '1. Upload a source video and a character image or sheet in **User inputs**.\n2. **Prepare Source** sets 24 fps and a canvas aligned to 32 pixels. Choose 480 or 768; duration 0 keeps the entire clip.\n3. In **SAM person description**, identify the person to replace. Inspect the expanded mask preview.\n4. In **H3 prompt**, describe the replacement and the performance to preserve. `<Picture 1>` is the character; `<Video 1>` is the source performance.\n5. Run and inspect every window boundary in the saved video.\n\n**Defaults:** both LoRAs at 1.0; 8 steps, `er_sde` / `normal`, CFG 1. This canvas has no separate scene anchor.\n\nFinal delivery uses the original source soundtrack. Start with one short continuous shot.')
    resolution = '''# MiniMax H3 · Size Settings Reference

**24 fps · width and height divisible by 32.** Match the source aspect ratio. More pixels, longer windows and larger reference images require more memory.

| Approx. MP | Landscape reference |
|---|---|
| 0.21 | 608 × 352 |
| 0.31 | 736 × 416 |
| 0.41 | 864 × 480 |
| 0.52 | 960 × 544 |
| 0.64 | 1056 × 608 |
| 0.74 | 1152 × 640 |
| 0.82 | 1216 × 672 |
| 0.94 | 1280 × 736 |
| 1.03 | 1344 × 768 |

These are manually chosen reference sizes. **1344 × 768 exceeds a strict 1 MP cap.**

'''
    resolution += ('This canvas reads the source size directly. Resize the source and clean first frame together before loading; they must match exactly. The table does not resize your inputs.' if removal else '**Prepare Source uses strict pixel caps**, not a fixed output height. For an ideal 16:9 input, 480 (≤0.5 MP) gives **928 × 512**; 768 (≤1 MP) gives **1344 × 736**. Portrait inputs swap those dimensions; square inputs give 704 × 704 or 992 × 992. Already aligned sources within the cap keep their size. The node outputs the actual width and height.')
    timing = '''# Windows, history & review

H3 window sizes follow **17n + 5**: 22, 39, 56, 73, 90, 107, 124…

| Window | Duration at 24 fps |
|---|---|
| 22 frames | 0.92 s |
| 39 frames | 1.63 s |
| 107 frames | 4.46 s |
| 124 frames | 5.17 s |

**History 18** carries generated context into the next window. Keep history smaller than the window. The planner owns source ranges and seeds; native loops execute them. Assembly trims repeated overlap and padding back to the source frame count.

Keep the loop connections intact. The groups separate inputs, models, masks, planning, conditioning, sampling and output; they do not change execution.

'''
    timing += ('**Review Windows:** previews persist with the saved workflow. Reroll regenerates a chosen window onward while keeping the exact accepted prefix. Preserve ComfyUI/output/__h3_removal_cache when moving the workflow. Changed media/models/settings invalidate incompatible checkpoints.\n\nGenerated audio is internal history. The saved video is silent by default; connect source audio to Create Video to retain the soundtrack.' if removal else '**Masking:** SAM selects and tracks the source person; Relay recovery/expansion prepares the edit mask. The final video is raw decoded generation, without pasting the subject through a mask. Inspect identity, hands, hair, occlusions and continuation seams.\n\nThe app adds reviewed anchors and independent segment rerenders on top of this same toolkit.')
    timing += '\n\n**Memory:** Text Stage Memory and Sampling Workspace manage stage transitions. They respect explicit high-VRAM mode. Normal offloading was required for the unchanged fixture on 24 GB cards; this is not a blanket memory-fit guarantee.'

    titles = {'source':'Source video', 'prepare':'Prepare Source · resolution & duration', 'reference':'Character reference · Picture 1', 'anchor':'Clean first frame · exact source framing', 'subject':'SAM person description', 'adapter':'Person Remover V1 · strength' if removal else 'Character Swap V1.5 · strength', 'turbo':'FL2V Turbo · 8-step', 'video_vae':'H3 video VAE · FP16', 'audio_vae':'H3 audio VAE · FP32', 'text':'H3 Qwen text encoder', 'model':'H3 Ref2VA · INT8', 'reference_conditioning':'H3 prompt · reference conditioning', 'mask_preview':'Review expanded person mask', 'review':'Review Windows · previews & reroll', 'save':'Save final video'}
    for key, title in titles.items():
        if key in nodes:
            nodes[key]['title'] = title
    notes = []
    for i, (title, body, pos, size) in enumerate([
        ('Note: Model Links', links, [-1640,100], [740,610]),
        ('Note: Model Storage & Custom Nodes', tree, [-1640,810], [740,920]),
        ('READ ME · Setup & defaults', intro, [-820,100], [700,450]),
        ('Note: Size Settings Reference', resolution, [-820,650], [700,610]),
        ('Note: Windows, History & Review', timing, [-820,1360], [700,820]),
    ], 1):
        notes.append({'id':i,'type':'MarkdownNote','pos':pos,'size':size,'flags':{},'order':len(nodes)+i,'mode':0,'inputs':[],'outputs':[], 'title':title,'properties':{},'widgets_values':[body],'color':'#443b27','bgcolor':'#29251d'})
    graph['nodes'] = list(nodes.values()) + notes
    graph['groups'] = groups
    graph.setdefault('extra', {})['ds'] = {'scale':0.29,'offset':[1900,350]}
    path.write_text(json.dumps(graph, indent=2)+'\n')


if __name__ == '__main__':
    for name in ('H3-Character-Swap', 'H3-Person-Removal'):
        organize(ROOT / 'example_workflows/toolkit' / (name+'.json'))
