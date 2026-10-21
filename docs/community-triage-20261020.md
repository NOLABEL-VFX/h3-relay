# Community security fixes and reproduction checks — 20 October 2026

Status: implemented on an isolated local branch; not pushed, released, or accepted by the Comfy Registry. Based on published main `64341d5c1259f32ab2a780285e1a19accb4dfe07`. The existing development checkout and running services were not modified. The nightly community monitor remains report-only.

## Implemented changes

- `SavedPersonMask.IS_CHANGED()` and `load()` now use the same resolved-input-path validator before opening a file. Outside absolute paths, parent traversal, outside symlinks, directories, missing files and FIFOs are rejected. Valid input files and inside-root symlinks still work. Hashing streams in 1 MiB chunks rather than reading the whole file into memory.
- Cache URIs, absolute artifact paths, preview sources and preview/write destinations use canonical paths for containment. Configured roots may themselves be symlinks. Cache maintenance skips symlink entries instead of reading outside JSON files or treating aliases as pruneable revisions.
- Removed the unused, unregistered upstream disk Save/Load classes and unrestricted latent path resolver from the vendored context module. No registered node ID was removed. Attribution and licenses remain intact; Relay's managed checkpoint adapters remain available.
- Context audio now rejects malformed encoder/checkpoint latents before constructing stereo conditioning. The error includes the actual shape and expected `[1,32,2,T]` contract. Valid stereo values and supported unbatched checkpoint tails are preserved. No tensor padding, packed-row duplication or sampling change was introduced.
- Corrected the general README requirement from ComfyUI 0.32 to 0.37, matching package metadata and toolkit guidance.

The path fixes address the reproduced read/containment defects. The public Registry reason does not identify exact source locations, so these changes do not establish that every scanner finding is resolved or that a new release will be accepted. Root paths are trusted configuration; this is not a filesystem sandbox against an adversarial local process concurrently swapping directory entries or creating hard links.

## Verification

The original eight security regression tests failed on unmodified main with 34 failing subcases and two errors. The final suite passes **111 tests**, including ten filesystem regression tests and five context-audio tests, followed by the expanded ComfyUI runtime contract. Tests use disposable sentinel files only.

Runtime checks used the existing ComfyGit `h3-workbench` environment on `akatzfeyserver`, ComfyUI `79be670e2d9be63e238785af307369d2b9039ed1`, PyTorch `2.11.0+cu128`, and an RTX 4090. Both GPU attempts completed and released their claims. The final check used `--highvram --reserve-vram 0`, no imposed CPU/RAM limits, and peaked at 7,835,672,064 allocated CUDA bytes (about 7.30 GiB). Existing model caches were reused.

| Check | Result |
| --- | --- |
| Real H3 audio VAE, 10.3 seconds of mono, stereo and silent mono | All produced `[1,32,2,412]`, 824 packed rows, finite values |
| Actual native `extra_conds` and forward packing with image/audio references | Passed |
| Boundary guide plus image/video/audio references at 1344×736 | Passed |
| Relay history, boundary and reference merge | Passed |
| Real audio and video VAEs through `MiniMaxH3ReferenceToVideo`, `H3RelayApplyHistory`, and `MiniMaxH3AddGuide`, first window and 18-frame continuation | Both passed at 1344×736 |
| Deliberately injected single-channel audio latent | Reproduced the exact `[412,32]` into `[824,32]` broadcast error |
| Deliberately removed boundary latent while retaining its layout slot | Produced a 966-row deficit and a failing assignment |

The real-video check uses synthetic constant footage (22 source-reference frames and an 18-frame history) and a fixture text embedding. Packing runs through a tiny initialized diffusion model with zero attention blocks. This exercises the real conditioning and row assignments, not a full learned denoising run, generated image quality, the reporter's custom extensions, or their complete workflow. No media result is claimed.

Audio model SHA-256: `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`.

Video model SHA-256: `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.

## What remains unresolved

Ordinary mono waveform input is **not** enough to cause the reported audio error on this qualified runtime: ComfyUI's VAE wrapper already pads mono to stereo. Malformed/stale latents, a different encoder/core version, or mismatched conditioning can produce the exact error. The new guard catches malformed audio at Relay's context boundary, but is not claimed to fix an unidentified third-party path.

At 1344×736, one plane is `42 × 23 = 966` rows. Removing a boundary payload recreates that deficit, but the normal native and continuation paths retain it. The reported total `37696 → 36730` and its original emitting function were not reproduced. The complete traceback, exported workflow and exact core/extension revisions are still needed to identify the real cause. No speculative layout patch was applied.

## Replay

Run the CPU suite from this checkout with an existing prepared ComfyUI environment:

```sh
COMFY_VENV/bin/python tests/run_cpu_tests.py --comfyui-root COMFY_ROOT
```

`benchmark/reproduce_community_shapes.py` runs the actual core packing checks on CPU when given only `--comfyui-root` and `--out`. Supplying `--audio-vae` also runs real audio encoding on GPU; adding `--video-vae` exercises the published first-window and continuation nodes with both models. Submit GPU mode through the host's GPU admission tool. Each run writes exact source/model hashes, shapes, error traces, peak allocation and completion state to the requested JSON output. It does not start a server, fetch weights, change an environment, or write outside its explicitly requested evidence output.

Retained validation includes baseline/final test logs, both GPU probe logs and JSON results, the patch, and the commit/base identifiers. Registry resubmission and publication remain separate steps.
