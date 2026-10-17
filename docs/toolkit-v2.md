# Reusable H3 toolkit migration

Status: implementation branch; not a release or GPU-equivalence claim.

The new workflows use native ComfyUI Start Loop / End Loop. Model loaders,
reference conditioning, target guides, noise, sampler, scheduler, VAE encode /
decode, and video saving remain visible native nodes. The toolkit owns source
timing, H3 window geometry, continuation history, H3 mask conversion and assembly.

## Public contracts

| Node | Responsibility |
| --- | --- |
| H3RelayPrepareSource | Source-exact 24 fps preparation, 32-pixel canvas, model padding and original length. |
| H3RelayWindowPlan | Immutable window descriptions and count for a native loop. |
| H3RelayWindowInputs | Source/mask/audio slicing, H-1 reference context, generated history and seed. |
| H3RelayApplyHistory | Shared sliding history with explicit first-frame and target-audio guide policies. |
| H3RelayApplyEditMask | Conservative H3 temporal/spatial pooling around native AV masked sampling. Accepts a native VAEEncode result. |
| H3RelayAppendWindow | Remove overlap, validate coverage, and assemble raw frames plus optional generated audio. |
| H3RelayFinishWindows | Require complete coverage, remove supplied segment history, choose source/generated/no audio and report provenance. |
| H3RelayReviewWindows | Optional prepared-source recovery, exact durable prefix restoration, previews and seed rerolls. Sampling remains in the native loop. |

The `h3-relay-window-v2` entry records source start/end, generated size, padding,
history extent, overlap to discard, delivered count and exact seed. Window state
is passed through the native loop carry socket. Ordinary guides use
MiniMaxH3AddGuide. History before target frame zero still uses Relay's shared
sliding-history implementation.

The adaptive schedule preserves Character Swap's existing planner. The
`removal_v1` schedule preserves Person Remover's original tail repositioning,
18-frame history and incrementing seeds. The segment schedule represents one
explicit app pass with supplied history; that history is excluded from delivery.
Do not silently substitute one schedule for another.

The pure `planning.py` module owns the planners used by these nodes, legacy
workflows and app planning. It imports neither ComfyUI nor Torch. The app can
consume that canonical file without keeping a second planner implementation.

Mask tools are optional policies/file adapters, separate from this eight-node
timeline surface. Native SAM owns detection and tracking. The retained extras
cover area/position selection, empty-frame recovery, reviewed NPZ interchange,
prior-mask point correction and exact Euclidean expansion. They share one
implementation rather than being copied between Relay and the app.

Two optional history-file adapters preserve reviewed NPZ input and 16-bit RGB
FFV1 continuation masters. The qualified native SaveVideo exposes H.264/AV1,
not that lossless continuation format. Ordinary deliverables use native saving.

## Qualified native-loop serialization

ComfyUI `79be670e2d9be63e238785af307369d2b9039ed1` with frontend `1.53.6`
executes native loops. Its List-mode connection was lost on a real browser
export/reload, so these workflows use the native count mode and explicit
iteration indices. They leave iteration caching at its native false default;
explicit reviewed prefixes are restored from disk before the loop starts.
Both downloadable canvas graphs must pass exact input/connection round trips.

## Compatibility and validation gates

- Existing Relay identifiers remain available during migration. Preparation's
  old duration policy is retained under its old identifier.
- App source audio, raw decoded output, H-1 source/video reference alignment,
  model provenance and explicit segment history must remain unchanged.
- Person Remover's lock/reroll/recovery contract uses Review Windows. Native
  iteration caching alone is not durable recovery; GPU/UI validation is pending.
- The stock reference encoder replaces the copied encoder. A scoped CLIP policy,
  H3RelayTextMemory, releases inactive VAEs before native Qwen encoding and
  releases Qwen afterward. The matched 24 GB test exhausted memory without
  this handoff. H3RelaySamplingMemory supplies the existing Character Swap
  17 GiB activation-workspace estimate to the native sampler. Stock estimation
  otherwise loaded 19.3 GiB of weights and failed on its first sampling step.
  These policies change residency only and respect explicit high-VRAM mode.
- Optional runtime compatibility fixes and app-owned lifecycle hooks must be
  isolated from model/timeline semantics.
- Canvas files must be exported and reloaded through the actual frontend.
- Both old/new workflows need matched GPU runs with identical media, masks,
  model hashes, prompts, seeds and settings; preserve originals and comparison
  evidence. Frame count, audio duration, masks and continuation boundaries are
  explicit checks, alongside visual review.
- Update and test the app recipe, allowed node list, graph inputs and output
  adapter together. Install the canonical Relay revision; do not retain a
  second fork of its implementation in the app.
- Versioning, Registry status and publication remain
  pending until these gates pass.

Initial evidence: 94 CPU tests and the legacy expanded runtime contract pass.
An actual ComfyUI PromptExecutor run using native loops preserved 44 source
frames and 440 test audio samples across three windows. A fresh executor restored
two exact checkpoints and evaluated only the third window, invalidated the full
prefix after a settings change, and restored prepared source pixels without
re-evaluating their upstream node. Native threshold/dilation/compositing matches
the legacy green mask exactly. A 16-bit FFV1 round trip preserves every encoded
sample. These are CPU mechanics checks, not generated-output acceptance.

The package minimum is ComfyUI 0.37.0, which already includes the native loop and video-saving schemas used here. The exact pinned commit above remains the qualified target; its H3 audio resampling implementation differs from the release tag.
