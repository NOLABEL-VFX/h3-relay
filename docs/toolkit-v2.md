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
| H3RelayWindowPlan | A list of immutable window descriptions for native List-mode loops. |
| H3RelayWindowInputs | Source/mask/audio slicing, H-1 reference context, generated history and seed. |
| H3RelayApplyHistory | Shared sliding history with explicit first-frame and target-audio guide policies. |
| H3RelayApplyEditMask | Conservative H3 temporal/spatial pooling around native AV masked sampling. Accepts a native VAEEncode result. |
| H3RelayAppendWindow | Remove overlap, validate coverage, and assemble raw frames plus optional generated audio. |
| H3RelayFinishWindows | Require complete coverage, remove supplied segment history, choose source/generated/no audio and report provenance. |

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

Mask tools are optional policies/file adapters, separate from this seven-node
timeline surface. Native SAM owns detection and tracking. The retained extras
cover area/position selection, empty-frame recovery, reviewed NPZ interchange,
prior-mask point correction and exact Euclidean expansion. They share one
implementation rather than being copied between Relay and the app.

## Compatibility and validation gates

- Existing Relay identifiers remain available during migration. Preparation's
  old duration policy is retained under its old identifier.
- App source audio, raw decoded output, H-1 source/video reference alignment,
  model provenance and explicit segment history must remain unchanged.
- Person Remover's lock/reroll/recovery contract must be ported before release.
  Native iteration caching alone is not durable recovery.
- The stock reference encoder replaces the copied encoder. Its GPU placement
  and memory behavior still require qualification at the intended settings.
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
- Versioning, minimum ComfyUI runtime, Registry status and publication remain
  pending until these gates pass.

Initial evidence: 91 CPU tests and the legacy expanded runtime contract pass.
An actual ComfyUI PromptExecutor run using native loops preserved 44 source
frames and 440 test audio samples across three windows. This is CPU mechanics
evidence, not generated-output acceptance.
