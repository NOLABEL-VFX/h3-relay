# Notices

H3 Relay is licensed under GNU GPL version 3.

H3 Relay original code is copyright (C) 2026 Sendy Software LLC and
contributors. Vendored portions retain the following copyrights and licenses.

It contains attributed implementation snapshots from:

- ComfyUI's MiniMax H3 packed-layout and guide engine, GPL-3.0, adapted as a
  guarded process-local compatibility path for pre-target history anchors.
- `ComfyUI-MiniMaxH3-Contex-Loop`, GPL-3.0, based on work by NikoDemon80
  and ethanfel. Its license is retained under
  `h3_relay/vendor/context_loop/LICENSE`.
- `ComfyUI-Spectrum-MiniMax-H3`, GPL-3.0, by xmarre and contributors. Its
  license is retained under `h3_relay/vendor/spectrum/LICENSE`.
- `ComfyUI_MinimaxH3HybridLoader`, MIT, copyright 2026 Scott Mudge. Its
  license is retained under `h3_relay/vendor/hybrid/LICENSE.txt`.

The Relay-owned FastH3 VSA adapter implements the public Sol-Attention API
introduced by Comfy-Org/comfy-kitchen PR #117 and is informed by Kijai's
temporary MiniMax test adapter published with that pull request. comfy-kitchen
itself is Apache-2.0 and remains an external ComfyUI runtime dependency; its
source is not redistributed by H3 Relay.

The vendored code has been adapted and combined for H3 Relay. See
`UPSTREAMS.md` for source URLs, snapshot commits, and the local adaptation
boundary.

## Version-bounded Comfy Kitchen INT8 compatibility

`h3_relay/compat/triton_int8_large.py` retains two Apache-2.0 Triton kernels
from Comfy Kitchen with 64-bit pointer offsets. Copyright notices and license
headers are retained. This is the same qualified correction used by Character
Swap 1.1.6, applied only to Comfy Kitchen 0.2.35 on CUDA. It adds no workflow
node and changes no sampling policy. Remove the adapter when the qualified
upstream release includes the fix.

Upstream issue: https://github.com/Comfy-Org/comfy-kitchen/issues/136
Upstream proposed fix: https://github.com/Comfy-Org/comfy-kitchen/pull/172
