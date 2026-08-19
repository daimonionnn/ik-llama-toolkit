#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-kvram-256k.sh -- fast-prefill 256k config
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-256k-kvram
#
#   ./serve-deepseek-v4-flash-mxfp4-kvram-256k.sh              262144 ctx
#   ./serve-deepseek-v4-flash-mxfp4-kvram-256k.sh --port 9000  override the port
#   ./serve-deepseek-v4-flash-mxfp4-kvram-256k.sh --dry-run    print, do not run
#
# The 262144 sibling of the kvram-128k wrapper: KV in RAM, freed VRAM spent on
# experts (--n-cpu-moe 18 -- the ceiling here; 17 OOMs on the bigger compute
# buffer), CPU experts repacked for AVX2/VNNI, -ub 2048.
#
#   MEASURED at 130k depth (RESULTS §12.1), caches on, vs the stock config:
#     stock        232.6 pp / 12.36 tg
#     this one     402.5 pp / 13.38 tg      +73 % prefill, +8 % generation
#
# Pick by window: 131072 is enough -> the kvram-128k wrapper is faster still
# (484 pp / 21 tg). Need more than 262144 -> the gpu-experts-512k wrapper.
# Robustness caveats as ever with kvram: manual placement (~3 GiB VRAM free),
# and -rtr re-reads the model at every start.
#
# Two deliberate defaults (same as the other wrappers):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by NVIDIA compute processes only,
#     so an LM Studio on the ROCm card is never touched.
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh deepseek-v4-flash-256k-kvram "$@"
