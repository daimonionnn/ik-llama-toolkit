#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh -- half-million token context
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-512k
#
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh              524288 ctx
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh --port 9000  override the port
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh --dry-run    print, do not run
#
# Same MXFP4 quant as the 128k wrapper, but with the placement inverted: the KV
# lives in RAM (-nkvo) and the VRAM it would have taken goes to experts instead
# (--n-cpu-moe 19, no --fit). DeepSeek Sparse Attention attends ~512 positions
# regardless of depth, so the KV is cheap to keep in slow memory -- while the
# experts are not. MEASURED at 130k depth: 277.9 tok/s prefill, 16.30 generation,
# versus 205.5 / 13.33 for the --fit default. See RESULTS.md section 10.
#
# WHAT 512k ACTUALLY COSTS YOU:
#   * a full 499 909-token prefill runs at 161.8 tok/s and takes ~52 minutes;
#   * generation at that depth is 10.08 tok/s;
#   * the prompt cache and context checkpoints are off (they cost 54 GiB of RAM
#     and 26 % of generation at this context), so re-sending a conversation
#     re-prefills all of it. This is for long single-shot contexts, not chat.
# For anything that fits in 128k, ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh
# is faster on every axis.
#
# Two deliberate defaults (same as the other wrappers):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by a GPU-resident LM Studio/Ollama
#     model. (It only ever sees NVIDIA processes, so an LM Studio running on a
#     ROCm card is not affected.)
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh deepseek-v4-flash-512k "$@"
