#!/usr/bin/env bash
# =============================================================================
# serve-qwen38-flash-next-q8-128k-mtp.sh -- Qwen3.8-Flash-Next Q8_0, 131072 ctx, + MTP
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh qwen38-flash-next-q8-128k-mtp
#
#   ./serve-qwen38-flash-next-q8-128k-mtp.sh              131072 ctx
#   ./serve-qwen38-flash-next-q8-128k-mtp.sh --port 9000  override port
#   ./serve-qwen38-flash-next-q8-128k-mtp.sh --dry-run    print, no run
#
#   MEASURED 2026-09-14 (RESULTS 54), against the plain Q8_0 profile:
#     generation, shallow   code 39 -> 64 t/s, JSON 40 -> 66, prose 40 -> 45
#     generation, 124k      code 32 -> 37 t/s, prose 32 -> 28
#     prefill               -12 to -15 % at every depth
#
# Needs the separate MTP head file (download line in the profile) and
# ik_llama.cpp >= 1b542a42 for per-step checkpoints. NOT SOAKED.
#
# Two deliberate defaults (same as the other wrappers):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by a GPU-resident LM Studio/Ollama
#     model. (It only ever sees NVIDIA processes.)
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh qwen38-flash-next-q8-128k-mtp "$@"
