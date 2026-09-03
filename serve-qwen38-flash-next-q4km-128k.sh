#!/usr/bin/env bash
# =============================================================================
# serve-qwen38-flash-next-q4km-128k.sh -- Qwen3.8-Flash-Next Q4_K_M, 131072 ctx -- the fastest of the four
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh qwen38-flash-next-q4km-128k
#
#   ./serve-qwen38-flash-next-q4km-128k.sh              131072 ctx
#   ./serve-qwen38-flash-next-q4km-128k.sh --port 9000  override port
#   ./serve-qwen38-flash-next-q4km-128k.sh --dry-run    print, no run
#
#   MEASURED 2026-09-03 (RESULTS 51), llama-sweep-bench, shallow:
#     3486.3 tok/s prefill, 128.9 t/s generation
#
# 3.2x the generation of the Q8 default, for a 4-bit quality trade. Only 33.9 GiB
# spills to host RAM here against 95.9 at Q8, and on this model expert residency
# is worth more than anything else: the -ncmoe 13 -> 0 sweep gained 12 % prefill
# and 113 % generation (RESULTS 51.2).
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

exec ./serve.sh qwen38-flash-next-q4km-128k "$@"
