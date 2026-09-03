#!/usr/bin/env bash
# =============================================================================
# serve-qwen38-flash-next-q8-128k.sh -- Qwen3.8-Flash-Next Q8_0, 131072 ctx -- THE DEFAULT
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh qwen38-flash-next-q8-128k
#
#   ./serve-qwen38-flash-next-q8-128k.sh              131072 ctx
#   ./serve-qwen38-flash-next-q8-128k.sh --port 9000  override port
#   ./serve-qwen38-flash-next-q8-128k.sh --dry-run    print, no run
#
#   MEASURED 2026-09-03 (RESULTS 51), llama-sweep-bench, shallow:
#     2302.7 tok/s prefill, 40.6 t/s generation
#
# This is what `./serve.sh` with no arguments starts (config/default.env).
#
# Q8_0 leaves ~94 GiB of experts in host RAM, which is where the generation
# figure comes from -- the Q4_K_M sibling runs 128.9 t/s, 3.2x this, if 4-bit
# quality is acceptable. Prefill differs far less (3486 against 2303).
#
# NOT SOAKED. It became the default on 2026-09-03 on benchmark evidence only,
# validated to N_KV 75 776 of 131072.
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

exec ./serve.sh qwen38-flash-next-q8-128k "$@"
