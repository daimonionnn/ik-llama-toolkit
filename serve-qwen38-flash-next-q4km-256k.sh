#!/usr/bin/env bash
# =============================================================================
# serve-qwen38-flash-next-q4km-256k.sh -- Qwen3.8-Flash-Next Q4_K_M, 262144 ctx
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh qwen38-flash-next-q4km-256k
#
#   ./serve-qwen38-flash-next-q4km-256k.sh              262144 ctx
#   ./serve-qwen38-flash-next-q4km-256k.sh --port 9000  override port
#   ./serve-qwen38-flash-next-q4km-256k.sh --dry-run    print, no run
#
#   MEASURED 2026-09-03 (RESULTS 51), llama-sweep-bench, shallow:
#     3341.7 tok/s prefill, 128.6 t/s generation
#
# 262144 is nearly free on this model: the hybrid keeps a KV cache on only every
# fourth layer, so doubling the window adds 2 112 MiB, not 12 GiB. Costs ~4 %
# prefill (the smaller -ub 1024 the compute buffer forces) and nothing measurable
# on generation.
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

exec ./serve.sh qwen38-flash-next-q4km-256k "$@"
