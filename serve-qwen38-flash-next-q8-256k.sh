#!/usr/bin/env bash
# =============================================================================
# serve-qwen38-flash-next-q8-256k.sh -- Qwen3.8-Flash-Next Q8_0, 262144 ctx
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh qwen38-flash-next-q8-256k
#
#   ./serve-qwen38-flash-next-q8-256k.sh              262144 ctx
#   ./serve-qwen38-flash-next-q8-256k.sh --port 9000  override port
#   ./serve-qwen38-flash-next-q8-256k.sh --dry-run    print, no run
#
#   MEASURED 2026-09-03 (RESULTS 51), llama-sweep-bench, shallow:
#     2162.7 tok/s prefill, 36.9 t/s generation
#
# The full advertised window at Q8 quality. Costs 6.1 % prefill and 9.1 %
# generation against the 128k profile: the extra KV plus a compute buffer going
# 6996 -> 12 628 MiB has to come out of resident weights (-ncmoe 17 -> 20).
#
# Watch host RAM the first time this sees a genuinely deep prompt -- ~104 GiB of
# weights live there, and RESULTS 26.3 records an OOM kill at 203 GiB RSS.
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

exec ./serve.sh qwen38-flash-next-q8-256k "$@"
