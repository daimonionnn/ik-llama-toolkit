#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-kvram-128k.sh -- the fast-prefill 128k config
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-128k-kvram
#
#   ./serve-deepseek-v4-flash-mxfp4-kvram-128k.sh              131072 ctx
#   ./serve-deepseek-v4-flash-mxfp4-kvram-128k.sh --port 9000  override the port
#   ./serve-deepseek-v4-flash-mxfp4-kvram-128k.sh --dry-run    print, do not run
#
# Same MXFP4 quant and same 131072 context as
# serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh -- only the placement differs.
# The KV lives in RAM (`kvram` in the name), the VRAM it would have taken goes to
# experts, and the CPU-side experts are repacked for the AVX2/VNNI kernels.
#
#   MEASURED at 32k depth (RESULTS §11), against the --fit wrapper as shipped:
#     --fit wrapper     287.1 tok/s prefill / 19.75 generation
#     this one          484.1 tok/s prefill / 20.98 generation
#   i.e. +69 % prefill and +6 % generation, with the prompt cache still on so a
#   re-send is free. Identical weights, identical KV precision, no -ser: there is
#   no quality trade here, only a placement one.
#
# THE TRADE IS ROBUSTNESS, not quality:
#   * manual placement, so no --fit adaptivity, and only ~1.6 GiB of VRAM is left
#     free -- anything else on this GPU will break the load;
#   * -rtr forces --no-mmap, so each start re-reads the model (~30 s warm, longer
#     from cold disk).
# If a server just has to come up unattended, use the --fit wrapper instead.
#
# Two deliberate defaults (same as the other wrappers):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by a GPU-resident LM Studio/Ollama
#     model. (It only ever sees NVIDIA processes, so an LM Studio on a ROCm card
#     is not affected.)
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh deepseek-v4-flash-128k-kvram "$@"
