#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh -- lossless, generation-first
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-128k-kvram-mtp
#
#   ./serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh              131072 ctx
#   ./serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh --port 9000  override port
#   ./serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh --dry-run    print, no run
#
# The kvram-128k wrapper with MTP speculative decoding added, which trades
# prefill for generation (RESULTS §14.1, 32k depth):
#
#   kvram-128k (--n-cpu-moe 17)   499.2 tok/s prefill / 21.26 generation
#   this one   (n19 + MTP)        434.5 tok/s prefill / 24.12 generation
#
# +13.5 % generation, -13 % prefill. Same lossless MXFP4 weights, and drafted
# tokens are verified against the target, so output is unchanged either way.
#
# PICK BY WORKLOAD:
#   long prompts / RAG / documents -> serve-deepseek-v4-flash-mxfp4-kvram-128k.sh
#   long answers / agents / chat   -> this one
#   raw speed, 2-bit acceptable    -> serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh
#
# Inherits the kvram caveats: manual placement (no --fit adaptivity), and -rtr
# forces --no-mmap so each start re-reads the model. Needs the 5.5 GiB predictor
# from philpax/DeepSeek-V4-Flash-MTP-Only-GGUF.
#
# Two deliberate defaults (same as the other wrappers):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by NVIDIA compute processes only.
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh deepseek-v4-flash-128k-kvram-mtp "$@"
