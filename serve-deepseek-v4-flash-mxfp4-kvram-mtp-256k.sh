#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh -- 256k, generation-first
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-256k-kvram-mtp
#
#   ./serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh              262144 ctx
#   ./serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh --port 9000  override port
#   ./serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh --dry-run    print, no run
#
# The kvram-256k wrapper with MTP speculative decoding. MEASURED at 130k depth
# (RESULTS §15):
#
#   kvram-256k (--n-cpu-moe 18)   406.1 tok/s prefill / 16.31 generation
#   this one   (n20 + MTP)        364.2 tok/s prefill / 20.48 generation
#
# +25.6 % generation for -10.3 % prefill — a better trade than the same
# treatment at 131072, because generation is more bandwidth-starved this deep.
# Same lossless MXFP4 weights; drafted tokens are verified against the target,
# so output is unchanged.
#
# PICK BY WORKLOAD, at this context:
#   long prompts / RAG / documents -> serve-deepseek-v4-flash-mxfp4-kvram-256k.sh
#   long answers / agents / chat   -> this one
# If 131072 is enough window, both 128k wrappers are faster on every axis.
#
# Inherits the kvram caveats: manual placement, and -rtr forces --no-mmap so
# each start re-reads the model. Needs the 5.5 GiB predictor from
# philpax/DeepSeek-V4-Flash-MTP-Only-GGUF.
#
# Two deliberate defaults (same as the other wrappers):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by NVIDIA compute processes only.
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh deepseek-v4-flash-256k-kvram-mtp "$@"
