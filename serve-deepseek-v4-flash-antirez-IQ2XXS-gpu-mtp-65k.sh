#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh -- fastest DeepSeek-V4
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-mtp
#
#   ./serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh              65536 ctx, ~87 tok/s
#   ./serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh --ctx 32768  ~94 tok/s
#   ./serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh --port 9000  override the port
#   ./serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh --dry-run    print, do not run
#
# The exact quant this serves (set in config/models/deepseek-v4-flash-mtp.env):
#   antirez/deepseek-v4-gguf
#   DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
# i.e. IQ2_XXS routed experts (w2 at Q2_K) with Q8 attention projections, shared
# experts and output tensor, imatrix-calibrated, chat revision v2.
#
# GPU-only, hence the name: that ~81 GiB quant plus its KV and the draft fit
# entirely in the 96 GiB card, so no expert ever spills to DDR5 -- which is
# exactly why it is fast. This is the fastest coherent DeepSeek-V4 config
# measured on this box (docs/RESULTS.md section 7). Second idea stacked on top:
# a 5.5 GiB MTP predictor companion drafts tokens the target verifies in one
# pass. MEASURED: ~87 tok/s at 65536 ctx, ~94 tok/s at 32768 -- vs ~20 tok/s for
# the spilling MXFP4 quant and 37-40 tok/s for a dual DGX Spark.
#
# Requires the MTP predictor (5.5 GiB, one-time download):
#   huggingface.co/philpax/DeepSeek-V4-Flash-MTP-Only-GGUF
# Path is set in config/models/deepseek-v4-flash-mtp.env (IK_MTP_DRAFT).
#
# Context ceiling: with MTP on, 65536 is the max here -- the target+KV+draft just
# fill the 96 GiB (131072 CUDA-OOMs, VERIFIED). Need full 128k? Two ways:
#   * lossless, slow: ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh (~20 tok/s,
#     experts spill to DDR5), or
#   * this same antirez model without MTP -- point the plain `deepseek-v4-flash`
#     profile at it: it fits at 128k (~90 GiB) and does ~69 tok/s.
# The trade is real: 65k @ 87 tok/s vs 128k @ 69 tok/s.
#
# Two deliberate defaults (same as the gpu-cpu wrapper):
#   * Port 8090, not 8080 -- LM Studio's API server usually holds 8080.
#   * IK_KILL_SQUATTERS=1 -- frees VRAM held by a GPU-resident LM Studio/Ollama
#     model so the ~93 GiB budget is actually available.
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

# Default context, right here in the launch script. 65536 is the MTP ceiling on
# this GPU (the draft needs the VRAM that a bigger KV would eat). A --ctx flag or
# IK_CTX in the environment still wins over this line.
export IK_CTX="${IK_CTX:-65536}"

exec ./serve.sh deepseek-v4-flash-mtp "$@"
