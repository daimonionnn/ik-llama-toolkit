#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mtp-65k.sh -- fastest DeepSeek-V4-Flash: fit-in-VRAM + MTP
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-mtp
#
#   ./serve-deepseek-v4-flash-mtp-65k.sh              start at profile defaults (65536 ctx)
#   ./serve-deepseek-v4-flash-mtp-65k.sh --ctx 32768  lower context (more draft headroom)
#   ./serve-deepseek-v4-flash-mtp-65k.sh --port 9000  override the port
#   ./serve-deepseek-v4-flash-mtp-65k.sh --dry-run    print the command without running it
#
# This is the fastest coherent DeepSeek-V4 config measured on this box
# (docs/RESULTS.md section 7): the ~81 GiB antirez IQ2XXS quant fits entirely in
# VRAM (no DDR5 spill) and a 5.5 GiB MTP predictor companion drafts tokens on top.
# MEASURED: ~87 tok/s at 65536 ctx, ~94 tok/s at 32768 -- vs ~20 tok/s for the
# spilling MXFP4 quant and 37-40 tok/s for a dual DGX Spark.
#
# Requires the MTP predictor (5.5 GiB, one-time download):
#   huggingface.co/philpax/DeepSeek-V4-Flash-MTP-Only-GGUF
# Path is set in config/models/deepseek-v4-flash-mtp.env (IK_MTP_DRAFT).
#
# Context ceiling: with MTP on, 65536 is the max here -- the target+KV+draft just
# fill the 96 GiB (131072 CUDA-OOMs, VERIFIED). Need full 128k? Use the plain
# ./serve-deepseek-v4-flash.sh but point it at the antirez model: ~69 tok/s at 128k, no
# MTP. The trade is real: 65k @ 87 tok/s vs 128k @ 69 tok/s.
#
# Two deliberate defaults (same as serve-deepseek-v4-flash.sh):
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
