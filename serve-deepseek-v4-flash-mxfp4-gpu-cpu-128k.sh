#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh -- DeepSeek-V4-Flash, GPU+CPU
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash
#
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh              131072 ctx
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh --ctx 262144 full trained window
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh --port 9000  override the port
#   ./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh --dry-run    print, do not run
#
# The exact quant this serves (set in config/models/deepseek-v4-flash.env):
#   lmstudio-community/DeepSeek-V4-Flash-0731-GGUF
#   DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf   (~145.6 GiB, 4-file split)
# QAT with native MXFP4 experts, so this quant is effectively lossless.
#
# GPU+CPU, hence the name. At ~146 GiB the model does NOT fit the 96 GiB card:
# `--fit` fills VRAM to ~90 GiB and the remaining ~55-60 GiB of routed experts
# live in DDR5 and are computed on the CPU. That spill is what pins generation
# to ~20 tok/s -- it is DDR5 bandwidth bound, not GPU bound. Prefill is still
# fast (~360-385 tok/s) and quality is the highest of the DeepSeek options here.
#
# Want speed instead of quality? ./serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh
# runs an 81 GiB 2-bit quant entirely in VRAM plus an MTP draft: ~87 tok/s at
# 65k. The trade is real: ~20 tok/s lossless vs ~87 tok/s at 2-bit.
#
# deepseek4 architecture: MLA + DeepSeek Sparse Attention, 1 048 576 trained ctx.
# This is what ik_llama.cpp is built for -- `--fit` fills VRAM properly (LM Studio
# only reaches ~74 GiB), MLA keeps the KV cache tiny (~2.75 GiB at 65k, ~5.5 GiB
# at 128k), and the fused DSA indexer (-fidx) is enabled.
# See config/models/deepseek-v4-flash.env and docs/RESULTS.md.
#
# Two deliberate defaults:
#   * Port 8090, not 8080 -- LM Studio's own API server usually holds 8080, and
#     ik_llama can't bind a port already in use. Override with --port if needed.
#   * IK_KILL_SQUATTERS=1 -- frees any model squatting on VRAM (a GPU-resident
#     LM Studio / Ollama model). LM Studio's CPU-only API server is left alone.
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

# Default context, right here in the launch script. MLA makes context cheap for
# DeepSeek (KV ~5.5 GiB at 128k), and every GiB of KV saved is a GiB of experts
# that stays on the GPU instead of DDR5 -- so a lower ctx is also slightly faster
# here. 262144 still works if you need it (profile margin is tuned for it).
# Edit this line to change it; a --ctx flag or IK_CTX in the environment still
# wins over it. (This overrides the value in config/models/deepseek-v4-flash.env,
# so this wrapper is now the single place to set it.)
export IK_CTX="${IK_CTX:-131072}"

# Fit margin, tuned for the 131072 above. The profile default is 8192, which was
# sized for 262144 -- at 128k the KV is ~5.5 GiB smaller, so 8192 leaves VRAM
# unused and pushes experts to DDR5 that would have fit. MEASURED here, one
# 256-token generation each at 131072 ctx:
#   margin 8192 -> 90 556 MiB VRAM, 19.66 tok/s   (profile default)
#   margin 4096 -> 93 814 MiB VRAM, 21.09 tok/s   (+7%, this line)
#   margin 2048 -> 97 074 MiB VRAM, 21.53 tok/s   (+9%, but only 813 MiB free)
# 4096 is the pick: 2048 is too tight to trust. Verified at depth -- a 94 015
# token prefill on margin 4096 grew VRAM by only 278 MiB (to 94 092) and did not
# OOM, at 282 tok/s prefill and 16.5 tok/s generation from that depth.
# Only applied at 131072 and below: at 262144 the profile's own 8192 is needed
# (the DSA caches and compute buffer are allocated after the fit decision, and
# 4096 does not cover them there), so a bigger --ctx keeps the profile default.
# A `--ctx N` flag is parsed by serve.sh, i.e. after this file runs, so look for
# it here too -- otherwise `--ctx 262144` would silently keep the 128k margin.
eff_ctx="$IK_CTX"
prev=""
for arg in "$@"; do
    [[ $prev == --ctx ]] && eff_ctx="$arg"
    prev="$arg"
done

if [[ $eff_ctx =~ ^[0-9]+$ && $eff_ctx -le 131072 ]]; then
    export IK_FIT_MARGIN="${IK_FIT_MARGIN:-4096}"
fi

exec ./serve.sh deepseek-v4-flash "$@"
