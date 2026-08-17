#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh -- experts computed on the GPU
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-gpu-experts-128k
#
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh              131072 ctx
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh --port 9000  override port
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh --dry-run    print, no run
#
# THIS ONE WORKS ON A DIFFERENT PRINCIPLE from the other wrappers. They all decide
# where the weights LIVE; this decides who COMPUTES them.
#
#   serve-...-gpu-cpu-128k.sh   --fit placement, experts on the CPU
#   serve-...-kvram-128k.sh     manual placement, KV in RAM, experts on the CPU
#   THIS ONE                    manual placement, KV in RAM, experts ON THE GPU
#
# The experts that do not fit in VRAM still sit in host RAM, but they are shipped
# across PCIe each forward pass and the GEMM runs on the GPU. What enables it is
# simply NOT passing -rtr: that flag repacks them into MXFP4_R8, a CPU-only type
# with no CUDA kernel, which pins the work to the processor.
#
#   MEASURED 2026-08-17 at 131072 ctx, after tuning (RESULTS §21-§22):
#     depth 4 101     kvram 478.0 pp   ->  this 1 329.4 pp   (2.78x)
#     depth 32 701    kvram 486.4 pp   ->  this 1 793.6 pp   (3.69x)
#     depth 127 981   kvram 436.7 pp   ->  this 1 327.7 pp   (3.04x)
#   Generation is 3-7 % lower. No quality trade: same weights, same quant.
#
# WHAT IT NEEDS, and when to prefer the others:
#   * a wide link. It moves weights, not just activations, so PCIe is central --
#     on a gen4 x4 slot this strategy collapses and -rtr wins instead.
#   * a GPU faster than the CPU at quantised GEMM. True here (Blackwell vs an
#     Arrow Lake with no AVX-512); a Zen 4/5 with AVX-512 could flip it back.
#
# NOT the shipped default yet -- but only one thing is missing now. Placement,
# batching, KV placement and threads have all been swept (RESULTS §22). What has
# not happened is a stability soak: the NaN-logits abort (RESULTS §19) was chased
# entirely in the -rtr path at -ub 2048, and this is a different path at -ub 8192.
# Use it deliberately and watch it; see TODO item 11.
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

exec ./serve.sh deepseek-v4-flash-gpu-experts-128k "$@"
