#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh -- 262144 ctx, experts on GPU
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-gpu-experts-256k
#
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh              262144 ctx
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh --port 9000  override port
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh --dry-run    print, no run
#
# The 262144 sibling of the 131072 gpu-experts wrapper. Experts that do not fit in
# VRAM stay in host RAM but are COMPUTED ON THE GPU -- see RESULTS §21 for the
# mechanism, §27.3 for this profile's conversion.
#
#   MEASURED 2026-08-19, against deepseek-v4-flash-256k-kvram (the -rtr version):
#     depth 4k      451.2 pp ->  1 100.6 pp   (2.44x)
#     depth 32k     468.2    ->  1 636.9      (3.50x)
#     depth 128k    429.3    ->  1 258.8      (2.93x)
#   Generation costs ~10 %, mostly the three extra expert layers -ub 8192 needs at
#   this context. No quality trade: same weights, same quant.
#
# WHY 262144 COSTS MORE THAN 131072 DOES. The CUDA compute buffer scales with
# CONTEXT as well as micro-batch -- 11 136 MiB here against 7040 at 131072, same
# -ub 8192 (§27.2). That forces --n-cpu-moe up to 21: 20 dies at depth and 19 does
# not load. The 0.859 MiB-per-unit rule from §18.3 is a 131072-only law.
#
# If it misbehaves, deepseek-v4-flash-256k-kvram is the same context on the -rtr
# path -- slower, but a different code path, which is the point of keeping it.
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

exec ./serve.sh deepseek-v4-flash-gpu-experts-256k "$@"
