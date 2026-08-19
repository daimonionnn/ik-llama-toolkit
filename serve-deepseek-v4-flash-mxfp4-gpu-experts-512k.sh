#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh -- 524288 ctx, experts on GPU
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash-512k
#
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh              524288 ctx
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh --port 9000  override port
#   ./serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh --dry-run    print, no run
#
# Renamed from serve-...-gpu-cpu-512k.sh on 2026-08-19: the profile was converted
# to the gpu-experts principle (RESULTS §29) and the old name described what it
# used to do. Experts that do not fit in VRAM stay in host RAM but are COMPUTED
# ON THE GPU -- see §21 for the mechanism.
#
#   MEASURED 2026-08-19, against the same profile before conversion:
#     depth 4k      309.9 pp ->  1 068.9 pp   (3.4x)
#     depth 32k     323.1    ->  1 721.3      (5.3x)
#     depth 128k    301.7    ->  1 334.9      (4.4x)
#   The largest gain measured on any single profile here. The old settings ran
#   -ub 512, which is far too small for weight streaming to amortise over -- its
#   prefill was flat at ~300 tok/s regardless of depth.
#
# WHY IT COSTS MORE THAN THE SMALLER CONTEXTS. The CUDA compute buffer scales with
# context: 21 376 MiB here at -ub 8192, three times the 131072 figure for the same
# batch (§29.4). That forces --n-cpu-moe up to 25 -- 22 and 23 do not load, and 24
# loads, serves a 4k prompt and then dies at 32k. Generation pays for those layers.
#
# CHECKPOINTS ARE OFF HERE, DELIBERATELY (-ctx-ckpt 0 in the profile). At 524288 a
# single checkpoint is ~3487 MiB and costs 28 % of prefill and 13 % of generation
# to create (§31) -- twice what it costs at 262144. Leaving the default 32 also
# reached 231 GiB RSS during a sweep on 2026-08-19 and the OOM killer took the
# server and the user's editor with it. The 262144 profile keeps checkpoints with
# a --cache-ram ceiling instead; here they are simply not worth it.
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

exec ./serve.sh deepseek-v4-flash-512k "$@"
