#!/usr/bin/env bash
# =============================================================================
# serve-...-gpu-experts-128k-ub1024.sh -- the gpu-experts profile at -ub 1024
# =============================================================================
# Same code path as serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh, with one
# variable changed: -ub 1024 instead of 8192, and --n-cpu-moe re-floored to 17.
#
# It is an experiment. All thirteen NaN aborts happened at -ub >= 2048 and none
# below; this accumulates evidence for or against that on the same code path the
# default uses. See config/models/deepseek-v4-flash-gpu-experts-128k-ub1024.env
# for the numbers and TODO 16 for why it is not yet a finding.
#
# Costs about four fifths of prefill (369 vs 1814 tok/s at 32k) and gains ~8 % of
# generation (20.16 vs 18.59). If that trade is too steep, -ub 4096 also has a
# clean record -- 1.6 M prefilled tokens, no aborts -- at roughly 15 % of prefill
# rather than 80 %:
#
#   IK_UBATCH=4096 IK_NCMOE=18 ./serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"
export IK_PORT="${IK_PORT:-8090}"

exec ./serve.sh deepseek-v4-flash-gpu-experts-128k-ub1024 "$@"
