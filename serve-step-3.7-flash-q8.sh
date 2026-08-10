#!/usr/bin/env bash
# =============================================================================
# serve-step-3.7-flash-q8.sh -- one-command launch of Step-3.7-Flash Q8_K_XL
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh step-3.7-flash-q8
#
#   ./serve-step-3.7-flash-q8.sh               start Q8 at its defaults (262144 ctx)
#   ./serve-step-3.7-flash-q8.sh --ctx 65536   shorter context -> a bit more speed
#   ./serve-step-3.7-flash-q8.sh --port 9000   any serve.sh flag is forwarded
#   ./serve-step-3.7-flash-q8.sh --dry-run     print the command without running it
#
# Q8_K_XL is ~195 GiB and runs at roughly half the speed of the default Q4
# (~13 vs ~26 tok/s here) -- it is the quality-reference model, not the daily
# driver. See config/models/step-3.7-flash-q8.env for the measured numbers.
#
# Because Q8 needs almost the whole GPU, this wrapper defaults to clearing any
# other process squatting on VRAM (LM Studio, Ollama). Set IK_KILL_SQUATTERS=0
# to keep them and only get a warning instead.
# -----------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

export IK_KILL_SQUATTERS="${IK_KILL_SQUATTERS:-1}"

# Default context, right here in the launch script. At 262144 the KV cache is
# ~24 GiB (this model has no MLA), so lowering it frees VRAM for more experts
# and a bit more speed. Edit this line to change it; a --ctx flag or IK_CTX in
# the environment still wins. (Overrides config/models/step-3.7-flash-q8.env, so
# this wrapper is the single place to set it.)
export IK_CTX="${IK_CTX:-262144}"

exec ./serve.sh step-3.7-flash-q8 "$@"
