#!/usr/bin/env bash
# =============================================================================
# serve-deepseek-v4-flash.sh -- one-command launch of DeepSeek-V4-Flash (Q8_K_XL)
# =============================================================================
# Thin convenience wrapper around:  ./serve.sh deepseek-v4-flash
#
#   ./serve-deepseek-v4-flash.sh                start at the profile defaults (262144 ctx)
#   ./serve-deepseek-v4-flash.sh --ctx 262144   raise context (MLA makes it cheap here)
#   ./serve-deepseek-v4-flash.sh --port 9000    override the port
#   ./serve-deepseek-v4-flash.sh --dry-run      print the command without running it
#
# DeepSeek-V4-Flash is ~151 GiB (deepseek4: MLA + DeepSeek Sparse Attention).
# This is the architecture ik_llama.cpp is built for -- `--fit` fills VRAM to
# ~94 GiB (LM Studio only reaches ~74), MLA keeps the KV cache tiny (~2.75 GiB
# at 65k), and the fused DSA indexer is enabled. Measured: ~17 tok/s generation,
# coherent output. See config/models/deepseek-v4-flash.env.
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
# DeepSeek (KV ~2.75 GiB at 65k), so raise this freely -- 262144 is realistic.
# Edit this line to change it; a --ctx flag or IK_CTX in the environment still
# wins over it. (This overrides the value in config/models/deepseek-v4-flash.env,
# so this wrapper is now the single place to set it.)
export IK_CTX="${IK_CTX:-262144}"

exec ./serve.sh deepseek-v4-flash "$@"
