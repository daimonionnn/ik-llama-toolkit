#!/usr/bin/env bash
# =============================================================================
# stop.sh -- stop whatever model this toolkit is serving (any profile, any port)
# =============================================================================
#   ./stop.sh          stop the ik_llama server(s) launched from this toolkit
#   ./stop.sh --all     also evict GPU-resident LM Studio / Ollama models
#
# Targets only this toolkit's own server binary by its absolute path, so it will
# never touch LM Studio, Ollama, or anything else -- unless you pass --all, which
# additionally frees VRAM from any other process holding it.
# -----------------------------------------------------------------------------
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/lib/common.sh"

want_all=0
case "${1:-}" in
    --all|-a) want_all=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    "") ;;
    *) die "unknown option: $1 (use --all, or nothing)" ;;
esac

before="$(gpu_free_mib)"
log "VRAM free before: ${before:-?} MiB"

# --- 1. this toolkit's own llama-server (any port / profile) -----------------
# Matched by its absolute path, so LM Studio / Ollama are never affected here.
server_running() { pgrep -f "$IK_BIN/llama-server" >/dev/null 2>&1; }

if server_running; then
    pids="$(pgrep -f "$IK_BIN/llama-server" | tr '\n' ' ')"
    log "stopping ik_llama server (pids: $pids)"
    pkill -TERM -f "$IK_BIN/llama-server" 2>/dev/null
    for _ in $(seq 1 15); do server_running || break; sleep 1; done
    if server_running; then
        warn "still alive after SIGTERM -- forcing (SIGKILL)"
        pkill -KILL -f "$IK_BIN/llama-server" 2>/dev/null
        sleep 2
    fi
    server_running && die "could not stop the server" || ok "ik_llama server stopped"
else
    dim "no ik_llama server from this toolkit is running"
fi

# --- 2. optionally evict other GPU-resident models ---------------------------
if [[ $want_all == 1 ]]; then
    squatters="$(gpu_squatters)"
    if [[ -n $squatters ]]; then
        log "--all: evicting other processes holding VRAM:"
        printf '%s\n' "$squatters" | sed 's/^/       /' >&2
        while IFS=$'\t' read -r pid _ _; do
            [[ -n $pid ]] && kill "$pid" 2>/dev/null && dim "  sent SIGTERM to $pid"
        done <<< "$squatters"
        sleep 4
    else
        dim "--all: no other process holding VRAM"
    fi
fi

# --- report ------------------------------------------------------------------
sleep 1
after="$(gpu_free_mib)"
ok "VRAM free now: ${after:-?} MiB (of $(gpu_total_mib) MiB)"
