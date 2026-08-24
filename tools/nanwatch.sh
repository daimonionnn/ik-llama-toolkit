#!/usr/bin/env bash
# =============================================================================
# nanwatch.sh -- desktop notification when the NaN abort fires
# =============================================================================
#   ./tools/nanwatch.sh [logfile]      default: newest logs/nancheck-soak*.log
#
# Since the sampler patch (RESULTS §33) the abort no longer kills the server: the
# request gets a 500, the slot's cache is dropped, and the next one answers
# normally. Which is the point -- and also the problem, because it is now
# invisible while you work. This watches for it and says so.
#
# It fires on either of two lines:
#   IK_NAN_CHECK: first NaN at node N   -- the probe, names the producing op
#   sampling failed, releasing slot     -- the sampler, the old signature
#
# The probe line comes first when IK_NAN_CHECK=1 is set, and it is the one worth
# reading: it names the op, tensor, type, shape and inputs.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

# Default to the newest logs/server-*.log, and ONLY that. A wrapper's own
# redirect (`./serve.sh > mylog`) is fed through tee's stdout, which is block
# buffered and can lag by hours -- watching one of those silently missed a real
# abort on 2026-08-23. tee writes logs/server-*.log directly, so it is current.
LOG="${1:-$(ls -t logs/server-*.log 2>/dev/null | head -1)}"
[ -n "$LOG" ] && [ -f "$LOG" ] || { echo "no log to watch"; exit 1; }

echo "  watching $LOG"
echo "  stop with: pkill -f 'nanwatc[h]\.sh'"

# --lines=0: only what arrives from now on. -F rather than -f so a server
# restart, which opens a new file, does not silently leave this watching a dead
# inode -- though it will still be the OLD path, so restart this too.
# RATE LIMITED, and it has to be: one abort produces ~162 probe lines, one per
# node in the poisoned graph. Firing per line meant 162 desktop popups for a
# single event, which had to be cleared by hand. 300 s of silence after a hit
# collapses a burst into one notification without hiding a genuinely separate
# abort later -- they are hours apart, not minutes.
LAST=0
tail -F --lines=0 "$LOG" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *"IK_NAN_CHECK: first NaN"*)
            now=$(date +%s)
            if [ $((now - LAST)) -ge 300 ]; then
                LAST=$now
                notify-send -u critical "ik-llama: NaN caught" \
                    "$line

Probe named the producing op. One notification per burst; details in ${LOG##*/}." 2>/dev/null
            fi
            printf '\n>>> %s\n' "$line"
            ;;
        *"CLEAN -> POISONED"*)
            notify-send -u critical "ik-llama: KV cache poisoned" \
                "The scan caught the poisoning decode. Details in ${LOG##*/}." 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
        *"IK_COPY_CHECK: HOST source already NaN"*)
            notify-send -u critical "ik-llama: host bytes already NaN" \
                "Poison predates the host->device copy. Details in ${LOG##*/}." 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
        *"IK_DST_CHECK:"*[!a]*)
            case "$line" in
                *alive*) ;;
                *) notify-send -u critical "ik-llama: device copy differs" \
                       "Destination read back does not match the source. Details in ${LOG##*/}." 2>/dev/null
                   printf '\n>>> %s\n' "$line" ;;
            esac
            ;;
        *"IK_LIFETIME:"*CHANGED*|*"IK_LIFETIME:"*"NOT copied"*|*"IK_LIFETIME:"*"ALREADY NaN"*)
            notify-send -u critical "ik-llama: lifetime probe fired" \
                "$line" 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
        *"IK_NAN_CHECK VERDICT"*)
            notify-send -u critical "ik-llama: VERDICT" "$line" 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
        *"sampling failed, releasing slot"*)
            notify-send -u critical "ik-llama: sampler abort" \
                "One request returned 500, slot cache dropped, server still up." 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
    esac
done
