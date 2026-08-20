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

LOG="${1:-$(ls -t logs/nancheck-soak*.log logs/server-*.log 2>/dev/null | head -1)}"
[ -n "$LOG" ] && [ -f "$LOG" ] || { echo "no log to watch"; exit 1; }

echo "  watching $LOG"
echo "  stop with: pkill -f 'nanwatc[h]\.sh'"

# --lines=0: only what arrives from now on. -F rather than -f so a server
# restart, which opens a new file, does not silently leave this watching a dead
# inode -- though it will still be the OLD path, so restart this too.
tail -F --lines=0 "$LOG" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *"IK_NAN_CHECK: first NaN"*)
            notify-send -u critical "ik-llama: NaN caught" \
                "$line

The probe named the producing op. Details in ${LOG##*/}." 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
        *"sampling failed, releasing slot"*)
            notify-send -u critical "ik-llama: sampler abort" \
                "One request returned 500, slot cache dropped, server still up." 2>/dev/null
            printf '\n>>> %s\n' "$line"
            ;;
    esac
done
