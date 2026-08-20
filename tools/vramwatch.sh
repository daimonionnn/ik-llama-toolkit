#!/usr/bin/env bash
# =============================================================================
# vramwatch.sh -- sample VRAM once a second, so an abort can be priced afterwards
# =============================================================================
#   ./tools/vramwatch.sh [interval_seconds]      default 1
#
# Written to answer one question: does the NaN abort (RESULTS §32) coincide with
# VRAM running out? The evidence so far says no -- no CUDA error appears in any
# of the ten crash logs, and the run with the LARGEST headroom aborted too -- but
# every one of those numbers is from load time. Nobody has looked at the moment
# of the abort itself, because nothing was recording it.
#
# One second, not five: a micro-batch at -ub 8192 takes about five seconds, so a
# five-second sample can sit entirely inside one and miss its peak. At 1 s this
# writes ~5 MB a day, which is nothing next to the logs it sits beside.
#
# Two rows per sample:
#   gpu,<iso>,<used MiB>,<free MiB>,<total MiB>,<gpu util %>
#   app,<iso>,<pid>,<used MiB>,<process name>
# The app rows matter as much as the gpu rows: if something else grabs VRAM --
# LM Studio, Ollama, a browser compositor -- this is what will show it, and that
# is a far more plausible squeeze than llama-server growing on its own.
#
# Reading it back after an abort:
#   ts=$(grep -a 'Failed to sample token' -B2 logs/server-<stamp>.log \
#        | grep -aoE 'timestamp=[0-9]+' | tail -1 | cut -d= -f2)
#   date -d @$ts   # then look at the rows either side of that time
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

INTERVAL="${1:-1}"
mkdir -p logs
OUT="logs/vram-$(date +%Y%m%d-%H%M%S).csv"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }

echo "# vramwatch, every ${INTERVAL}s, started $(date --iso-8601=seconds)" > "$OUT"
echo "# gpu,<iso>,<used MiB>,<free MiB>,<total MiB>,<util %>" >> "$OUT"
echo "# app,<iso>,<pid>,<used MiB>,<name>" >> "$OUT"

echo "  logging to $OUT  (stop with: pkill -f 'vramwatc[h]\.sh')"

# Read-only throughout: nothing here can disturb the server being watched.
while true; do
    now=$(date --iso-8601=seconds)
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null \
        | sed "s/^/gpu,$now,/;s/, */,/g" >> "$OUT"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name \
               --format=csv,noheader,nounits 2>/dev/null \
        | sed "s/^/app,$now,/;s/, */,/g" >> "$OUT"
    sleep "$INTERVAL"
done
