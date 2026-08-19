#!/usr/bin/env bash
# =============================================================================
# ckpt-value.sh -- do context checkpoints pay for themselves on YOUR traffic?
# =============================================================================
#   ./tools/ckpt-value.sh [logfile ...]        default: newest logs/server-*.log
#
# RESULTS §30 measured what checkpoints COST: 11-27 % of prefill throughput and
# ~8 % of generation, all of it in checkpoint creation (the prompt cache is free).
# §11.3 measured what they BUY: a re-send that would otherwise re-prefill.
#
# Which side wins is a property of the workload, not of the toolkit, so it has to
# be read off real traffic. This does that from the server's own log, in seconds
# rather than percentages:
#
#   cost    = time spent in "created context checkpoint ... took N ms"
#   benefit = for each "restored context checkpoint ... n_past = N", the N tokens
#             that did not have to be prefilled, priced at the run's own measured
#             prefill rate, minus the restore's own cost
#
# A caution the raw numbers do not carry: a straight continuation of a
# conversation does NOT need a checkpoint -- the KV is still in the slot and the
# reuse is free. Checkpoints only earn their keep when the prompt DIVERGES from
# what is cached, which is why "n_past_prompt" summed over requests badly
# overstates their value. Only "restored context checkpoint" lines count.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

LOGS=("$@")
[ ${#LOGS[@]} -eq 0 ] && LOGS=("$(ls -t logs/server-*.log 2>/dev/null | head -1)")
[ -z "${LOGS[0]:-}" ] && { echo "no server log found"; exit 1; }

python3 - "${LOGS[@]}" <<'PY'
import re, sys

tot_created = tot_create_ms = 0
tot_restored = tot_restore_ms = tot_tokens_saved = 0
tot_scratch = 0
pf_tokens = pf_ms = 0.0

for path in sys.argv[1:]:
    try:
        t = open(path, errors="replace").read()
    except OSError:
        continue
    for ms in re.findall(r'created context checkpoint[^)]*took ([\d.]+) ms', t):
        tot_created += 1; tot_create_ms += float(ms)
    for ms, npast in re.findall(r'restored context checkpoint took +([\d.]+) ms[^)]*n_past = (\d+)', t):
        tot_restored += 1; tot_restore_ms += float(ms); tot_tokens_saved += int(npast)
    tot_scratch += len(re.findall(r'reprocessing from scratch', t))
    for ms, n in re.findall(r'prompt eval time = *([\d.]+) ms / *(\d+) tokens', t):
        pf_ms += float(ms); pf_tokens += int(n)

print(f"  logs analysed        : {len(sys.argv)-1}")
if not pf_tokens:
    print("  no prefill recorded -- nothing to judge"); sys.exit()

rate = pf_tokens / (pf_ms / 1000.0)
print(f"  measured prefill rate: {rate:.0f} tok/s\n")
print(f"  checkpoints created  : {tot_created:5}   costing {tot_create_ms/1000:8.1f} s")
print(f"  checkpoints restored : {tot_restored:5}   costing {tot_restore_ms/1000:8.1f} s")
print(f"  reprocessed from zero: {tot_scratch:5}   (a checkpoint was wanted and missing)")

saved_s = tot_tokens_saved / rate
net = saved_s - (tot_create_ms + tot_restore_ms) / 1000.0
print(f"\n  prefill avoided      : {tot_tokens_saved} tokens = {saved_s:.1f} s")
print(f"  net                  : {net:+.1f} s")

if tot_created:
    print(f"  restore rate         : {100*tot_restored/tot_created:.1f} % of checkpoints were ever used")
print()
if net > 0:
    print(f"  VERDICT: checkpoints pay for themselves here (+{net:.0f} s). Keep -ctx-ckpt on.")
else:
    print(f"  VERDICT: checkpoints cost {-net:.0f} s more than they saved on this traffic.")
    print(f"           -ctx-ckpt 0 is worth 11-27 % of prefill on top of that (RESULTS §30).")
print("\n  Note: this counts ONLY checkpoint restores. Ordinary conversation")
print("  continuation reuses the slot's KV for free and is not attributable to")
print("  checkpoints, so do not compare this against n_past_prompt totals.")
PY
