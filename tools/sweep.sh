#!/usr/bin/env bash
# =============================================================================
# sweep.sh -- compare configurations of one profile, one server load each
# =============================================================================
#   ./tools/sweep.sh <profile> <depths> <arm> [arm ...]
#
#     <profile>  profile name, e.g. deepseek-v4-flash-gpu-experts-128k
#     <depths>   comma-separated, e.g. 4096,32768
#     <arm>      "label:VAR=VAL,VAR=VAL"  -- IK_* overrides for that arm
#
#   ./tools/sweep.sh deepseek-v4-flash-gpu-experts-128k 4096,32768 \
#       "baseline:" "no-nkvo n19:IK_NCMOE=19,IK_EXTRA_ARGS=-mla 3 -fidx"
#
# Placement and repack are applied at load time, so an arm genuinely needs its
# own server. That is the cost: ~35 s of load per arm from warm page cache.
#
# A configuration that will not load is a RESULT, not a failure -- an arm that
# OOMs is recorded as such and the sweep carries on. Half these questions are
# "does this even fit", and a sweep that dies on the first OOM cannot answer it.
#
# Everything else follows tools/depthbench.sh: max_tokens 160, temperature 0,
# a salt unique per request so the prompt cache cannot serve it, 2 repeats.
#
# Output: results/sweep-<profile>-<timestamp>.md
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

PROFILE="${1:?usage: sweep.sh <profile> <depths> <arm> [arm ...]}"; shift
DEPTHS="${1:?missing depths, e.g. 4096,32768}"; shift
[ $# -gt 0 ] || { echo "no arms given"; exit 1; }

PORT="${IK_PORT:-8090}"
STAMP=$(date +%Y%m%d-%H%M%S)
REPORT="results/sweep-${PROFILE}-${STAMP}.md"
mkdir -p results logs
: > "$REPORT.tmp"

echo "sweep: $PROFILE, depths $DEPTHS, $# arms"

for arm in "$@"; do
    label="${arm%%:*}"
    vars="${arm#*:}"
    log="logs/sweep-${STAMP}-$(echo "$label" | tr ' /' '__').log"

    ./stop.sh >/dev/null 2>&1
    for ((i=0;i<30;i++)); do ps -eo comm | grep -q '^llama-serv' || break; sleep 2; done

    # Each arm's overrides are exported into a subshell so they cannot leak into
    # the next arm -- an env var that survives would silently contaminate it.
    (
        if [ -n "$vars" ]; then
            # split on commas that separate VAR=VAL pairs; values may contain spaces
            IFS=',' read -ra kvs <<< "$vars"
            for kv in "${kvs[@]}"; do export "${kv?}"; done
        fi
        export IK_PORT="$PORT" IK_KILL_SQUATTERS=1
        ./serve.sh "$PROFILE" > "$log" 2>&1
    ) &

    ok=1
    for ((i=0;i<240;i++)); do
        grep -qa 'HTTP server listening' "$log" 2>/dev/null && { ok=0; break; }
        if [ "$i" -gt 6 ] && ! ps -eo comm | grep -q '^llama-serv'; then ok=2; break; fi
        sleep 5
    done

    # -ub above -b is clamped down SILENTLY, so an arm can measure a different
    # configuration than its label claims. This cost one wasted sweep on
    # 2026-08-19: a profile with no IK_BATCH line inherited -b 4096 from
    # default.env and quietly ran -ub 8192 as 4096, ~20 % of prefill at depth.
    if [ "$ok" -eq 0 ]; then
        want_ub=$(sed -n 's/.*IK_UBATCH=\([0-9]*\).*/\1/p' <<< "$vars")
        got_ub=$(grep -aoE 'n_ubatch += [0-9]+' "$log" | head -1 | grep -oE '[0-9]+$')
        if [ -n "$want_ub" ] && [ "$want_ub" != "$got_ub" ]; then
            echo "  $label: WARNING asked for -ub $want_ub, server reports $got_ub (clamped by -b?)"
        fi
    fi

    if [ "$ok" -ne 0 ]; then
        reason=$(grep -aoE 'out of memory|failed to allocate[^,]*|unknown argument[^ ]*' "$log" | head -1)
        printf '| %s | %s | — | — | %s |\n' "$label" "${vars:-(profile defaults)}" \
            "**did not load**${reason:+ — \`$reason\`}" >> "$REPORT.tmp"
        echo "  $label: did not load${reason:+ ($reason)}"
        continue
    fi

    python3 - "$label" "$vars" "$log" "$REPORT.tmp" "$PORT" "$DEPTHS" <<'PY'
import json, os, re, statistics, sys, urllib.request
label, vars_, log, out, port, depths = sys.argv[1:7]
depths = [int(d) for d in depths.split(",")]
api = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=3600):
    req = urllib.request.Request(api + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

ntok = lambda s: len(post("/tokenize", {"content": s}, 900)["tokens"])

def build(target, salt):
    line = lambda i: f"[{salt}] Record {i}: sensor {i%89} read {i*7%1000} at cycle {i%433}.\n"
    head = f"Corpus {salt}.\n"
    per = ntok("".join(line(i) for i in range(200))) / 200
    n = max(1, int((target - ntok(head) - 200) / per))
    body = head + "".join(line(i) for i in range(n))
    got = ntok(body)
    if got > target * 1.01 or got < target * 0.97:
        n = max(1, int(n * (target - ntok(head)) / max(1, got - ntok(head))))
        body = head + "".join(line(i) for i in range(n))
    return body

def measure(target, salt):
    t = post("/v1/chat/completions", {
        "model": "deepseek-v4-flash", "max_tokens": 160, "temperature": 0.0, "stream": False,
        "messages": [{"role": "user", "content": build(target, salt) + "\nReply with one sentence."}]
    })["timings"]
    return t["prompt_per_second"], t["predicted_per_second"]

try:
    measure(512, f"warm{os.getpid()}")
    pp_cells, tg_cells = [], []
    for d in depths:
        pf, gn = [], []
        for r in range(2):
            p, g = measure(d, f"{label}d{d}r{r}p{os.getpid()}")
            pf.append(p); gn.append(g)
        pp_cells.append(f"{statistics.mean(pf):.1f}")
        tg_cells.append(f"{statistics.mean(gn):.2f}")
        print(f"  {label}: depth {d} -> {statistics.mean(pf):7.1f} pp  {statistics.mean(gn):5.2f} tg", flush=True)
    lt = open(log, errors="replace").read()
    grab = lambda p, d="?": (re.findall(p, lt) or [d])[0]
    note = (f"cb {grab(r'CUDA0 compute buffer size\s*=\s*([\d.]+)')} MiB, "
            f"host {grab(r'CUDA_Host buffer size\s*=\s*([\d.]+)')} MiB")
    open(out, "a").write(f"| {label} | {vars_ or '(profile defaults)'} | "
                         f"{' / '.join(pp_cells)} | {' / '.join(tg_cells)} | {note} |\n")
except Exception as e:
    open(out, "a").write(f"| {label} | {vars_ or '(profile defaults)'} | — | — | "
                         f"**died mid-run** — `{repr(e)[:120]}` |\n")
    print(f"  {label}: died mid-run: {repr(e)[:120]}", flush=True)
    sys.exit(1)
PY
done

./stop.sh >/dev/null 2>&1

{
    echo "# sweep: \`$PROFILE\`"
    echo
    echo "Depths $DEPTHS. Two repeats each, \`max_tokens\` 160, temperature 0, unique salt"
    echo "per request. One fresh server per arm."
    echo
    echo "| arm | overrides | prefill tok/s | generation t/s | notes |"
    echo "|---|---|---|---|---|"
    cat "$REPORT.tmp"
} > "$REPORT"
rm -f "$REPORT.tmp"
echo
cat "$REPORT"
echo
echo "  -> $REPORT"
