#!/usr/bin/env bash
# =============================================================================
# depthbench.sh -- prefill and generation vs prompt depth, through the API
# =============================================================================
#   ./tools/depthbench.sh <ubatch> [-r N] [--attach] [depth ...]
#
#     <ubatch>    -ub to run with; the server is restarted to apply it
#     -r N        repeats per depth (default 2) -- the noise floor is ~2 %, so a
#                 single sample cannot resolve small differences (RESULTS §16)
#     --attach    measure the server that is ALREADY running and leave it alone.
#                 <ubatch> is then only a label -- it is verified against the log
#                 and the run aborts on a mismatch. Use this to measure without
#                 restarting something you are watching for a crash.
#     depth ...   default 512 4096 32768 128000
#
# This is the tool that produced RESULTS §17, §18 and §19, so it lives in the
# repo rather than in a scratch directory: those sections quote its methodology,
# and results are only comparable if the next run uses the same one.
#
# Measures through llama-server rather than llama-bench, because the question is
# what the SERVED configuration does: -ub interacts with the prompt cache and
# with KV placement in ways llama-bench does not reproduce.
#
# METHODOLOGY, fixed here so results stay comparable (RESULTS §15.1 records what
# happened when it was not):
#   * max_tokens = 160, temperature 0. §12.1 used 64 and §14/§15 used 160, which
#     left those generation figures incomparable.
#   * 32768 is in the default set on purpose: it is the depth §9/§11 tuned at, so
#     it is the only point that can be held against the existing results.
#   * every prompt carries a salt unique to (ubatch, depth, repeat, pid). Reusing
#     a salt would let the prompt cache serve the prompt and report a prefill
#     rate that is really a cache hit.
#   * prompts are built, tokenized, then corrected once to land within 1 % of
#     target. Overshooting the context window makes the server return 500.
#   * one warm-up request precedes the measurements and is discarded.
#   * if the server dies mid-run, the script says which depth and repeat it died
#     on rather than reporting a partial table as if it were complete.
#
# Output: results/depthbench-ub<N>-<timestamp>.md, self-describing -- it records
# the placement, buffer sizes and flags it actually measured, read back from the
# server log, not from what was requested.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
TOOLKIT_ROOT="$PWD"

UB=""; REPEATS=2; ATTACH=0; DEPTHS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -r)        REPEATS="$2"; shift 2 ;;
        --attach)  ATTACH=1; shift ;;
        *)         [ -z "$UB" ] && UB="$1" || DEPTHS+=("$1"); shift ;;
    esac
done
[ -z "$UB" ] && { echo "usage: tools/depthbench.sh <ubatch> [-r N] [--attach] [depth ...]"; exit 1; }
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(512 4096 32768 128000)

PORT="${IK_PORT:-8090}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/depthbench-ub${UB}-${STAMP}.log"
REPORT="results/depthbench-ub${UB}-${STAMP}.md"
mkdir -p results logs

if [ "$ATTACH" -eq 1 ]; then
    LOG=$(ls -t logs/server-*.log | head -1)
    ps -eo comm | grep -q '^llama-serv' || { echo "no server is running -- drop --attach"; exit 1; }
    got=$(grep -aoE 'n_ubatch += [0-9]+' "$LOG" | head -1 | grep -oE '[0-9]+$')
    [ "$got" = "$UB" ] || { echo "running -ub $got, not $UB -- stopping so the result is not mislabelled"; exit 1; }
    echo "measuring the running server, -ub $UB (log ${LOG##*/})"
else
    ./stop.sh >/dev/null 2>&1
    for ((i=0;i<30;i++)); do ps -eo comm | grep -q '^llama-serv' || break; sleep 2; done
    IK_UBATCH="$UB" IK_PORT="$PORT" IK_KILL_SQUATTERS=1 ./serve.sh > "$LOG" 2>&1 &
    # The liveness check must tolerate the first seconds, while serve.sh has been
    # backgrounded but has not exec'd llama-server yet.
    for ((i=0;i<240;i++)); do
        grep -qa 'HTTP server listening' "$LOG" && break
        if [ "$i" -gt 6 ] && ! ps -eo comm | grep -q '^llama-serv'; then
            echo "server failed to start"; tail -5 "$LOG"; exit 1
        fi
        sleep 5
    done
    echo "server up, -ub $UB (load took $((i*5)) s)"
fi

python3 - "$UB" "$REPEATS" "$LOG" "$REPORT" "$PORT" "${DEPTHS[@]}" <<'PY'
import json, os, re, statistics, sys, urllib.request

ub, repeats, log, report, port = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
depths = [int(d) for d in sys.argv[6:]]
api = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=3600):
    req = urllib.request.Request(api + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

ntok = lambda s: len(post("/tokenize", {"content": s}, 900)["tokens"])

def build(target, salt):
    line = lambda i: f"[{salt}] Record {i}: sensor {i%89} read {i*7%1000} at cycle {i%433}.\n"
    head = f"Corpus {salt}, revision {salt}.\n"
    per = ntok("".join(line(i) for i in range(200))) / 200
    n = max(1, int((target - ntok(head) - 200) / per))
    body = head + "".join(line(i) for i in range(n))
    got = ntok(body)
    if got > target * 1.01 or got < target * 0.97:
        n = max(1, int(n * (target - ntok(head)) / max(1, got - ntok(head))))
        body = head + "".join(line(i) for i in range(n))
    return body

def measure(target, salt):
    body = build(target, salt)
    t = post("/v1/chat/completions", {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": body + "\nReply with a one-sentence summary."}],
        "max_tokens": 160, "temperature": 0.0, "stream": False})["timings"]
    return t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"]

lt = open(log, errors="replace").read()
grab = lambda p, d="-": (re.findall(p, lt) or [d])[0]
cfg = {
    "n_ubatch":       grab(r"n_ubatch += (\d+)"),
    "n_batch":        grab(r"n_batch += (\d+)"),
    "n_ctx":          grab(r"n_ctx += (\d+)"),
    "-rtr":           "on" if "Repacked" in lt else "off",
    "prompt cache":   "off" if "prompt cache is disabled" in lt else "on",
    "CUDA0 weights":  grab(r"CUDA0 buffer size\s*=\s*([\d.]+)") + " MiB",
    "host weights":   grab(r"CUDA_Host buffer size\s*=\s*([\d.]+)") + " MiB",
    "compute buffer": grab(r"CUDA0 compute buffer size\s*=\s*([\d.]+)") + " MiB",
}

measure(512, f"warm{os.getpid()}")

rows, died = [], None
for d in depths:
    pf, gn, last_pn = [], [], d
    for r in range(repeats):
        try:
            pn, pps, gps = measure(d, f"ub{ub}d{d}r{r}p{os.getpid()}")
        except Exception as e:
            died = (d, r, repr(e)[:200]); break
        pf.append(pps); gn.append(gps); last_pn = pn
        print(f"  depth {pn:>7}  repeat {r+1}/{repeats}   prefill {pps:7.1f} tok/s   generation {gps:6.2f} t/s",
              flush=True)
    if died: break
    spread = (max(pf) - min(pf)) / statistics.mean(pf) * 100 if len(pf) > 1 else 0.0
    rows.append((last_pn, statistics.mean(pf), statistics.mean(gn), spread))

with open(report, "w") as f:
    f.write(f"# depthbench, `-ub {ub}`\n\n")
    f.write(f"{repeats} repeat(s) per depth, `max_tokens` 160, temperature 0, unique salt per\n"
            f"request so nothing is served from the prompt cache.\n\n")
    f.write("## Configuration measured\n\n(read back from the server log, not from what was asked for)\n\n")
    f.write("| setting | value |\n|---|---|\n")
    for k, v in cfg.items():
        f.write(f"| `{k}` | {v} |\n")
    f.write("\n## Results\n\n| depth (tokens) | prefill tok/s | generation t/s | prefill spread |\n")
    f.write("|---:|---:|---:|---:|\n")
    for pn, pps, gps, spread in rows:
        f.write(f"| {pn} | **{pps:.1f}** | **{gps:.2f}** | {spread:.1f} % |\n")
    if died:
        d, r, err = died
        f.write(f"\n## INCOMPLETE\n\nThe server died at depth {d}, repeat {r+1}: `{err}`\n\n"
                f"The rows above are still valid; the depths below it were never run.\n")

print(flush=True)
for pn, pps, gps, spread in rows:
    extra = f"   (spread {spread:.1f} %)" if spread else ""
    print(f"  {pn:>7} tokens   prefill {pps:7.1f} tok/s   generation {gps:6.2f} t/s{extra}", flush=True)
if died:
    d, r, err = died
    print(f"\n  SERVER DIED at depth {d}, repeat {r+1}: {err}", flush=True)
print(f"\n  -> {report}", flush=True)
sys.exit(2 if died else 0)
PY
rc=$?
[ "$rc" -eq 2 ] && echo "INCOMPLETE: the server died during measurement, see $REPORT"
exit "$rc"
