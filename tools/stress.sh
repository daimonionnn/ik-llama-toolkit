#!/usr/bin/env bash
# =============================================================================
# stress.sh -- try to reproduce the NaN-logits abort quickly
# =============================================================================
#   ./tools/stress.sh <profile> [--budget N] [--ub N] [--extra "ARGS"]
#
#     <profile>   profile to serve
#     --budget N  stop after N prefilled tokens without an abort (default 600000)
#     --ub N      override IK_UBATCH
#     --extra "…" override IK_EXTRA_ARGS wholesale (for the fused-kernel bisect)
#
# WHY THIS EXISTS. The abort (RESULTS §19, TODO 9) takes ~30 minutes of real
# traffic to appear, which makes bisecting it impractical -- every arm would cost
# half a day. This drives the one thing the crashing runs had and the benchmark
# runs did not: **prompts much shorter than -ub**, i.e. a single partial
# micro-batch per request.
#
#   crash 7   median prompt 790 tokens at -ub 8192, 60 of 70 below -ub -> abort
#   benchmark eight prompts of 32k-128k at -ub 2048, nearly all full -> clean
#
# So the prompts here are deliberately short and varied, and a few conversations
# grow across turns so context checkpoints are exercised too, as they were in
# every crashing run.
#
# OUTCOMES, both useful:
#   * abort inside ~100-330k prefilled tokens -> the partial micro-batch path is
#     implicated, there is now a fast test case, and the fused-kernel bisect
#     (-no-fmoe, no -fidx, -mla 2, -no-fa) becomes minutes per arm.
#   * clean past the budget -> the hypothesis is wrong and the search moves on.
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

PROFILE="${1:?usage: stress.sh <profile> [--budget N] [--ub N] [--extra ARGS]}"; shift
BUDGET=600000; UB=""; EXTRA=""
while [ $# -gt 0 ]; do
    case "$1" in
        --budget) BUDGET="$2"; shift 2 ;;
        --ub)     UB="$2"; shift 2 ;;
        --extra)  EXTRA="$2"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

PORT="${IK_PORT:-8090}"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/stress-${STAMP}.log"
mkdir -p logs results

./stop.sh >/dev/null 2>&1
for ((i=0;i<30;i++)); do ps -eo comm | grep -q '^llama-serv' || break; sleep 2; done

(
    [ -n "$UB" ]    && export IK_UBATCH="$UB"
    [ -n "$EXTRA" ] && export IK_EXTRA_ARGS="$EXTRA"
    export IK_PORT="$PORT" IK_KILL_SQUATTERS=1
    ./serve.sh "$PROFILE" > "$LOG" 2>&1
) &

for ((i=0;i<240;i++)); do
    grep -qa 'HTTP server listening' "$LOG" && break
    if [ "$i" -gt 6 ] && ! ps -eo comm | grep -q '^llama-serv'; then
        echo "server failed to start"; tail -5 "$LOG"; exit 1
    fi
    sleep 5
done
echo "server up: profile $PROFILE${UB:+, -ub $UB}${EXTRA:+, extra '$EXTRA'}"
echo "log: $LOG"

python3 - "$PORT" "$BUDGET" "$LOG" <<'PY'
import json, os, re, sys, time, urllib.error, urllib.request
port, budget, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]
api = f"http://127.0.0.1:{port}"

def post(path, payload, timeout=600):
    req = urllib.request.Request(api + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

# Lengths chosen around crash 7's profile: median 790, nearly all below -ub, with
# a long tail so context checkpoints still get built.
LENGTHS = [80, 260, 500, 790, 1100, 340, 900, 620, 1500, 180, 2600, 740, 430, 6400]

def body(n_tokens, salt):
    # ~4.4 tokens per generated line, so this lands near the target without a
    # tokenize round trip per request -- speed matters more than precision here.
    lines = max(1, int(n_tokens / 4.4))
    return (f"Note {salt}.\n" +
            "".join(f"[{salt}] item {i}: value {i*7%1000}, state {i%37}.\n" for i in range(lines)))

sent = tokens = 0
t0 = time.time()
history = []
try:
    while tokens < budget:
        n = LENGTHS[sent % len(LENGTHS)]
        salt = f"s{os.getpid()}n{sent}"
        # every 5th request continues the previous conversation, so the context
        # grows and checkpoints are created, as in every crashing run
        if sent % 5 == 4 and history:
            msgs = history[-6:] + [{"role": "user", "content": body(n, salt)}]
        else:
            msgs = [{"role": "user", "content": body(n, salt)}]
            history = []
        r = post("/v1/chat/completions", {"model": "x", "messages": msgs,
                                          "max_tokens": 16, "temperature": 0.0})
        t = r.get("timings", {})
        tokens += int(t.get("prompt_n", 0)); sent += 1
        history = msgs + [{"role": "assistant",
                           "content": r["choices"][0]["message"].get("content") or "ok"}]
        if sent % 25 == 0:
            print(f"  {sent} requests, {tokens} prefilled tokens, "
                  f"{(time.time()-t0)/60:.0f} min", flush=True)
except Exception as e:
    mins = (time.time() - t0) / 60
    alive = os.system("ps -eo comm | grep -q '^llama-serv'") == 0
    lt = open(log, errors="replace").read()
    nan = "Failed to sample token" in lt
    print(f"\n  STOPPED after {sent} requests, {tokens} prefilled tokens, {mins:.0f} min")
    print(f"  server alive: {'yes' if alive else 'NO'}   NaN abort in log: {'YES' if nan else 'no'}")
    print(f"  exception: {repr(e)[:160]}")
    sys.exit(2 if nan else 3)

print(f"\n  CLEAN: {sent} requests, {tokens} prefilled tokens, "
      f"{(time.time()-t0)/60:.0f} min, no abort")
PY
rc=$?
case "$rc" in
  0) echo "  -> clean run. CAREFUL: this only clears the configuration AS BUILT."
     echo "     If the ik_llama build changed since the aborts, a clean run does not"
     echo "     disprove anything about the configuration -- it is equally explained"
     echo "     by the build. Say which build it ran on when reporting the result." ;;
  2) echo "  -> NaN ABORT REPRODUCED -- this is the fast test case"
     cp -n probabilities.txt "docs/external/crashes/probabilities-stress-${STAMP}.txt" 2>/dev/null ;;
  3) echo "  -> the run stopped without a NaN abort in the log; check $LOG" ;;
esac
./stop.sh >/dev/null 2>&1
exit "$rc"
