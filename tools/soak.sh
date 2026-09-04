#!/usr/bin/env bash
# =============================================================================
# soak.sh -- hours of agent-shaped traffic against the server that is ALREADY
#            running, and a loud stop on the first sign of trouble
# =============================================================================
#   ./tools/soak.sh [--hours H] [--budget N] [--max-depth N] [--port P] [--seed S]
#
#     --hours H      stop after H hours without a failure (default 6)
#     --budget N     ... or after N prefilled tokens, whichever comes first
#     --max-depth N  deepest n_past a conversation is grown to (default 120000)
#     --port P       server port (default IK_PORT or 8090)
#     --seed S       RNG seed, printed at the start so a run can be repeated
#
# HOW IT DIFFERS FROM stress.sh. stress.sh owns its server: it stops whatever is
# running, starts the profile it was given and tears it down at the end. This
# one owns nothing. It talks to the server that is already listening -- the
# systemd unit, normally -- and refuses to start if there is none, so it cannot
# fight the unit for the GPU and the unit's own log (logs/server-*.log, written
# by tee) is where the evidence lands. It is the tool for "the build changed,
# does the production profile still hold up under a night of traffic".
#
# WHAT IT SENDS, per cycle:
#   short    14 fresh single-turn prompts, the crash-7 length profile stress.sh
#            uses (median 790 tokens, nearly all below -ub, one 6.4k tail).
#   probe    one fixed ~3k-token prompt at temperature 0. The first answer is
#            kept and every later one compared against it; a MISMATCH is logged
#            with both texts but does not stop the run -- the RAM prompt cache
#            and the u-batch split legitimately change the arithmetic.
#   climb    one conversation grown turn by turn with 0.8k-6k tokens of real
#            prose (the docs of this repo) to --max-depth, generating up to 160
#            tokens per turn at temperature 0.7 (every 5th turn at 0). Every
#            4th turn BRANCHES: the history is cut back to an earlier turn and
#            continued differently, so the slot reuses a cached prefix that
#            ends at an arbitrary, non-256-aligned position. That shape is what
#            produced every NaN abort (RESULTS 49.2, TODO 9) and it is the path
#            the mask-share patch (RESULTS 49.9-49.10) rewrote, so it gets the
#            most time.
#
# WHAT STOPS IT: any non-200 response, a timeout (600 s), a connection failure,
# the unit's log file changing (a restart), or one of these in the server log:
#   ERR [...]                       any -- sampling failed, NaN logits, task error
#   ==== Failed to sample token     the sampler's own line
#   IK_NAN_CHECK: first NaN         the probe, when IK_NAN_CHECK=1 is set
#   received signal SIG...          gdb, when the unit runs the server under it
# Server output from before the run is not scanned, only what arrives during it,
# and only in those formats -- see FAIL below for why not a plain grep.
#
# ONE SLOT. The production profiles run --parallel 1, so this shares the slot
# with anyone using the model: their conversation is evicted by every soak
# request and re-prefilled on their next turn (a 50k context costs ~30 s).
# Run it when nobody is using the model, or accept that.
#
# OUTPUT. logs/soak-<stamp>.log is the narrative, results/soak-<stamp>.tsv has
# one line per request (phase, prefilled tokens, n_past, timings) so that
# throughput-at-depth can be plotted afterwards. Exit 0 clean, 2 the server
# failed, 3 the driver stopped for another reason. To run it detached from a
# terminal:
#   systemd-run --user --unit=ik-soak --collect --same-dir \
#       -p WorkingDirectory=$PWD ./tools/soak.sh --hours 6
#   journalctl --user -u ik-soak -f          # or tail -f logs/soak-*.log
#   systemctl --user stop ik-soak            # stop early
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

HOURS=6; BUDGET=0; MAXDEPTH=120000; PORT="${IK_PORT:-8090}"; SEED="$(date +%s)"
while [ $# -gt 0 ]; do
    case "$1" in
        --hours)     HOURS="$2"; shift 2 ;;
        --budget)    BUDGET="$2"; shift 2 ;;
        --max-depth) MAXDEPTH="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="logs/soak-${STAMP}.log"
TSV="results/soak-${STAMP}.tsv"
mkdir -p logs results

if ! curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "no server on port ${PORT} -- this tool does not start one." | tee "$LOG"
    echo "  systemctl --user start ik-llama-server   (or ./serve.sh <profile>)" | tee -a "$LOG"
    exit 1
fi
SERVER_LOG=$(ls -t logs/server-*.log 2>/dev/null | head -1)
[ -n "$SERVER_LOG" ] || { echo "no logs/server-*.log to watch"; exit 1; }

# Identity of what is being soaked. A clean run clears the BUILD it ran on and
# nothing else, so say which one that was (stress.sh learned this the hard way).
{
    echo "==> soak        ${STAMP}   hours ${HOURS}   budget ${BUDGET:-0}   max-depth ${MAXDEPTH}   seed ${SEED}"
    echo "==> server      port ${PORT}   log ${SERVER_LOG}"
    echo "==> model       $(curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)"
    echo "==> build       $(grep -a -m1 -oE 'build=[0-9]+ commit="[0-9a-f]+"' "$SERVER_LOG")   libllama.so $(stat -c %y ik_llama.cpp/build/src/libllama.so 2>/dev/null | cut -c1-19)   clone diff $(git -C ik_llama.cpp diff --shortstat | sed 's/^ *//')"
    echo "==> toolkit     $(git rev-parse --short HEAD) $(git status --porcelain | grep -q . && echo '(dirty)' || echo '(clean)')"
    grep -a -m1 "CUDA0 compute buffer size" "$SERVER_LOG" | sed 's/^/==> /'
    echo "==> gpu         $(nvidia-smi --query-gpu=power.limit,memory.used --format=csv,noheader 2>/dev/null)"
    echo "==> per-request ${TSV}"
} | tee "$LOG"

python3 - "$PORT" "$HOURS" "$BUDGET" "$MAXDEPTH" "$SEED" "$SERVER_LOG" "$TSV" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, random, re, sys, time, urllib.error, urllib.request
port, hours, budget, max_depth, seed, server_log, tsv = sys.argv[1:8]
hours, budget, max_depth = float(hours), int(budget), int(max_depth)
api = f"http://127.0.0.1:{port}"
rng = random.Random(seed)
deadline = time.time() + hours * 3600

class ServerFailed(Exception): pass
class Done(Exception): pass

# --- the server log, read incrementally from where it is now -----------------
# Anchored to the server's own line formats, because the server echoes 30-50
# tokens of prompt and cache around every divergence point ("cache : ..." /
# "prompt: ...", print_tokens in server-context.cpp) and the prose sent here is
# this repo's docs, which quote every crash signature there is. The first run
# stopped on "IK_NAN_CHECK" inside such an echo, 35 min in. Every ERR line the
# server has ever written was a real failure (sampling failed, NaN logits, task
# error, KV cache full), so any ERR line stops the run; crashes come through
# gdb's "received signal" line and, gdb or not, as a failed request.
FAIL = re.compile(r"^\s*ERR \[|^=+ Failed to sample token|^IK_NAN_CHECK: first NaN|"
                  r"received signal SIG|^Program terminated", re.MULTILINE)
log_pos = os.path.getsize(server_log)
def check_server_log():
    global log_pos
    try:
        with open(server_log, "rb") as f:
            f.seek(log_pos); new = f.read(); log_pos += len(new)
    except OSError as e:
        raise ServerFailed(f"cannot read {server_log}: {e}")
    text = new.decode(errors="replace")
    m = FAIL.search(text)
    if m:
        line = text[text.rfind("\n", 0, m.start()) + 1:].split("\n", 1)[0]
        raise ServerFailed(f"server log: {line.strip()[:300]}")
    newest = max((os.path.join("logs", n) for n in os.listdir("logs") if n.startswith("server-")),
                 key=os.path.getmtime)
    if os.path.realpath(newest) != os.path.realpath(server_log) and \
       os.path.getmtime(newest) > os.path.getmtime(server_log):
        raise ServerFailed(f"server restarted: new log {newest}")

# --- one request ---------------------------------------------------------------
def post(payload, timeout=600):
    req = urllib.request.Request(api + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise ServerFailed(f"HTTP {e.code}: {e.read(300).decode(errors='replace')}")
    except (OSError, ValueError) as e:   # URLError, timeouts, resets, broken JSON
        raise ServerFailed(f"request failed: {repr(e)[:200]}")

# --- prose: the repo's own docs, chosen because they are real text -------------
docs = []
for p in ["docs/RESULTS.md", "TODO.md", "README.md", "docs/TUNING.md", "docs/FAQ.md",
          "docs/TROUBLESHOOTING.md", "ik_llama.cpp/src/graphs/build_deepseek4.cpp"]:
    try: docs.append(open(p, errors="replace").read())
    except OSError: pass
prose = "\n\n".join(docs)
cpt = 4.0  # chars per token, corrected from the server's own count as it goes

def chunk(n_tokens):
    n = int(n_tokens * cpt)
    start = rng.randrange(0, max(1, len(prose) - n))
    return prose[start:start + n]

QUESTIONS = ["What does the passage above measure, and which number stands out?",
             "Summarize the passage above in three sentences.",
             "List the configuration flags mentioned above and what each one does.",
             "Is there a contradiction in the passage above? Answer briefly.",
             "Which of the claims above would you verify first, and how?"]

# crash-7 length profile, as in stress.sh -- but one of these lines is ~17
# tokens on this tokenizer, not the 4.4 stress.sh assumes (measured 2026-09-04:
# its "80-token" prompt is 316 tokens, its "2600" is 10 040), so the division
# here is by 17 and the lengths land where the list says.
LENGTHS = [80, 260, 500, 790, 1100, 340, 900, 620, 1500, 180, 2600, 740, 430, 6400]
def synthetic(n_tokens, salt):
    lines = max(1, int(n_tokens / 17))
    return (f"Note {salt}.\n" +
            "".join(f"[{salt}] item {i}: value {i*7%1000}, state {i%37}.\n" for i in range(lines)))

# --- bookkeeping ---------------------------------------------------------------
sent = prefilled = generated = 0
deepest = branches = mismatches = cycles = 0
t0 = time.time()
tsvf = open(tsv, "w")
tsvf.write("t\tcycle\tphase\ttemp\tprompt_n\tn_past\tprompt_ms\tpredicted_n\tpredicted_ms\n")

def answer_text(r):
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or "")

def ask(phase, msgs, temp, max_tokens, chars=None):
    global sent, prefilled, generated, deepest, cpt
    r = post({"model": "x", "messages": msgs, "max_tokens": max_tokens, "temperature": temp})
    t = r.get("timings", {})
    pn, npast = int(t.get("prompt_n", 0)), int(t.get("n_past", 0))
    sent += 1; prefilled += pn; generated += int(t.get("predicted_n", 0))
    deepest = max(deepest, npast)
    if chars and pn > 50:
        cpt = 0.8 * cpt + 0.2 * (chars / pn)
    tsvf.write(f"{time.time()-t0:.1f}\t{cycles}\t{phase}\t{temp}\t{pn}\t{npast}\t{t.get('prompt_ms',0):.0f}\t"
               f"{t.get('predicted_n',0)}\t{t.get('predicted_ms',0):.0f}\n"); tsvf.flush()
    check_server_log()
    if budget and prefilled >= budget: raise Done("budget")
    if time.time() > deadline: raise Done("time")
    return r, npast

def progress(note=""):
    mins = (time.time() - t0) / 60
    print(f"  {mins:6.0f} min  cycle {cycles:3d}  {sent:5d} req  {prefilled/1e6:5.2f}M prefilled  "
          f"{generated:6d} generated  deepest {deepest:6d}  branches {branches:3d}  "
          f"mismatch {mismatches}{('  ' + note) if note else ''}", flush=True)

probe_prompt = [{"role": "user", "content": prose[:12000] + "\n\nSummarize the above in two sentences."}]
probe_ref = None
reason = None
try:
    while True:
        cycles += 1
        # short: fresh single turns of the shape that used to abort
        for i, n in enumerate(LENGTHS):
            ask("short", [{"role": "user", "content": synthetic(n, f"c{cycles}s{i}")}],
                0.7, 32)
        # probe: same prompt, temperature 0, compared with the first answer
        r, _ = ask("probe", probe_prompt, 0.0, 48)
        text = "|".join(answer_text(r))
        if probe_ref is None:
            probe_ref = text
            print(f"  probe reference ({len(text)} chars): {text[:160]!r}", flush=True)
        elif text != probe_ref:
            mismatches += 1
            print(f"  MISMATCH #{mismatches} on the probe at cycle {cycles}:\n    ref {probe_ref[:200]!r}\n    now {text[:200]!r}",
                  flush=True)
        # climb: one conversation to max_depth, branching every 4th turn
        history, npast, turn = [], 0, 0
        while npast < max_depth and turn < 120:
            turn += 1
            if turn % 4 == 0 and len(history) >= 4:
                # drop the last one or two turns: the cached prefix then ends
                # wherever that earlier answer happened to end
                keep = max(1, len(history) // 2 - rng.randint(1, 2))
                history = history[:2 * keep]
                branches += 1
            n = rng.choice([800, 1500, 2500, 3000, 4000, 6000])
            body = chunk(n) + "\n\n" + rng.choice(QUESTIONS)
            msgs = history + [{"role": "user", "content": body}]
            temp = 0.0 if turn % 5 == 0 else 0.7
            r, npast = ask("climb", msgs, temp, 160, chars=len(body) if not history else None)
            content, reasoning = answer_text(r)
            history = msgs + [{"role": "assistant", "content": content or reasoning[-400:] or "ok"}]
        progress(f"climb ended at n_past {npast} after {turn} turns")
except Done as e:
    reason = str(e)
except ServerFailed as e:
    progress("STOPPED")
    print(f"\n  SERVER FAILED after {sent} requests, {prefilled} prefilled tokens, "
          f"{(time.time()-t0)/60:.0f} min:\n  {e}", flush=True)
    sys.exit(2)
except KeyboardInterrupt:
    reason = "interrupted"
except Exception as e:
    progress("STOPPED")
    print(f"\n  DRIVER STOPPED after {sent} requests: {repr(e)[:300]}", flush=True)
    sys.exit(3)

progress()
print(f"\n  CLEAN ({reason}): {sent} requests, {prefilled} prefilled and {generated} generated tokens, "
      f"{cycles} cycles, deepest n_past {deepest}, {branches} branches, {mismatches} probe mismatches, "
      f"{(time.time()-t0)/60:.0f} min, no abort", flush=True)
PY
rc=${PIPESTATUS[0]}
case "$rc" in
  0) echo "  -> clean. This clears the build named above on this profile, and nothing else." ;;
  2) echo "  -> SERVER FAILED -- see $LOG and $SERVER_LOG" ;;
  *) echo "  -> stopped without a server failure (rc $rc); see $LOG" ;;
esac | tee -a "$LOG"
exit "$rc"
