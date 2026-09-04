#!/usr/bin/env bash
# =============================================================================
# bench.sh -- performance measurement for the configured model
# =============================================================================
#   ./bench.sh quick     [profile]   pp512/2048/8192 + tg128       (~5 min)
#   ./bench.sh sweep     [profile]   throughput vs context depth   (~10 min)
#   ./bench.sh threads   [profile]   find the best -t              (~10 min)
#   ./bench.sh ncmoe     [profile]   find the best expert split    (~25 min)
#   ./bench.sh batch     [profile]   find the best -b / -ub        (~15 min)
#   ./bench.sh rtr       [profile]   is -rtr worth it?             (~15 min)
#   ./bench.sh full      [profile]   everything above              (~80 min)
#
# Results land in results/<profile>-<mode>-<timestamp>.md
#
# Note on runtime: loading this model costs real time (~114 GiB), so modes that
# can sweep inside a single llama-bench invocation do -- 'threads' and 'batch'
# load the model once. 'ncmoe' and 'rtr' change how tensors are placed at load
# time, so those genuinely need one load per data point.
# -----------------------------------------------------------------------------
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/lib/common.sh"

MODE="${1:-quick}"
PROFILE="${2:-}"
[[ $MODE == -h || $MODE == --help ]] && { sed -n '2,18p' "$0"; exit 0; }

load_config "$PROFILE"
BENCH="$(require_binary llama-bench)"
SWEEP="$(require_binary llama-sweep-bench)"

resolve_model
check_model_complete
preflight_gpu

mkdir -p "$RESULT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="$RESULT_DIR/${IK_PROFILE}-${MODE}-${STAMP}"
REPORT="$OUT_BASE.md"
RAW="$OUT_BASE.raw.log"

{
    echo "# ik-llama benchmark -- $MODE"
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| date | $(date -Is) |"
    echo "| profile | \`$IK_PROFILE\` |"
    echo "| model | \`$(basename "$MODEL_PATH")\` (${MODEL_SIZE_GB} GiB) |"
    echo "| GPU | $(nvidia-smi --query-gpu=name --format=csv,noheader) |"
    echo "| VRAM free at start | ${GPU_FREE_MIB} / ${GPU_TOTAL_MIB} MiB |"
    echo "| CPU | $(lscpu | sed -n 's/^Model name: *//p' | head -1) |"
    echo "| RAM | $(free -g | awk '/^Mem:/ {print $2" GiB"}') |"
    echo "| ik_llama.cpp | $(git -C "$IK_SRC" log -1 --format='%h %ad' --date=short 2>/dev/null) |"
    echo "| context | $IK_CTX |"
    echo "| KV cache | ${IK_CTK} / ${IK_CTV} |"
    echo "| expert placement | $([[ -n ${IK_NCMOE:-} ]] && echo "--n-cpu-moe $IK_NCMOE" || echo "--fit (margin ${IK_FIT_MARGIN} MiB)") |"
    echo
} > "$REPORT"

log "writing $REPORT"

# -----------------------------------------------------------------------------
# llama-bench plumbing
# -----------------------------------------------------------------------------
# Careful: llama-bench has its own parser, and several flags that are booleans
# on llama-server take an explicit 0/1 here (--fit, -thp, -fmoe, -rtr, -fa).

bench_base_args() {
    BB_ARGS=(
        -m "$MODEL_PATH"
        -ngl "${IK_NGL:-99}"
        -fa 1
        -ctk "${IK_CTK:-q8_0}" -ctv "${IK_CTV:-q8_0}"
        -thp "${IK_THP:-1}"
        -r "${IK_BENCH_REPS:-3}"
    )
    if [[ -n ${IK_BENCH_NCMOE:-${IK_NCMOE:-}} ]]; then
        BB_ARGS+=( --n-cpu-moe "${IK_BENCH_NCMOE:-$IK_NCMOE}" )
    else
        BB_ARGS+=( --fit 1 --fit-margin "${IK_FIT_MARGIN:-2048}" )
    fi
    [[ -n ${IK_BENCH_RTR:-} ]] && BB_ARGS+=( -rtr "$IK_BENCH_RTR" )
    [[ ${IK_OOAE:-1} == 1 ]] || BB_ARGS+=( -no-ooae 1 )
}

section() { { echo; echo "## $*"; echo; } >> "$REPORT"; log "$*"; }
note()    { { echo; cat; echo; } >> "$REPORT"; }

# The benchmark binary is piped through tee|grep|tee to split its output
# between the raw log and the trimmed report. That pipeline hides the binary's
# exit code (grep/tee succeed even when the binary aborts with a core dump), so
# a crash used to look like a clean run. Check PIPESTATUS[0] -- the binary's own
# status -- and stop loudly instead.
FAILED=0
check_bench_status() {
    local rc="$1" what="$2"
    if [[ $rc -ne 0 ]]; then
        FAILED=1
        warn "$what exited with status $rc -- see $RAW"
        { echo; echo "> **$what failed (exit $rc).** See the raw log."; echo; } >> "$REPORT"
        # 132-139 are fatal signals (SIGILL/ABRT/BUS/SEGV...). One of those means
        # a genuine crash, not just a bad config -- worth surfacing hard.
        if [[ $rc -ge 132 && $rc -le 139 ]]; then
            warn "that is a fatal signal (crash), not a soft error"
        fi
    fi
}

run_bench() {
    bench_base_args
    show_cmd "$BENCH" "${BB_ARGS[@]}" "$@"
    "$BENCH" "${BB_ARGS[@]}" "$@" 2>&1 | tee -a "$RAW" | grep -E '^\|' | tee -a "$REPORT"
    check_bench_status "${PIPESTATUS[0]}" "llama-bench"
}

run_sweep() {
    local label="$1"; shift
    build_common_args
    { echo; echo "### $label"; echo; } >> "$REPORT"
    show_cmd "$SWEEP" "${COMMON_ARGS[@]}" --output-format md "$@"
    "$SWEEP" "${COMMON_ARGS[@]}" --output-format md "$@" 2>&1 \
        | tee -a "$RAW" | grep -E '^\|' | tee -a "$REPORT"
    check_bench_status "${PIPESTATUS[0]}" "llama-sweep-bench"
}

TG="${IK_THREADS:-6},${IK_THREADS_BATCH:-18}"

# =============================================================================
# Modes
# =============================================================================

do_quick() {
    section "Prompt processing and generation"
    note <<'EOF'
`pp` is prefill (tokens/s of input ingested), `tg` is generation (tokens/s of
output produced). `tg` is the number you feel while using the model.
EOF
    run_bench -p 512,2048,8192 -n 128 \
              -b "${IK_BATCH:-4096}" -ub "${IK_UBATCH:-1024}" -tgb "$TG"
}

do_sweep() {
    section "Throughput vs context depth"
    note <<'EOF'
Each row advances the context by one `ubatch` and then generates from that
depth, so this shows how throughput decays as the KV cache fills. `S_PP` is
prefill speed, `S_TG` generation speed, both tokens/s.
EOF
    run_sweep "sweep to ${IK_CTX} tokens"
}

do_threads() {
    section "Thread count"
    note <<'EOF'
Each row is `<generation threads>,<prefill threads>`. Generation over
CPU-resident experts is bound by DDR5 bandwidth and saturates around the
6 P-cores; prefill is compute-bound and wants all 18 cores. Mixing P- and
E-cores can *cost* generation throughput, because ggml's thread pool waits for
the slowest thread at every barrier.

All rows share one model load.
EOF
    run_bench -p 2048 -n 128 -tgb "4,18;6,18;8,18;12,18;18,18" \
              -b "${IK_BATCH:-4096}" -ub "${IK_UBATCH:-1024}"

    { echo; echo "Pinned to P-cores only (\`taskset -c 0-5\`, 6 threads):"; echo; } >> "$REPORT"
    bench_base_args
    taskset -c 0-5 "$BENCH" "${BB_ARGS[@]}" -p 2048 -n 128 -tgb "6,6" 2>&1 \
        | tee -a "$RAW" | grep -E '^\|' | tee -a "$REPORT"
}

do_batch() {
    section "Batch and micro-batch size"
    note <<'EOF'
Larger `-ub` raises prefill throughput but grows the CUDA compute buffers,
which eats VRAM that would otherwise hold experts. Watch for a row that wins on
`pp` but loses on `tg` -- that is the trade being made, and on this machine it
is usually a bad one, since you prefill once and generate many times.

All rows share one model load (cartesian product of `-b` x `-ub`).
EOF
    run_bench -p 8192 -n 128 -b 4096,8192 -ub 512,1024,2048 -tgb "$TG"
}

do_ncmoe() {
    section "Expert placement (--n-cpu-moe)"
    note <<'EOF'
`--n-cpu-moe N` keeps the routed experts of the first N layers in system RAM.
Layers 0-2 of this model are dense and own no expert tensors, so N maps to
**N-3** layers of experts actually on the CPU.

Lower N is better until the GPU runs out of memory -- a run that fails or
collapses marks the ceiling. Compare the winner against the `--fit` row, which
is what the server picks on its own.

One model load per row, so this mode is the slow one.
EOF
    { echo "Baseline, \`--fit\` choosing for itself:"; echo; } >> "$REPORT"
    run_bench -p 2048 -n 128 -tgb "$TG"

    for n in 10 14 17 20 24; do
        log "  --n-cpu-moe $n  ($((n > 3 ? n - 3 : 0)) expert layers on CPU)"
        { echo; echo "\`--n-cpu-moe $n\` ($((n > 3 ? n - 3 : 0)) expert layers on CPU):"; echo; } >> "$REPORT"
        IK_BENCH_NCMOE="$n" run_bench -p 2048 -n 128 -tgb "$TG"
    done
}

do_rtr() {
    section "Run-time repack (-rtr)"
    note <<'EOF'
`-rtr` repacks CPU-side experts into the row-interleaved `_R4`/`_R8` layout the
AVX2/VNNI kernels prefer -- and, since those types have no CUDA kernel, it moves
the expert GEMM onto the CPU instead of streaming the experts to the GPU per
micro-batch. So this is a CPU-vs-GPU comparison and it turns on `-ub`: at 512
the CPU tends to win, at 4096 the GPU won 3x here (RESULTS 21). Both rows below
run at the profile's `-b`/`-ub`, the ones you would serve with. It forces
`--no-mmap`, so the whole model is re-read from disk at every start.

Enable it only if `pp` gains at that `-ub`. Do not repack the file offline with
`llama-quantize --repack`: that pins the experts to the CPU permanently (TODO 8).
EOF
    for r in 0 1; do
        log "  -rtr $r"
        { echo; echo "\`-rtr $r\`:"; echo; } >> "$REPORT"
        IK_BENCH_RTR="$r" run_bench -p 2048 -n 128 \
                  -b "${IK_BATCH:-4096}" -ub "${IK_UBATCH:-1024}" -tgb "$TG"
    done
}

case "$MODE" in
    quick)   do_quick ;;
    sweep)   do_sweep ;;
    threads) do_threads ;;
    ncmoe)   do_ncmoe ;;
    batch)   do_batch ;;
    rtr)     do_rtr ;;
    full)    do_quick; do_threads; do_batch; do_ncmoe; do_rtr; do_sweep ;;
    *)       die "unknown mode '$MODE' (quick|sweep|threads|ncmoe|batch|rtr|full)" ;;
esac

echo >&2
if [[ $FAILED -ne 0 ]]; then
    warn "report: $REPORT"
    warn "raw:    $RAW"
    die "one or more benchmark runs failed -- results above are incomplete"
fi
ok "report: $REPORT"
ok "raw:    $RAW"
