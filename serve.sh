#!/usr/bin/env bash
# =============================================================================
# serve.sh -- start the ik_llama.cpp inference server
# =============================================================================
#   ./serve.sh                          default profile (DeepSeek-V4-Flash MXFP4,
#                                       kvram 128k -- see config/default.env)
#   ./serve.sh step-3.7-flash-q4        the original default, still available
#   ./serve.sh --ctx 131072             override context length
#   ./serve.sh --port 9000 --host 0.0.0.0
#   ./serve.sh --dry-run                print the command without running it
#   ./serve.sh --list                   show available profiles
#
# Serves an OpenAI-compatible API plus a web UI at http://HOST:PORT
# -----------------------------------------------------------------------------
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/lib/common.sh"

PROFILE=""
DRY_RUN=0
declare -a CLI_OVERRIDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)     source "$TOOLKIT_ROOT/config/default.env"
                    echo "available profiles:"; list_profiles | sed 's/^/  /'
                    echo; echo "default: $IK_PROFILE"; exit 0 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --ctx)      export IK_CTX="$2"; shift 2 ;;
        --port)     export IK_PORT="$2"; shift 2 ;;
        --host)     export IK_HOST="$2"; shift 2 ;;
        --model)    export IK_MODEL="$2"; shift 2 ;;
        --threads)  export IK_THREADS="$2"; shift 2 ;;
        --ncmoe)    export IK_NCMOE="$2"; export IK_FIT=0; shift 2 ;;
        --parallel) export IK_PARALLEL="$2"; shift 2 ;;
        -h|--help)  sed -n '2,16p' "$0"; exit 0 ;;
        --)         shift; CLI_OVERRIDES+=( "$@" ); break ;;
        -*)         die "unknown option: $1  (pass extra llama-server flags after --)" ;;
        *)          [[ -n $PROFILE ]] && die "more than one profile given"
                    PROFILE="$1"; shift ;;
    esac
done

load_config "$PROFILE"
SERVER="$(require_binary llama-server)"

resolve_model
check_model_complete
preflight_gpu

log "profile     $IK_PROFILE"
log "model       $(basename "$MODEL_PATH")  (${MODEL_SIZE_GB} GiB)"
log "context     $IK_CTX tokens, KV ${IK_CTK}/${IK_CTV}, flash-attn ${IK_FA}"
log "GPU         ${GPU_FREE_MIB} of ${GPU_TOTAL_MIB} MiB free"
if [[ -n ${IK_OT:-} ]]; then
    log "experts     manual -ot placement"
elif [[ -n ${IK_NCMOE:-} ]]; then
    log "experts     first $IK_NCMOE layers on CPU (fixed)"
elif [[ ${IK_FIT:-1} == 1 ]]; then
    log "experts     auto-fit, leaving ${IK_FIT_MARGIN} MiB VRAM headroom"
else
    log "experts     all on GPU (-ngl ${IK_NGL:-99}, no --fit)"
fi

build_common_args

SERVER_ARGS=( "${COMMON_ARGS[@]}" )
SERVER_ARGS+=( --host "$IK_HOST" --port "$IK_PORT" )
SERVER_ARGS+=( --parallel "${IK_PARALLEL:-1}" )
[[ -n ${IK_ALIAS:-}   ]] && SERVER_ARGS+=( --alias "$IK_ALIAS" )
[[ -n ${IK_API_KEY:-} ]] && SERVER_ARGS+=( --api-key "$IK_API_KEY" )
[[ -n ${IK_MMPROJ:-}  ]] && SERVER_ARGS+=( --mmproj "$IK_MMPROJ" )
[[ ${IK_JINJA:-1} == 1 ]] && SERVER_ARGS+=( --jinja )
[[ ${IK_CONT_BATCHING:-1} == 1 ]] && SERVER_ARGS+=( --cont-batching )
[[ ${#CLI_OVERRIDES[@]} -gt 0 ]] && SERVER_ARGS+=( "${CLI_OVERRIDES[@]}" )

PIN="$(maybe_taskset)"

echo >&2
show_cmd ${PIN:+$PIN} "$SERVER" "${SERVER_ARGS[@]}"
echo >&2

if [[ $DRY_RUN == 1 ]]; then
    ok "dry run -- not starting"
    exit 0
fi

log "loading ~${MODEL_SIZE_GB} GiB; first start reads from disk, later starts hit the page cache"
log "API will be at http://${IK_HOST}:${IK_PORT}/v1  (web UI at http://${IK_HOST}:${IK_PORT})"
echo >&2

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
log "logging to $LOG_FILE"
echo >&2

# --- IK_GDB: run under gdb so an abort leaves a backtrace ---------------------
# The NaN-logits abort (RESULTS §19, §32) has fired nine times and not once left
# a stack trace, which is most of why it is still open.
#
# The reason is not that ggml does not try. GGML_ABORT calls ggml_print_backtrace,
# which forks and runs `gdb --batch -ex "attach <parent pid>"` -- a CHILD
# attaching to its PARENT. Yama's ptrace_scope=1 (the default on Ubuntu, and what
# this box runs) permits tracing only descendants, so it is refused. gdb then
# reports it as "ptrace: Inappropriate ioctl for device", which is an Ubuntu patch
# clobbering errno to ENOTTY before perror -- the real error is EPERM.
#
# ggml has a fallback for exactly this, backtrace_symbols_fd, gated on gdb exiting
# EXIT_FAILURE. Measured: gdb exits 0 after a refused attach. So the fallback never
# runs, and on any ptrace_scope=1 system the abort prints no backtrace at all.
#
# Starting the server UNDER gdb inverts the relationship -- gdb is the parent, and
# Yama has no objection. No root, and no loosening ptrace_scope machine-wide.
#
# gdb sits waiting for the whole run, so the expected cost is nothing until
# something aborts -- but that is an expectation, NOT a measurement: ptrace does
# intercept signals and thread creation. If a benchmark here ever disagrees with a
# stored result by more than the ~2 % noise floor, check this first, and note that
# tools/ all go through serve.sh so they inherit it too.
#
# Set IK_GDB=0 to turn it off.
: "${IK_GDB:=1}"
GDB_PREFIX=()
if [[ ${IK_GDB} == 1 ]]; then
    if command -v gdb >/dev/null 2>&1; then
        # SIGTERM must pass STRAIGHT through. Without the handle, stop.sh leaves a
        # full backtrace in every log -- gdb stops on the signal, runs the bt
        # commands, then kills the inferior, so an ordinary stop is indistinguishable
        # from a crash at a glance and the server never runs its shutdown path.
        # Verified both ways: abort still yields the stack, stop yields none.
        GDB_PREFIX=(gdb --batch
            -ex "set confirm off"
            -ex "set debuginfod enabled off"
            -ex "handle SIGINT nostop pass noprint"
            -ex "handle SIGTERM nostop pass noprint"
            -ex run
            -ex "bt -frame-info source-and-location"
            -ex "thread apply all bt"
            -ex quit --args)
        log "running under gdb: an abort will leave a backtrace in the log (IK_GDB=0 disables)"
    else
        warn "IK_GDB=1 but gdb is not installed -- an abort will leave no backtrace"
        warn "  sudo apt install gdb"
    fi
fi

# Piped through tee so the run is logged; PIPESTATUS carries the server's own
# exit code rather than tee's.
# shellcheck disable=SC2086
${PIN:+$PIN} "${GDB_PREFIX[@]}" "$SERVER" "${SERVER_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
