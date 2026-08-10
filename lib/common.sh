#!/usr/bin/env bash
# Shared helpers for the ik-llama-toolkit scripts.
# Sourced by build.sh, serve.sh and bench.sh -- not meant to be run directly.

set -uo pipefail

TOOLKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IK_SRC="${IK_SRC:-$TOOLKIT_ROOT/ik_llama.cpp}"
IK_BUILD="${IK_BUILD:-$IK_SRC/build}"
IK_BIN="${IK_BIN:-$IK_BUILD/bin}"
LOG_DIR="${LOG_DIR:-$TOOLKIT_ROOT/logs}"
RESULT_DIR="${RESULT_DIR:-$TOOLKIT_ROOT/results}"

if [[ -t 1 ]]; then
    C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'
    C_BLU=$'\033[34m'; C_DIM=$'\033[2m';  C_OFF=$'\033[0m'
else
    C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_DIM=''; C_OFF=''
fi

log()  { printf '%s==>%s %s\n' "$C_BLU" "$C_OFF" "$*" >&2; }
ok()   { printf '%s ok %s %s\n' "$C_GRN" "$C_OFF" "$*" >&2; }
warn() { printf '%swarn%s %s\n' "$C_YEL" "$C_OFF" "$*" >&2; }
die()  { printf '%sfail%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; exit 1; }
dim()  { printf '%s%s%s\n' "$C_DIM" "$*" "$C_OFF" >&2; }

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
# Precedence (lowest to highest):
#   config/default.env  <  config/models/<PROFILE>.env  <  environment  <  CLI flags
#
# Every config file assigns with ${VAR:=default} ("set only if unset/empty"), so
# whatever is already in the environment always wins. The subtlety: because of
# those semantics, whichever file is sourced FIRST wins over later files. So to
# make the profile override default.env, the profile must be sourced *first* and
# default.env second (it then only fills in what the profile left unset). The
# environment, set before either, still beats both.

load_config() {
    local profile="${1:-}"

    local default_cfg="$TOOLKIT_ROOT/config/default.env"
    [[ -f $default_cfg ]] || die "missing config file: $default_cfg"

    # Resolve the profile name WITHOUT polluting the current shell with
    # default.env's values yet: CLI arg > IK_PROFILE in the environment >
    # IK_PROFILE declared in default.env.
    if [[ -z $profile ]]; then
        profile="${IK_PROFILE:-}"
    fi
    if [[ -z $profile ]]; then
        profile="$(bash -c "source '$default_cfg' 2>/dev/null; printf '%s' \"\${IK_PROFILE:-}\"")"
    fi
    [[ -n $profile ]] || die "no model profile selected (set IK_PROFILE in config/default.env)"

    local model_cfg="$TOOLKIT_ROOT/config/models/${profile}.env"
    [[ -f $model_cfg ]] || die "unknown profile '$profile' (no $model_cfg)
available: $(ls "$TOOLKIT_ROOT/config/models/" 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')"

    # Profile first (its :=defaults win), then default.env fills the rest.
    # shellcheck source=/dev/null
    source "$model_cfg"
    # shellcheck source=/dev/null
    source "$default_cfg"
    IK_PROFILE="$profile"
    export IK_PROFILE
}

list_profiles() {
    ls "$TOOLKIT_ROOT/config/models/" 2>/dev/null | sed 's/\.env$//'
}

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
# IK_MODEL_GLOB points at the *first shard* of a split GGUF, e.g.
#   /path/Step-3.7-Flash-UD-Q4_K_XL-00001-of-00004.gguf
# llama.cpp loads the remaining shards automatically from the same directory.

resolve_model() {
    local glob="${IK_MODEL:-${IK_MODEL_GLOB:-}}"
    [[ -n $glob ]] || die "IK_MODEL / IK_MODEL_GLOB is not set"

    # Expand the glob; take the lexicographically first match (shard 00001).
    local matches=()
    # shellcheck disable=SC2206
    matches=( $glob )
    if [[ ${#matches[@]} -eq 0 || ! -f ${matches[0]} ]]; then
        die "model not found: $glob
The download may still be in progress. Check with:
  ls -la $(dirname "$glob")"
    fi

    MODEL_PATH="${matches[0]}"
    export MODEL_PATH
}

# Refuse to start if the model is still downloading. LM Studio writes
# 'downloading_<name>.part' files next to the finished shards.
check_model_complete() {
    local dir; dir="$(dirname "$MODEL_PATH")"
    local base; base="$(basename "$MODEL_PATH")"

    # Derive the shard family, e.g. Step-3.7-Flash-UD-Q4_K_XL-00001-of-00004.gguf
    # -> prefix 'Step-3.7-Flash-UD-Q4_K_XL', total 4
    local prefix total
    if [[ $base =~ ^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
        prefix="${BASH_REMATCH[1]}"
        total=$((10#${BASH_REMATCH[3]}))
    else
        # Single-file model: just make sure no .part sits beside it.
        prefix="${base%.gguf}"
        total=1
    fi

    local partials
    partials="$(ls "$dir" 2>/dev/null | grep -F "$prefix" | grep -E '\.part$' || true)"
    if [[ -n $partials ]]; then
        warn "these shards are still downloading:"
        printf '%s\n' "$partials" | sed 's/^/       /' >&2
        die "model '$prefix' is incomplete -- wait for the download to finish"
    fi

    if [[ $total -gt 1 ]]; then
        local n=0 i
        for ((i = 1; i <= total; i++)); do
            local shard
            shard="$(printf '%s/%s-%05d-of-%05d.gguf' "$dir" "$prefix" "$i" "$total")"
            [[ -f $shard ]] && n=$((n + 1))
        done
        [[ $n -eq $total ]] || die "found $n of $total shards for '$prefix' -- model is incomplete"
    fi

    local size_gb
    size_gb="$(du -bc "$dir/$prefix"-*.gguf 2>/dev/null | tail -1 | awk '{printf "%.1f", $1/1073741824}')"
    MODEL_SIZE_GB="$size_gb"
    export MODEL_SIZE_GB
}

# ---------------------------------------------------------------------------
# GPU preflight
# ---------------------------------------------------------------------------

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_total_mib() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# List compute processes holding VRAM, excluding our own binaries.
gpu_squatters() {
    nvidia-smi --query-compute-apps=pid,used_memory,process_name \
               --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2 + 0 > 512 { printf "%s\t%s MiB\t%s\n", $1, $2, $3 }'
}

# Warn (or abort) when another runtime is sitting on the GPU. LM Studio and
# Ollama both keep models resident long after the last request, which silently
# pushes our expert tensors onto the CPU and destroys throughput.
preflight_gpu() {
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- is the NVIDIA driver installed?"

    local free total
    free="$(gpu_free_mib)"; total="$(gpu_total_mib)"
    [[ -n $free ]] || die "could not query GPU memory"

    local squatters; squatters="$(gpu_squatters)"
    if [[ -n $squatters ]]; then
        warn "other processes are holding GPU memory:"
        printf '%s\n' "$squatters" | sed 's/^/       /' >&2
        warn "free VRAM is only ${free} of ${total} MiB"
        if [[ ${IK_KILL_SQUATTERS:-0} == 1 ]]; then
            warn "IK_KILL_SQUATTERS=1 -- terminating them"
            while IFS=$'\t' read -r pid _ _; do
                [[ -n $pid ]] && kill "$pid" 2>/dev/null && log "sent SIGTERM to $pid"
            done <<< "$squatters"
            sleep 5
            free="$(gpu_free_mib)"
            ok "free VRAM after cleanup: ${free} MiB"
        else
            warn "free them first (quit LM Studio / 'ollama stop <model>'),"
            warn "or re-run with IK_KILL_SQUATTERS=1 to terminate them automatically."
        fi
    fi

    GPU_FREE_MIB="$free"
    GPU_TOTAL_MIB="$total"
    export GPU_FREE_MIB GPU_TOTAL_MIB
}

# ---------------------------------------------------------------------------
# Binaries
# ---------------------------------------------------------------------------

require_binary() {
    local name="$1"
    [[ -x "$IK_BIN/$name" ]] || die "$name not built -- run ./build.sh first (looked in $IK_BIN)"
    echo "$IK_BIN/$name"
}

# ---------------------------------------------------------------------------
# Argument assembly
# ---------------------------------------------------------------------------
# Builds the array of flags shared by llama-server, llama-bench and
# llama-sweep-bench. Every knob comes from the config so that the server and
# the benchmark measure exactly the same configuration.

build_common_args() {
    COMMON_ARGS=()

    COMMON_ARGS+=( --model "$MODEL_PATH" )
    COMMON_ARGS+=( -ngl "${IK_NGL:-99}" )
    COMMON_ARGS+=( -c "${IK_CTX:-65536}" )
    COMMON_ARGS+=( -fa "${IK_FA:-on}" )
    COMMON_ARGS+=( -ctk "${IK_CTK:-q8_0}" -ctv "${IK_CTV:-q8_0}" )
    COMMON_ARGS+=( -b "${IK_BATCH:-4096}" -ub "${IK_UBATCH:-1024}" )
    COMMON_ARGS+=( -t "${IK_THREADS:-6}" -tb "${IK_THREADS_BATCH:-18}" )

    # --- expert placement -------------------------------------------------
    # --fit is mutually exclusive with -ncmoe and -ot, so honour the explicit
    # knobs first and only fall back to auto-fitting.
    if [[ -n ${IK_OT:-} ]]; then
        # IK_OT may contain several patterns separated by ';'
        local pattern
        while IFS= read -r pattern; do
            [[ -n $pattern ]] && COMMON_ARGS+=( -ot "$pattern" )
        done <<< "${IK_OT//;/$'\n'}"
    elif [[ -n ${IK_NCMOE:-} ]]; then
        COMMON_ARGS+=( -ncmoe "$IK_NCMOE" )
    elif [[ ${IK_FIT:-1} == 1 ]]; then
        COMMON_ARGS+=( --fit --fit-margin "${IK_FIT_MARGIN:-2048}" )
    fi

    # --- optional performance switches ------------------------------------
    [[ ${IK_FMOE:-1}    == 1 ]] || COMMON_ARGS+=( --no-fmoe )
    [[ ${IK_OOAE:-1}    == 1 ]] || COMMON_ARGS+=( -no-ooae )
    [[ ${IK_RTR:-0}     == 1 ]] && COMMON_ARGS+=( -rtr )
    [[ ${IK_THP:-1}     == 1 ]] && COMMON_ARGS+=( -thp )
    [[ ${IK_MLOCK:-0}   == 1 ]] && COMMON_ARGS+=( --mlock )
    [[ ${IK_NO_MMAP:-0} == 1 ]] && COMMON_ARGS+=( --no-mmap )
    [[ ${IK_DEFER_EXPERTS:-0}    == 1 ]] && COMMON_ARGS+=( --defer-experts )
    [[ ${IK_PREFETCH_EXPERTS:-0} == 1 ]] && COMMON_ARGS+=( --prefetch-experts )
    [[ -n ${IK_AMB:-} ]] && COMMON_ARGS+=( -amb "$IK_AMB" )
    [[ -n ${IK_SER:-} ]] && COMMON_ARGS+=( -ser "$IK_SER" )

    # Free-form escape hatch for anything not modelled above.
    if [[ -n ${IK_EXTRA_ARGS:-} ]]; then
        # shellcheck disable=SC2206
        local extra=( $IK_EXTRA_ARGS )
        COMMON_ARGS+=( "${extra[@]}" )
    fi

    export COMMON_ARGS
}

# Pretty-print a command so the user can copy/paste or debug it.
show_cmd() {
    local first=1
    printf '%s' "$C_DIM" >&2
    for a in "$@"; do
        if [[ $first == 1 ]]; then printf '%s' "$a" >&2; first=0
        elif [[ $a == -* ]];  then printf ' \\\n    %s' "$a" >&2
        else                       printf ' %q' "$a" >&2; fi
    done
    printf '%s\n' "$C_OFF" >&2
}

# Optional CPU pinning. On this hybrid Intel part the P-cores are 0-5 and the
# E-cores 6-17; pinning to P-cores helps when IK_THREADS is small.
maybe_taskset() {
    if [[ -n ${IK_CPU_LIST:-} ]]; then
        echo "taskset -c $IK_CPU_LIST"
    fi
}
