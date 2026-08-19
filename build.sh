#!/usr/bin/env bash
# =============================================================================
# build.sh -- compile ik_llama.cpp for this machine
# =============================================================================
#   ./build.sh            configure (if needed) and build
#   ./build.sh --clean    wipe the build directory first
#   ./build.sh --update   git pull, then rebuild from scratch
#
# --update warns first if the clone carries local source changes. It has to: the
# clone is gitignored here, so a patch applied to it exists in no commit at all.
# The re-appliable copies live in docs/external/*.patch.
#
# The awkward part on this box is CUDA: the RTX PRO 6000 Blackwell is compute
# capability 12.0 (sm_120), and nvcc only learned that target in CUDA 12.8.
# Ubuntu's default nvidia-cuda-toolkit is 12.4, which cannot emit code for this
# GPU at all. This script hunts for a new enough toolkit and refuses to build a
# binary that would not run.
# -----------------------------------------------------------------------------
set -uo pipefail

source "$(dirname "$(readlink -f "$0")")/lib/common.sh"
load_config

CLEAN=0
UPDATE=0
for arg in "$@"; do
    case "$arg" in
        --clean)  CLEAN=1 ;;
        --update) UPDATE=1; CLEAN=1 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown option: $arg" ;;
    esac
done

[[ -d $IK_SRC ]] || die "ik_llama.cpp source not found at $IK_SRC
Clone it with:
  git clone https://github.com/ikawrakow/ik_llama.cpp.git '$IK_SRC'"

# -----------------------------------------------------------------------------
# 1. Find a CUDA toolkit that can target sm_120
# -----------------------------------------------------------------------------
ARCH_NUM="${IK_CUDA_ARCH%%-*}"          # "120-real" -> "120"
ARCH_NUM="${ARCH_NUM%[af]}"              # "120a-real" -> "120": nvcc --list-gpu-arch
                                         # reports compute_120, never compute_120a,
                                         # so the arch-specific suffix must come off
                                         # before probing. It stays in IK_CUDA_ARCH,
                                         # which is what gets passed to CMake.

# Ask nvcc directly which architectures it knows. This is authoritative and,
# unlike a test compile, cannot be confused by unrelated header problems --
# which matters here, see generate_compat_headers below.
nvcc_supports_arch() {
    local nvcc="$1" arch="$2"
    "$nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_${arch}"
}

log "looking for a CUDA toolkit that supports sm_${ARCH_NUM}"

CUDA_CANDIDATES=()
[[ -n ${CUDA_HOME:-} ]] && CUDA_CANDIDATES+=( "$CUDA_HOME/bin/nvcc" )
# Newest versioned install first.
while IFS= read -r d; do
    CUDA_CANDIDATES+=( "$d/bin/nvcc" )
done < <(ls -d /usr/local/cuda-* /opt/cuda-* 2>/dev/null | sort -Vr)
CUDA_CANDIDATES+=( /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc )
command -v nvcc >/dev/null && CUDA_CANDIDATES+=( "$(command -v nvcc)" )

NVCC=""
for cand in "${CUDA_CANDIDATES[@]}"; do
    [[ -x $cand ]] || continue
    local_ver="$("$cand" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
    if nvcc_supports_arch "$cand" "$ARCH_NUM"; then
        NVCC="$cand"; CUDA_VER="$local_ver"
        ok "using CUDA $local_ver at ${cand%/bin/nvcc}"
        break
    else
        dim "    skipping CUDA $local_ver at ${cand%/bin/nvcc} (no sm_${ARCH_NUM})"
    fi
done

if [[ -z $NVCC ]]; then
    die "no CUDA toolkit on this system can target sm_${ARCH_NUM}.

Your GPU is:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)
Compute cap:      $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo unknown)
sm_${ARCH_NUM} needs:    CUDA >= 12.8

Install one, e.g.:
  sudo apt install -y cuda-toolkit-13-1

Then re-run ./build.sh -- it picks up /usr/local/cuda-13.1 automatically."
fi

CUDA_ROOT="${NVCC%/bin/nvcc}"
export CUDACXX="$NVCC"
export PATH="$CUDA_ROOT/bin:$PATH"

# -----------------------------------------------------------------------------
# 2. Work around the glibc 2.41+ / CUDA 13.x rsqrt clash
# -----------------------------------------------------------------------------
# glibc >= 2.41 declares the C23 functions rsqrt/rsqrtf in <math.h> as
# noexcept. CUDA declares its own __host__ __device__ rsqrt/rsqrtf with no
# exception specification. The nvcc 13.1 front end rejects that mismatch as a
# hard error, so *every* .cu file fails -- even an empty one, because nvcc
# pre-includes cuda_runtime.h. (CUDA 12.4 tolerated it; this is a regression.)
#
# It cannot be fixed with a flag: -U_GNU_SOURCE breaks libstdc++, the
# diagnostic is not suppressible, and nvcc always pre-includes its own headers
# before any --pre-include of ours. So we put a <math.h> wrapper earlier in the
# include path that renames glibc's declarations out of the way and forwards to
# the real header. CUDA's rsqrt/rsqrtf then win, which is what device code
# wants anyway.
COMPAT_DIR="$TOOLKIT_ROOT/compat"

generate_compat_headers() {
    mkdir -p "$COMPAT_DIR"
    {
        cat <<'HDR'
/* Generated by build.sh -- do not edit.
   glibc >= 2.41 declares C23 rsqrt/rsqrtf as noexcept; CUDA declares its own
   __host__ __device__ rsqrt/rsqrtf without one, and nvcc 13.x rejects the
   mismatch. Hide glibc's declarations and let CUDA's win. */
#ifndef IK_LLAMA_MATH_H_WRAPPER
#define IK_LLAMA_MATH_H_WRAPPER
#define rsqrt  __ik_llama_hidden_rsqrt
#define rsqrtf __ik_llama_hidden_rsqrtf
/* glibc pastes the (now renamed) name into a SIMD-attribute macro, once per
   floating-point variant. Neutralise every one of them. */
HDR
        for s in "" f l f16 f32 f64 f128 f32x f64x f128x bf16; do
            echo "#define __DECL_SIMD___ik_llama_hidden_rsqrt${s}"
            echo "#define __DECL_SIMD___ik_llama_hidden_rsqrtf${s}"
        done
        cat <<'HDR'
#include_next <math.h>
#undef rsqrt
#undef rsqrtf
#endif
HDR
    } > "$COMPAT_DIR/math.h"
}

# Only interpose when this toolkit actually needs it.
needs_math_compat() {
    local tmp rc=1
    tmp="$(mktemp -d)"
    printf '#include <cmath>\nint main(){return 0;}\n' > "$tmp/probe.cu"
    "$NVCC" -arch="sm_${ARCH_NUM}" -c "$tmp/probe.cu" -o "$tmp/probe.o" >/dev/null 2>&1 || rc=0
    rm -rf "$tmp"
    return $rc
}

CUDA_COMPAT_FLAGS=""
if needs_math_compat; then
    generate_compat_headers
    CUDA_COMPAT_FLAGS="-I$COMPAT_DIR"
    warn "CUDA $CUDA_VER clashes with this glibc over rsqrt/rsqrtf"
    ok "interposing a <math.h> wrapper from $COMPAT_DIR"
else
    rm -rf "$COMPAT_DIR"
fi

# -----------------------------------------------------------------------------
# 3. Pick a host compiler nvcc will accept
# -----------------------------------------------------------------------------
# nvcc hard-fails on host compilers newer than it knows about, and this box has
# GCC 15 as the default. Probe downwards until one works.
probe_host_compiler() {
    local nvcc="$1" cc="$2" tmp rc=1
    tmp="$(mktemp -d)"
    echo '#include <cstdio>
#include <cmath>
int main(){return 0;}' > "$tmp/probe.cu"
    # shellcheck disable=SC2086
    if "$nvcc" -ccbin "$cc" -arch="sm_${ARCH_NUM}" $CUDA_COMPAT_FLAGS \
               -c "$tmp/probe.cu" -o "$tmp/probe.o" >/dev/null 2>&1; then
        rc=0
    fi
    rm -rf "$tmp"
    return $rc
}

HOST_CC=""; HOST_CXX=""
for v in "" -15 -14 -13 -12; do
    cc="/usr/bin/gcc${v}"; cxx="/usr/bin/g++${v}"
    [[ -x $cc && -x $cxx ]] || continue
    if probe_host_compiler "$NVCC" "$cxx"; then
        HOST_CC="$cc"; HOST_CXX="$cxx"
        ok "host compiler: $($cxx --version | head -1)"
        break
    else
        dim "    nvcc rejects $($cxx --version | head -1)"
    fi
done
[[ -n $HOST_CXX ]] || die "no GCC on this system is accepted by $NVCC
Try: sudo apt install -y gcc-13 g++-13"

# -----------------------------------------------------------------------------
# 4. Configure and build
# -----------------------------------------------------------------------------
if [[ $UPDATE == 1 ]]; then
    log "updating ik_llama.cpp"

    # The clone is gitignored by this repo, so any local patch lives ONLY in its
    # working tree -- there is no commit anywhere that holds it. Currently that
    # is the --cache-ram fix (docs/external/local-cache-limit.patch, RESULTS
    # §31). `pull --ff-only` will not overwrite such a change silently: it stops
    # if upstream touched the same file. But it stops with a message about the
    # merge, not about the patch, which reads as a mystery unless you already
    # know the patch is there. So say so first.
    dirty=$(git -C "$IK_SRC" status --porcelain -- '*.c' '*.cpp' '*.h' '*.cu' '*.cuh')
    if [[ -n $dirty ]]; then
        warn "the clone carries local source changes, which no commit holds:"
        while read -r line; do warn "  $line"; done <<< "$dirty"
        warn "If the pull below aborts, upstream touched one of these. Recover with"
        warn "  git -C $IK_SRC stash && ./build.sh --update"
        warn "then re-apply from docs/external/*.patch -- check they still apply first."
    fi

    git -C "$IK_SRC" pull --ff-only || die "git pull failed"
fi

if [[ $CLEAN == 1 && -d $IK_BUILD ]]; then
    log "removing $IK_BUILD"
    rm -rf "$IK_BUILD"
fi

JOBS="${IK_BUILD_JOBS:-$(nproc)}"

CMAKE_ARGS=(
    -S "$IK_SRC"
    -B "$IK_BUILD"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_ARCHITECTURES="$IK_CUDA_ARCH"
    -DCMAKE_CUDA_COMPILER="$NVCC"
    -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX"
    -DCMAKE_C_COMPILER="$HOST_CC"
    -DCMAKE_CXX_COMPILER="$HOST_CXX"
    # -march=native: this CPU has AVX2 + AVX-VNNI (no AVX-512), which the CPU
    # side of the MoE kernels uses heavily.
    -DGGML_NATIVE=ON
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_TESTS=OFF
    # No libcurl-dev on this box, and we load models from local disk anyway.
    -DLLAMA_CURL=OFF
    # q8_0 KV with flash attention is in the default kernel set for head_size
    # 128, which is what step35 uses -- no need for the (very slow to compile)
    # all-quants variant.
    -DGGML_CUDA_FA_ALL_QUANTS=OFF
)

# --- Pin the CUDA runtime libraries to the SAME toolkit as nvcc --------------
# This box also has Ubuntu's nvidia-cuda-toolkit 12.4 installed, whose stub
# libraries live in /usr/lib/x86_64-linux-gnu -- a directory CMake's
# FindCUDAToolkit searches with high priority. The result is a silent, lethal
# mismatch: device code compiled against CUDA 13.1 headers, linked against the
# 12.4 runtime. The cudaDeviceProp struct grew between those versions, so
# cudaGetDeviceProperties() fills a 12.4-layout struct that ggml then reads at
# 13.1 offsets. sharedMemPerBlockOptin comes back as 1 instead of 101376, the
# MMQ kernels find no tile that fits, and every quantized matmul aborts with
# "mmq_x_best=0". It builds and loads fine -- it only dies at first inference.
#
# Force every CUDA library ggml links to come from nvcc's own toolkit, and bake
# that lib dir into the RPATH so the binaries resolve the .so.13 files at run
# time regardless of ldconfig ordering.
CUDA_LIBDIR=""
for d in "$CUDA_ROOT/targets/x86_64-linux/lib" "$CUDA_ROOT/lib64" "$CUDA_ROOT/lib"; do
    [[ -f "$d/libcudart.so" ]] && { CUDA_LIBDIR="$d"; break; }
done
if [[ -n $CUDA_LIBDIR ]]; then
    ok "pinning CUDA runtime libs to $CUDA_LIBDIR"
    CMAKE_ARGS+=(
        -DCUDAToolkit_ROOT="$CUDA_ROOT"
        -DCMAKE_LIBRARY_PATH="$CUDA_LIBDIR"
        -DCUDA_CUDART="$CUDA_LIBDIR/libcudart.so"
        -DCUDA_cudart_LIBRARY="$CUDA_LIBDIR/libcudart.so"
        -DCUDA_cublas_LIBRARY="$CUDA_LIBDIR/libcublas.so"
        -DCUDA_cublasLt_LIBRARY="$CUDA_LIBDIR/libcublasLt.so"
        -DCMAKE_BUILD_RPATH="$CUDA_LIBDIR"
        -DCMAKE_INSTALL_RPATH="$CUDA_LIBDIR"
    )
    [[ -f "$CUDA_LIBDIR/libcudart_static.a" ]] && \
        CMAKE_ARGS+=( -DCUDA_cudart_static_LIBRARY="$CUDA_LIBDIR/libcudart_static.a" )
else
    warn "could not locate the runtime lib dir under $CUDA_ROOT -- skipping the"
    warn "runtime-library pin. If inference aborts with 'mmq_x_best=0', this is why."
fi

# The <math.h> interposer, if this toolkit needs it. CUDA sources only -- the
# pure host C/C++ files never see CUDA's declarations, so they have no clash.
[[ -n $CUDA_COMPAT_FLAGS ]] && CMAKE_ARGS+=( -DCMAKE_CUDA_FLAGS="$CUDA_COMPAT_FLAGS" )

command -v ccache >/dev/null && CMAKE_ARGS+=( -DGGML_CCACHE=ON )

log "configuring"
show_cmd cmake "${CMAKE_ARGS[@]}"
cmake "${CMAKE_ARGS[@]}" || die "cmake configure failed"

log "building with $JOBS jobs (CUDA kernels take a while -- 15-30 min cold)"
cmake --build "$IK_BUILD" --config Release -j "$JOBS" \
      --target llama-server llama-bench llama-sweep-bench llama-cli \
               llama-quantize llama-perplexity \
    || die "build failed"

echo >&2
ok "built:"
for b in llama-server llama-bench llama-sweep-bench llama-cli llama-quantize llama-perplexity; do
    if [[ -x "$IK_BIN/$b" ]]; then
        printf '       %s\n' "$IK_BIN/$b" >&2
    else
        warn "  missing: $b"
    fi
done

echo >&2
log "next: ./serve.sh        (start the server)"
log "      ./bench.sh quick  (sanity benchmark)"
