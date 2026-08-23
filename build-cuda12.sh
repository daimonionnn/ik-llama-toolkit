#!/usr/bin/env bash
# =============================================================================
# build-cuda12.sh -- build ik_llama.cpp with CUDA 12.8 inside a container
# =============================================================================
# Why: on Blackwell (sm_120) a CUDA 13.x-built llama.cpp loses most of its
# throughput once the KV cache passes 8192 tokens - the same source built with
# CUDA 12.8 does not. Proven on the mainline engine; see
# ~/development/multi-gpu-llm-toolkit/doc/cuda-fa-blackwell.md. CUDA 12.8 cannot be
# installed on this host (glibc 2.43 rejects its headers), so the build runs in
# an Ubuntu 22.04 container and the CUDA 12 runtime libraries are bundled next
# to the binaries.
#
#   ./build-cuda12.sh              build into ik_llama.cpp/build-cuda12
#   ./build-cuda12.sh --image IMG  use a different CUDA 12.x devel image
#
# Run servers against it with (IK_BIN is honoured by lib/common.sh; the
# LD_LIBRARY_PATH is needed because the bundled CUDA 12 libs are not on the
# host and the container build's RPATH points at container paths):
#   IK_BIN=$PWD/ik_llama.cpp/build-cuda12/bin \
#   LD_LIBRARY_PATH=$PWD/ik_llama.cpp/build-cuda12/bin ./serve.sh <profile>
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

IMAGE="nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04"
ARCH="${IK_CUDA_ARCH:-120a-real}"
[[ "${1:-}" == "--image" ]] && { IMAGE="${2:?}"; shift 2; }

command -v docker >/dev/null || { echo "docker is required"; exit 1; }
[[ -d ik_llama.cpp ]] || { echo "ik_llama.cpp checkout missing - run ./build.sh once first"; exit 1; }

echo "==> building ik_llama.cpp with CUDA 12.8 for $ARCH (container: $IMAGE)"
docker run --rm \
    -v "$PWD/ik_llama.cpp":/src \
    -e ARCH="$ARCH" \
    "$IMAGE" bash -c '
set -e
chmod 1777 /tmp
apt-get update -qq >/dev/null 2>&1 || apt-get update
# Ubuntu 22.04 ships CMake 3.22, which does not know the CUDA20 dialect that
# ik_llama.cpp requires ("CMake does not know the compile flags to use").
apt-get install -y -qq ninja-build git python3-pip > /dev/null
pip3 install --quiet --upgrade cmake
hash -r
cmake --version | head -1
cmake -S /src -B /src/build-cuda12 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CCACHE=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" > /dev/null
cmake --build /src/build-cuda12 -j "$(nproc)" \
  --target llama-server --target llama-bench --target llama-sweep-bench 2>&1 | tail -2
# The host has no CUDA 12 runtime; ship the ones the binaries were linked against.
# Clear previously bundled CUDA libs first. Without this, building against a
# different minor version leaves BOTH sets side by side -- 12.8 and 12.9 were
# present together on 2026-08-23 -- and which one loads then rests on a symlink
# nobody checked. One version in the directory, no ambiguity to resolve later.
rm -f /src/build-cuda12/bin/libcudart.so.* /src/build-cuda12/bin/libcublas.so.* \
      /src/build-cuda12/bin/libcublasLt.so.* /src/build-cuda12/bin/libnccl.so.* 2>/dev/null || true
for lib in libcudart libcublas libcublasLt libnccl; do
    src="$(ldconfig -p | grep -oE "/[^ ]*${lib}\.so\.[0-9]+" | head -1)"
    [ -n "$src" ] && cp -a "$src"* /src/build-cuda12/bin/ 2>/dev/null || true
done
# ggml/llama/mtmd land in their own subdirectories; llama-server needs them
# beside it because the container build RPATH does not survive to the host.
find /src/build-cuda12 -name "*.so" -type f ! -path "*/bin/*" \
     -exec cp -a {} /src/build-cuda12/bin/ \; 2>/dev/null || true
# The container runs as root; hand the tree back so the host user can use it.
chown -R "$(stat -c %u:%g /src)" /src/build-cuda12
' || { echo "container build failed"; exit 1; }

BIN="ik_llama.cpp/build-cuda12/bin"
[[ -x "$BIN/llama-server" ]] || { echo "build finished but llama-server missing"; exit 1; }

echo "==> built: $BIN"
echo "    CUDA runtime bundled: $(ls "$BIN"/libcudart.so.* 2>/dev/null | head -1 | xargs -r basename)"
echo "    arch: $(grep -E '^CMAKE_CUDA_ARCHITECTURES' ik_llama.cpp/build-cuda12/CMakeCache.txt | cut -d= -f2)"
echo
echo "    use with:"
echo "      IK_BIN=$PWD/$BIN LD_LIBRARY_PATH=$PWD/$BIN ./serve.sh <profile>"
