#!/usr/bin/env bash
# Rebuilds the depth corpus for qwen38-q8-mtp-depth-20260914.py, byte for byte,
# from git: this repository's docs/*.md as of af7cf52 (the tree §54 measured
# from), then lib/common.sh and tools/depthbench.sh from the same commit, then
# three ik_llama.cpp sources as of d5f53d9f (upstream, without the keep-*
# patches). §54's own corpus.txt was not kept; this is the closest rebuild of
# "the docs, then lib/common.sh, tools/depthbench.sh, build_qwen4exp.cpp,
# speculative.cpp and server-context.cpp". At 124k tokens only the docs and the
# first three source files are reached.
#
#   results/qwen38-q8-mtp-corpus-20260929.sh > corpus.txt
set -euo pipefail
root="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
for f in $(git -C "$root" ls-tree --name-only af7cf52 docs/ | grep '\.md$' | sort); do
    git -C "$root" show "af7cf52:$f"
done
git -C "$root" show af7cf52:lib/common.sh
git -C "$root" show af7cf52:tools/depthbench.sh
for f in src/graphs/build_qwen4exp.cpp common/speculative.cpp examples/server/server-context.cpp; do
    git -C "$root/ik_llama.cpp" show "d5f53d9f:$f"
done
