#!/usr/bin/env bash
# =============================================================================
# check-driver-change.sh -- what to re-check after an NVIDIA driver change
# =============================================================================
# Every number in docs/RESULTS.md was measured on driver 595.84 / CUDA 13.2
# runtime / Ubuntu 26.04 (kernel 7.0.0-29). Two findings depend on the driver
# specifically, so after an upgrade run this before trusting the tables:
#
#   1. cudaDevAttrHostRegisterReadOnlySupported. It reads 0 on 595.84, which is
#      the first of the four causes behind ds4 being unable to start (RESULTS
#      §13.2, antirez/ds4#791). If a newer driver reports 1, ds4's own
#      registration path may work without the local patches -- though causes
#      3 and 4 would still leave it at ~0.54 tok/s, so expect "starts but
#      useless" rather than "fixed".
#   2. Whether the shipped profile still performs. A driver change can move
#      prefill/generation on its own.
#
# Usage:  ./check-driver-change.sh          # attributes only, seconds
#         ./check-driver-change.sh --bench  # also re-runs the default profile
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

BASELINE_DRIVER="595.84"
BASELINE_PP="499"      # kvram 128k default, 32k prompt, RESULTS §14
BASELINE_TG="21.3"

echo "=== driver / toolkit ==="
now="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
printf '  driver now      : %s   (measurements were taken on %s)\n' "$now" "$BASELINE_DRIVER"
nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/  CUDA runtime    : \1/p' | head -1
printf '  nvcc            : %s\n' "$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
printf '  kernel          : %s\n' "$(uname -r)"

echo
echo "=== device attributes that RESULTS depends on ==="
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/a.cu" <<'EOF'
#include <cstdio>
int main(){
    int hr=0, ro=0, pa=0, hpt=0;
    cudaDeviceGetAttribute(&hr,  cudaDevAttrHostRegisterSupported, 0);
    cudaDeviceGetAttribute(&ro,  cudaDevAttrHostRegisterReadOnlySupported, 0);
    cudaDeviceGetAttribute(&pa,  cudaDevAttrPageableMemoryAccess, 0);
    cudaDeviceGetAttribute(&hpt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    printf("  HostRegisterSupported          = %d   (was 1)\n", hr);
    printf("  HostRegisterReadOnlySupported  = %d   (was 0)  <- the ds4 one\n", ro);
    printf("  PageableMemoryAccess           = %d   (was 1)\n", pa);
    printf("  ...UsesHostPageTables          = %d   (was 0)\n", hpt);
    return ro;   /* exit 1 while still unsupported, 0 if it flipped */
}
EOF
if nvcc="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)" && [[ -x $nvcc ]]; then
    if "$nvcc" -o "$tmp/a" "$tmp/a.cu" 2>/dev/null; then
        "$tmp/a"; ro_rc=$?
        echo
        if [[ $ro_rc -eq 0 ]]; then
            echo "  >> ReadOnly is STILL unsupported. ds4 needs the local patches"
            echo "     (docs/external/ds4-blackwell-discrete-fixes.patch)."
        else
            echo "  >> ReadOnly is now SUPPORTED. Worth retrying ds4 on clean upstream:"
            echo "     git -C ~/development/ds4 checkout 84cc882 -- ds4.c ds4_cuda.cu"
            echo "     ...but expect causes 3 and 4 (RESULTS §13.2) to still bite."
        fi
    else
        echo "  (could not compile the probe -- check the CUDA toolkit)"
    fi
else
    echo "  (nvcc not found)"
fi

if [[ ${1:-} == --bench ]]; then
    echo
    echo "=== re-measuring the shipped default (config/default.env, 32k prompt) ==="
    echo "    baseline on ${BASELINE_DRIVER}: ${BASELINE_PP} tok/s prefill / ${BASELINE_TG} generation"
    echo "    NOTE: the baseline above was taken on the DeepSeek kvram profile that"
    echo "          was the default until 2026-09-03. If default.env now points"
    echo "          elsewhere the comparison is apples to oranges -- set"
    echo "          IK_PROFILE to the profile the baseline belongs to."
    # Port 8090 explicitly, matching the probe below. (default.env also says
    # 8090; passing it here keeps this working if that ever changes.)
    IK_PORT=8090 IK_KILL_SQUATTERS=1 ./serve.sh > /tmp/driver-check-server.log 2>&1 &
    pid=$!
    for ((i=0;i<400;i++)); do
        grep -q 'HTTP server listening' /tmp/driver-check-server.log 2>/dev/null && break
        kill -0 $pid 2>/dev/null || { echo "    server failed to start -- see /tmp/driver-check-server.log"; exit 1; }
        sleep 5
    done
    sleep 3
    python3 - <<'PY'
import json, urllib.request
api="http://127.0.0.1:8090"
def post(p, pl, t=3600):
    r=urllib.request.Request(api+p, data=json.dumps(pl).encode(), headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))
ntok=lambda s: len(post("/tokenize",{"content":s},900)["tokens"])
salt="drvchk"; depth=32768
head=f"Document set {salt}. Reference corpus, revision {salt}-{depth}.\n"
line=lambda i: f"[{salt}] Record {i}: sensor {i%89} reported value {i*7%1000} at cycle {i%433}, status nominal.\n"
per=ntok("".join(line(i) for i in range(200)))/200
need=max(1,int((depth-ntok(head)-200)/per))
body=head+"".join(line(i) for i in range(need))
t=post("/v1/chat/completions", {"model":"deepseek-v4-flash-128k-kvram",
    "messages":[{"role":"user","content":body+"\nWrite one paragraph about the sea."}],
    "max_tokens":160,"temperature":0.0,"stream":False})["timings"]
print(f'    now: {t["prompt_per_second"]:.1f} tok/s prefill / {t["predicted_per_second"]:.2f} generation')
PY
    ./stop.sh > /dev/null 2>&1; wait $pid 2>/dev/null
fi
