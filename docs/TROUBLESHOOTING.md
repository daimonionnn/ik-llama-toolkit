# Troubleshooting

---

## Build

### `no CUDA toolkit on this system can target sm_120`

The expected error on a fresh machine. The RTX PRO 6000 Blackwell is compute
capability 12.0, and `nvcc` gained `sm_120` in CUDA 12.8. Ubuntu ships
`nvidia-cuda-toolkit` 12.4, which tops out at `sm_90`.

```bash
sudo apt install -y cuda-toolkit-13-1
./build.sh
```

`build.sh` finds `/usr/local/cuda-13.1` on its own. To force a specific one:

```bash
CUDA_HOME=/usr/local/cuda-13.1 ./build.sh
```

Verify the driver is happy with it:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### Compile error: `exception specification is incompatible … rsqrt`

Handled automatically, documented here so the mechanism is not a mystery.

glibc 2.41+ added the C23 functions `rsqrt`/`rsqrtf` to `<math.h>`, declared
`noexcept`. CUDA declares its own `__host__ __device__ rsqrt`/`rsqrtf` without
an exception specification, and the CUDA 13.x front end rejects the mismatch as
a hard error — so *every* `.cu` file fails to compile, even an empty one,
because nvcc pre-includes `cuda_runtime.h`. CUDA 12.4 tolerated it; 13.x does
not.

`build.sh` detects this and drops a `<math.h>` wrapper in `compat/` that renames
glibc's declarations aside and forwards to the real header, so CUDA's versions
win (which is what device code wants anyway). It is applied only when the
active toolkit actually needs it, via `-I compat` on CUDA sources only. If you
ever build ik_llama.cpp by hand on this box, add `-Icompat` to `CMAKE_CUDA_FLAGS`
or expect this error.

### Inference aborts immediately with `mmq_x_best=0`

The nastiest failure on this machine, because it builds and loads perfectly and
only dies at the first matrix multiply. `build.sh` now prevents it; this is what
it was.

The box has two CUDA runtimes: the 12.4 one from Ubuntu's `nvidia-cuda-toolkit`
(stub libraries in `/usr/lib/x86_64-linux-gnu`) and the 13.1 one we installed.
CMake's `FindCUDAToolkit` searches that system directory with high priority, so
by default it compiled device code against 13.1 headers but linked the **12.4**
`libcudart`. The `cudaDeviceProp` struct grew between those versions, so
`cudaGetDeviceProperties()` fills a 12.4-layout struct that ggml reads at 13.1
offsets. `sharedMemPerBlockOptin` comes back as **1** instead of 101376, the
MMQ kernels can't find a shared-memory tile that fits, and every quantized
matmul hits `GGML_ABORT("fatal error")` after printing `mmq_x_best=0`.

`build.sh` pins every CUDA library ggml links (`cudart`, `cublas`, `cublasLt`)
to nvcc's own toolkit and bakes that lib dir into the RPATH. To confirm a built
binary is clean, every CUDA line should say `.so.13` (only the driver
`libcuda.so.1` comes from the system):

```bash
ldd ik_llama.cpp/build/bin/llama-server | grep -iE 'cudart|cublas'
# libcudart.so.13   => /usr/local/cuda-13.1/...
# libcublas.so.13   => /usr/local/cuda-13.1/...
# libcublasLt.so.13 => /usr/local/cuda-13.1/...
```

If any says `.so.12`, the build picked up the wrong runtime — rebuild clean
with `./build.sh --clean`.

### `nvcc rejects g++ (Ubuntu 15.2.0)`

Not an error — `build.sh` printing that it fell back. `nvcc` refuses host
compilers newer than it knows about; the script probes GCC 15 → 14 → 13 → 12
and uses the first that works. (With the `<math.h>` wrapper above in place, GCC
15 is actually accepted on this box, so you should not see this.) If *all* are
rejected:

```bash
sudo apt install -y gcc-13 g++-13
```

### `unsupported architecture` or PTX errors at runtime

The binary was built for the wrong compute capability. Rebuild clean:

```bash
./build.sh --clean
```

### `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed`

Long prefills on DeepSeek-class models used to abort here: the sparse-attention
masks produce more than 32 tensors crossing the CPU/GPU split. **Fixed
upstream** — pull and rebuild (`./build.sh --update`). Verified with
65 536-token prefills on MXFP4.

### Should I worry about CUDA 13 on Blackwell?

Not for this engine. Mainline llama.cpp built with CUDA 13.x loses most of its
throughput on `sm_120` past 8192 context, and `build-cuda12.sh` exists as
insurance against that — but measurement shows ik_llama is unaffected, because
its MLA path does not use the flash-attention kernels involved. Build with the
host toolkit; see [TUNING.md §9](TUNING.md).

### The build takes forever

Normal: 15–30 minutes cold, almost all of it CUDA kernels. Installing `ccache`
before the first build makes rebuilds dramatically faster:

```bash
sudo apt install -y ccache && ./build.sh --clean
```

---

## Starting the server

### `model 'X' is incomplete -- wait for the download to finish`

LM Studio writes `downloading_*.part` files next to finished shards.
`serve.sh` refuses to start until all shards of the set are present — loading a
half-written GGUF produces confusing failures much later.

```bash
ls -la "$IK_MODELS_ROOT/unsloth/Step-3.7-Flash-GGUF/"
```

### `other processes are holding GPU memory`

LM Studio and Ollama keep models resident long after their last request. This
is worth taking seriously rather than ignoring: `--fit` sizes the GPU/CPU split
from *free* VRAM at launch, so starting with 20 GiB free instead of 95 GiB
quietly moves ~28 extra layers of experts into system RAM.

```bash
# quit LM Studio, then:
ollama stop <model>          # or: systemctl --user stop ollama

# or let the toolkit do it:
IK_KILL_SQUATTERS=1 ./serve.sh
```

### CUDA out of memory during warmup

`--fit` reserves `IK_FIT_MARGIN` MiB (2048 by default), which can be too tight
if a desktop session is also drawing on the GPU. In order of what to try:

```bash
IK_FIT_MARGIN=4096 ./serve.sh    # more headroom
IK_UBATCH=512 ./serve.sh         # smaller compute buffers
IK_CTX=32768 ./serve.sh          # smaller KV cache
IK_NCMOE=20 IK_FIT=0 ./serve.sh  # take manual control
```

### Loading takes minutes

The first start after boot reads ~114 GiB from NVMe. Later starts hit the page
cache and are much faster. Things that make it worse:

- `IK_RTR=1` forces `--no-mmap`, so the whole file is re-read **every** start.
- `IK_NO_MMAP=1` likewise.

`IK_DEFER_EXPERTS=1` makes the load return sooner by faulting expert pages in
on demand, but moves the cost into the first few responses.

---

## Output quality

### Gibberish or incoherent replies

Known upstream interaction between split-mode `graph` and partial offload.
ik_llama's README flags it directly:

```bash
IK_EXTRA_ARGS="-cuda graphs=0" ./serve.sh
```

If that fixes it, keep the flag in the profile and note why.

Otherwise, work through the quality-affecting settings in order:

1. `IK_SER` — if set, unset it. It runs fewer than 8 experts and *does* change
   output.
2. `IK_CTK`/`IK_CTV` — set both to `f16` to rule the KV cache out.
3. `IK_JINJA` — `0` falls back to a generic chat template, which will not match
   what this model was trained on. Leave it at `1`.

### Replies degrade in long conversations

Check you have not exceeded `IK_CTX`. 33 of 45 layers use a 512-token sliding
window, so the model is designed for this — but the 12 full-attention layers
still need cache, and past `IK_CTX` the server starts discarding context.

```bash
IK_CTX=131072 ./serve.sh    # costs ~2 GiB of KV, about one layer of experts
```

---

## Performance

### Generation is slower than expected

In rough order of likelihood:

1. **Something else is on the GPU.** Check `nvidia-smi`. This is the usual
   answer.
2. **Too many expert layers on the CPU.** The server log prints the split at
   load time. Each CPU layer costs about 1 ms per token
   ([TUNING.md §1](TUNING.md#what-it-costs)).
3. **Thread count.** Run `./bench.sh threads`. Generation wants ~6 (the P-core
   count); more can be slower because every barrier waits on the slowest thread.
4. **Expert pages evicted from the page cache.** If something else consumed
   RAM, the CPU-side experts get re-read from disk. Check `free -g` — `buff/cache`
   should be large.
5. **Thermal throttling.** `watch -n1 'grep MHz /proc/cpuinfo | sort -u'`.

### Prompt processing is slow

Raise `IK_UBATCH` (1024 → 2048) and confirm `IK_THREADS_BATCH` is 18. Verify
with `./bench.sh batch`. Watch that `tg` does not fall in exchange — bigger
compute buffers mean fewer experts on the GPU.

### It was fast yesterday and is slow today

Nothing about the config changed, so look at state:

```bash
nvidia-smi                   # who is holding VRAM now?
free -g                      # is buff/cache still large?
./serve.sh --dry-run         # is the command what you expect?
```

`--fit` adapting to a GPU that is 20 GiB busier is by far the most common cause.

---

## Getting more detail

```bash
./serve.sh --dry-run              # exact command, nothing executed
./serve.sh -- --verbose           # verbose llama-server logging
tail -f logs/server-*.log         # every start is logged
```

The load-time log lines listing buffer sizes per device are the ground truth
for where tensors actually ended up — trust them over the estimates in
[TUNING.md](TUNING.md).

Upstream references:

- [`ik_llama.cpp/docs/parameters.md`](../ik_llama.cpp/docs/parameters.md) — every flag
- [ik_llama.cpp issues](https://github.com/ikawrakow/ik_llama.cpp/issues)
