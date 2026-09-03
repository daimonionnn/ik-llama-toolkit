# FAQ

Common questions about ik_llama.cpp and this toolkit — platform support,
portability, and why the setup on this particular machine was bumpy. For
*tuning* questions (speed, context, expert placement) see
[TUNING.md](TUNING.md); for *errors* see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Setup & platform

### Why was adapting ik_llama.cpp to this machine so painful?

Almost none of it was ik_llama.cpp's fault. It was a "everything is
brand-new at once" integration problem, specific to this box:

- **The GPU is a brand-new generation.** The RTX PRO 6000 Blackwell is compute
  capability `sm_120`, which `nvcc` can only target from CUDA **12.8** onward.
  The system shipped with Ubuntu's CUDA **12.4**, which physically cannot emit
  code for this card (it tops out at `sm_90`).
- **Two CUDA toolkits side by side.** After installing CUDA 13.1, the old 12.4
  was still present, and CMake linked the *wrong* runtime. The `cudaDeviceProp`
  struct changed between the two, so a field (`sharedMemPerBlockOptin`) was read
  as garbage → every quantized matmul aborted with `mmq_x_best=0`. It built and
  loaded fine and only died at the first token.
- **A very new glibc (2.43).** It added the C23 `rsqrt`/`rsqrtf` functions,
  which clash with CUDA's own declarations and make nvcc 13.x reject *every*
  `.cu` file.

All three are bleeding-edge-hardware / messy-system issues. On an ordinary box —
say an RTX 4090 with a matching CUDA and an older glibc — this builds with a
single `cmake` command and none of the above happens. `build.sh` now detects and
works around each one automatically; the details are in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Wasn't ik_llama.cpp built for CUDA in the first place?

Yes — **CUDA is its primary, best-supported backend.** ik_llama.cpp is a fork of
llama.cpp focused precisely on fast hybrid CPU+CUDA inference (MLA, fused MoE,
row-interleaved `_R4` quant packing, tensor overrides). The problem was never
"does it support CUDA" — it fully does. The problem was that this machine's
*toolchain* couldn't target this *specific new GPU generation*, on top of a
messy multi-CUDA + new-glibc environment. Once pointed at CUDA 13.1 with the
runtime libraries pinned correctly, the CUDA backend ran perfectly (~26 tok/s on
a 220 B model).

### Does ik_llama.cpp run on AMD GPUs (ROCm)?

Yes. The build has a `GGML_HIPBLAS` option and extensive HIP/ROCm code paths
(RDNA1/2/3 handling) — the CUDA backend is compiled through HIP for AMD. There
are also **Vulkan** and Apple **Metal** backends.

- AMD via ROCm: configure with `-DGGML_HIPBLAS=ON` (ROCm installed).
- Caveat: NVIDIA/CUDA is the most-tested path, and the `_R4` / new-quant / MoE
  optimizations are most mature there. AMD works, but expect rougher edges.

This toolkit's `build.sh` is written for the CUDA path; using ROCm/Vulkan/Metal
means invoking CMake yourself with the corresponding backend flag.

### Can I use this toolkit on a different NVIDIA RTX card?

Yes, and it's essentially a one-line change. The build architecture is
`IK_CUDA_ARCH` in [`config/default.env`](../config/default.env) (currently
`120-real` for Blackwell). Set it to your card's compute capability:

| card                     | `IK_CUDA_ARCH` |
|--------------------------|----------------|
| RTX 30xx (Ampere)        | `86-real`      |
| RTX 40xx (Ada Lovelace)  | `89-real`      |
| RTX 50xx / PRO Blackwell | `120-real`     |
| anything (auto-detect)   | `native`       |

Two things to know:

- **Older cards (Ada / Ampere) work with the stock CUDA 12.x**, so they would
  hit *none* of the problems we did — no need for CUDA 13.1, and the glibc issue
  depends on the OS, not the GPU. `build.sh` finds the toolkit itself, and its
  fixes only activate when actually needed.
- **The tuning must be redone.** The profiles here (`IK_NCMOE`, `IK_CTX`) are
  sized for **96 GiB of VRAM**. A 24 GiB card pushes far more experts onto the
  CPU and is much slower. Re-run `./bench.sh` and re-pick the split for your
  card's VRAM (see [BENCHMARKING.md](BENCHMARKING.md) and
  [TUNING.md §1](TUNING.md)).
- **It is handled automatically, but know how.** `load_config` calls
  `autofit_unless_reference_gpu`: if `nvidia-smi` reports less than 90 000 MiB of
  *total* VRAM (or cannot be queried at all), the profile's pinned `IK_NCMOE` /
  `IK_OT` are dropped, `--fit` takes over, and a warning explains it. Without
  that, a pinned split sized for 96 GiB does not merely run slowly on a smaller
  card — it dies in `cudaMalloc` at load, usually on the KV cache, which looks
  like a broken toolkit rather than a wrong setting.
  - It keys on **total** VRAM, deliberately. Free VRAM moves — an idle ComfyUI
    was found holding 33 GiB of the reference card — so deciding on free memory
    would mean the tuned values silently stop being used whenever something else
    is resident.
  - `IK_ASSUME_REFERENCE_GPU=1` forces the pinned values; `IK_REFERENCE_VRAM_MIN`
    moves the threshold.
- **Setting `IK_FIT=1` by hand still does nothing** on a profile that pins
  `IK_NCMOE`. Argument assembly checks `IK_OT`, then `IK_NCMOE`, and only reaches
  `--fit` if neither is set — and `: "${IK_NCMOE:=17}"` counts as set even when
  you pass `IK_NCMOE=`, because `:=` fills in null values too. The automatic path
  above works because it `unset`s the variable rather than blanking it.

### Can ik_llama.cpp use several GPUs in one PC, or is it only GPU+CPU?

Both — and multi-GPU is a first-class feature, not an afterthought. The relevant
flags:

- `-sm` / `--split-mode` — `layer`, `row`, `graph`, or `attn`: how to divide the
  model across GPUs.
- `-ts` / `--tensor-split` — the ratio to split by, which also lets you balance
  across **mismatched** cards by their VRAM.
- `-mg` / `--main-gpu` — which device holds shared/intermediate data.

Crucially, `--fit` and `-ncmoe` work *across* multiple GPUs too — the
CPU-resident layers are distributed proportionally across the devices. So you can
combine **multiple GPUs + CPU RAM** in one hybrid run. Running a model larger
than any single GPU's VRAM, spread over GPU(s) + CPU, is exactly what the project
is built for.

To use a second GPU here, add `IK_EXTRA_ARGS="-sm layer"` (optionally `-ts` to
balance), and `--fit` / `-ncmoe` handle the rest.

> **For this machine specifically:** a second GPU is the single biggest speed
> lever available. At 262 144 context ~22 expert layers live in system RAM and
> the bottleneck is DDR5 bandwidth; a second card would move those layers into
> fast VRAM. This beats a faster CPU or more GPU power, both of which we measured
> as near-useless here (the GPU sits at ~23% during generation) — see
> [TUNING.md §7](TUNING.md).

---

## Hardware & bottlenecks

### Is ik_llama.cpp actually faster than LM Studio / Ollama / standard llama.cpp?

**It depends entirely on the model, and the first answer here was wrong.**

**Step-3.7-Flash (GQA MoE):** no. Generation is identical (~27 tok/s both) and
standard llama.cpp prefills **~2.4× faster** (~635 vs ~270 tok/s). Mainline
already supports `step35` and runs it out of the box.

**DeepSeek-V4-Flash (MLA + sparse attention):** yes, clearly — but only once
both sides are given a fair build. The original measurement here used an
ik binary two weeks older than its own checkout, running f16 KV and default
threads, and concluded ik was slower and crash-prone. Re-measured 2026-08-10
with a current build, q8_0 KV and 24 threads:

| MXFP4, `-ncmoe 16` | mainline (CUDA 12.8) | ik_llama |
|---|---:|---:|
| prefill @16k | ~305 | **385** |
| generation @16k | 16.4 | **19.5** |

**+26% prefill, +19% generation.** Part of that is capability rather than
speed: ik runs the KV cache at q8_0, and mainline *segfaults* with `-ctk q8_0`
on this model past ~4k context. The long-prefill crash that made the first
verdict so damning (`GGML_SCHED_MAX_SPLIT_INPUTS`) is fixed in current ik —
65k-token prefills now complete normally.

**The honest rule:** run mainline for plain GQA MoE models, run this toolkit
for DeepSeek-class MLA architectures — which is, in the end, exactly what
ik_llama.cpp claims about itself. Full numbers in
[RESULTS.md §5](RESULTS.md).

### Does the PCIe link speed affect inference? Would PCIe 5.0 x8 hurt?

**Measured** on this box by sampling PCIe throughput (`nvidia-smi dmon -s t`)
during each phase. The GPU runs at **PCIe 5.0 x16 (~63 GB/s)** under load (it
idles down to gen 1 to save power, which is normal). What actually crosses the
bus:

| phase           | avg PCIe read | peak PCIe read |
|-----------------|---------------|----------------|
| generation (tg) | 89 MB/s       | 134 MB/s       |
| prefill (pp)    | 670 MB/s      | **7.4 GB/s**   |

The answer that falls out of this:

- **Generation barely touches PCIe** — peak 134 MB/s is ~0.2% of the available
  63 GB/s. During generation the CPU-resident experts are computed *on the CPU*
  (only tiny activations cross the bus), so tg is bound by DDR5, not PCIe.
  **Slowing the link to PCIe 5.0 x8 — or even PCIe 3.0 x8 — would not change tg
  at all.**
- **Prefill is the PCIe-sensitive phase.** For a batch, the scheduler copies the
  CPU-resident expert *weights* to the GPU to do the fast batched matmul there,
  which bursts to ~7.4 GB/s. That is real traffic — but still only ~12% of PCIe
  5.0 x16, and ~24% of x8. So:
  - **PCIe 5.0 x8 (31 GB/s):** the 7.4 GB/s bursts still fit ~4× over, so
    expect only a small prefill penalty, if any.
  - Dropping much lower (**PCIe 3.0 x8 ≈ 8 GB/s**) would start to throttle those
    bursts and slow prefill noticeably.
  - **Speeding the link up** from the current x16 gen 5 gives nothing — there is
    already 8× headroom over the peak.

**Short version:** PCIe affects **prefill, not generation**, and even prefill
isn't saturating the bus here. x16 → x8 would cost you a little prefill and zero
generation; going faster than the current PCIe 5.0 x16 buys nothing.

### What upgrade actually speeds this up, then?

Neither more GPU power nor a faster CPU nor a faster PCIe link moves the needle
much — all measured, all bottlenecked elsewhere. Generation is limited by
**reading the CPU-resident experts out of DDR5**. In rough order of impact:

1. **A second GPU** — moves expert layers off the CPU into fast VRAM. Biggest
   lever. ik_llama supports it natively (`-sm layer`, `-ts`).
2. **Fewer experts on the CPU** — shorter `IK_CTX`, or `q4_0` KV to free VRAM,
   or a smaller quant. Trades context/quality for speed.
3. **Faster RAM** — was the third lever here and has since been spent: this box
   went 6267 → 6400 → 6667, and RESULTS §25/§26 measured generation tracking
   MT/s almost exactly (+5.5 % for +18 % of bandwidth). Further gains need a
   kit above 6667, and the return is proportional, not dramatic.
4. Faster CPU cores / more GPU power / faster PCIe — **near-zero** here.

The full reasoning is in [TUNING.md §7](TUNING.md).

---

## Quick pointers

| I want to…                             | See |
|----------------------------------------|-----|
| See the actual measured numbers        | [RESULTS.md](RESULTS.md) |
| Understand why a setting is what it is | [TUNING.md](TUNING.md) |
| Measure / re-tune for my hardware      | [BENCHMARKING.md](BENCHMARKING.md) |
| Fix a build or runtime error           | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| See every flag                         | [`../ik_llama.cpp/docs/parameters.md`](../ik_llama.cpp/docs/parameters.md) |
