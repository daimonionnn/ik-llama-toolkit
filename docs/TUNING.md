# Tuning

Why every default is what it is, and what to change when something about the
setup changes. All figures are for **Step-3.7-Flash Q4_K_XL** on the machine
described in the [README](../README.md).

---

## 1. The one decision that matters: where the experts live

### The arithmetic

Step-3.7-Flash is 45 layers: 3 dense, then 42 MoE. Each MoE layer holds 288
experts of three matrices each, all `4096 × 1280`:

```
per expert   3 × 4096 × 1280            = 15.7 M params
per layer    288 experts × 15.7 M       =  4.53 B params
all layers   42 × 4.53 B                =  190 B params      ← ~86% of the model
```

Everything else — embeddings, all attention, the 3 dense FFNs, the 42 shared
experts, every norm — comes to roughly **7 B parameters, about 4.4 GiB** at
Q4_K_XL. It always fits on the GPU with room to spare.

The whole file is ~114 GiB, of which ~110 GiB is experts, so:

```
one layer of experts ≈ 110 GiB / 42 ≈ 2.6 GiB
```

That number is the currency of this entire document. The question "should I
raise the context?" is really "am I willing to pay one layer of experts per
2.6 GiB?"

### The VRAM budget

```
  96.0 GiB   total VRAM
-  2.0 GiB   fit margin (desktop, CUDA context, fragmentation)
-  4.4 GiB   non-expert weights
-  6.0 GiB   KV cache @ 65 536 context, q8_0   (measured — see §3)
-  3.0 GiB   CUDA compute buffers @ -ub 1024
──────────
≈ 80.6 GiB   left for experts  →  80.6 / 2.6 ≈ 31 layers on the GPU
                                  42 - 31    ≈ 11 layers on the CPU (~29 GiB RAM)
```

The KV term is the one that bites: it is not a rounding error, it is ~2.3
expert layers, and it scales with `IK_CTX` (§3). Measured reality on this box:
`-ncmoe 11` (only 8 expert layers on the CPU) loads and benchmarks fine at
*small* context — `llama-bench` allocates almost no cache — but the **server at
65 536 context OOMs on it**, because the 6 GiB cache no longer fits. The
production sweet spot lands a few layers more conservative than the bench
optimum; see §1's table of measured splits below and always confirm with a real
server start.

### Measured splits (2026-07-25, GPU idle, q8_0 KV, 6/18 threads)

The whole point, in one table. `tg` is measured from the real server at each
context; "loads" means the server actually started (KV cache and all), not just
that `llama-bench` was happy.

| context     | KV cache      | fastest `-ncmoe` that loads | tg      | notes                                          |
|-------------|---------------|-----------------------------|---------|------------------------------------------------|
| 65 536      | 6 GiB         | 11                          | ~46     | bench only; server at this ctx wants ~14 → ~38 |
| 131 072     | 12 GiB        | 16                          | ~33     | 17 → ~28, safer headroom                       |
| **262 144** | **24 GiB**    | **21**                      | **~26** | 22 → ~25 with ~4 GiB free (the default)        |
| 262 144     | 13 GiB (q4_0) | 18                          | ~30     | q4_0 KV; 16 → ~33 but VRAM basically full      |

Two things this makes concrete:

- **Context is bought with generation speed.** 65 536 → 262 144 costs ~18 GiB of
  extra KV (≈7 expert layers), dropping `tg` from ~38 to ~26. That is the price
  of the full 262k window, and it is the deliberate default here because you
  asked for maximum context.
- **`--n-cpu-moe` beats `--fit` by ~5 tok/s** because `--fit` reserves more
  headroom. The cost is that you must pick the number yourself and it is
  context-specific — lower the context, lower the `-ncmoe`.

### The three ways to place experts

```bash
IK_NCMOE=22 IK_FIT=0        # (the default) experts of layers 0..21 on CPU
IK_FIT=1                    # auto: measures free VRAM, ~5 tok/s slower, always fits
IK_OT='blk\.([3-9]|1[0-9]|2[01])\.ffn_.*_exps\.weight=CPU'   # full manual control
```

`--fit` is the safe fallback: it measures free VRAM at launch and re-decides on
every start, so it adapts when something else is using the GPU or when you
change `-c`. It is what to reach for if a fixed `-ncmoe` ever OOMs (e.g. after
raising the context). Use `IK_OT` when you want to split *within* a layer — for
example keeping `ffn_down_exps` on the GPU (read every token, the most
latency-sensitive of the three) while exiling `gate` and `up`.

> **Always confirm a fixed `-ncmoe` with a real server start at your real
> context.** `llama-bench` allocates almost no KV cache, so a split that
> benchmarks fine there can OOM the server once the multi-GiB cache is added.
> This is exactly how the numbers above were validated.

> **Off-by-three:** `--n-cpu-moe N` walks layer indices `0..N-1`, but layers
> 0, 1 and 2 are dense and own no expert tensors. `N=17` therefore places
> **14** layers of experts on the CPU, not 17.

### What it costs

Per token, each CPU-resident layer must read its 8 active experts:

```
8 experts × 3 matrices × 4096 × 1280 × ~0.56 bytes/param ≈ 70 MiB
```

At a realistic ~75 GB/s of achievable DDR5-6333 bandwidth that is **~0.95 ms
per layer per token**. Ten layers ≈ 10 ms ≈ a ceiling near 100 tok/s from
memory alone, before any compute, sync or sampling overhead. The same 10 layers
on the GPU would cost ~0.4 ms *in total*.

This is why the RAM upgrade question has a clear answer — see §7.

---

## 2. Threads

```bash
IK_THREADS=18         # generation
IK_THREADS_BATCH=18   # prompt processing
```

This is measured, and it overturned the obvious guess. The intuition was that
generation — one token at a time, thin matrix-vector products over the experts —
is pure memory streaming, so it should saturate the two DDR5 channels at ~6
threads (the P-core count) and get slower past that. **The measurement says
otherwise** (`./bench.sh threads`, at the default `ncmoe=22`):

| generation threads   | tg       |
|----------------------|----------|
| 4                    | 24.5     |
| 6                    | 25.9     |
| 8                    | 27.2     |
| 12                   | 27.3     |
| 18                   | **27.5** |
| 6, pinned to P-cores | 24.3     |

**Correction (2026-08-10): that table only measured `tg`, and `tg` is the
metric threads do *not* affect.** Re-swept on MXFP4 / `--n-cpu-moe 16` /
q8_0 KV, measuring both:

| threads | pp512 | tg128 |
|--------:|------:|------:|
| 12      | 268.6 | 24.63 |
| 16      | 316.7 | 25.03 |
| 18      | 333.4 | 24.97 |
| 20      | 350.8 | 25.07 |
| 22      | 343.0 | 25.24 |
| **24**  | **354.6** | 24.92 |

Generation is flat from 12 upward (24.6-25.2, inside the ±0.35 run-to-run
spread) because it is bound by memory bandwidth, not cores. Prefill keeps
scaling: **+32% from 12 to 24 threads, +6% over the old 18 default.** All 24
threads of an Arrow Lake-S part help here, E-cores included - the default is
now 24.

tg keeps rising to 18 cores, and pinning to the P-cores is *slower*, not faster.
The reason is the split: with 22 expert layers on the CPU (the default for
262 144 context), the per-token CPU work is large enough that it is compute- and
latency-bound, not bandwidth-bound, so the 12 E-cores contribute real work. The
"6 P-cores is enough" rule only holds when few layers are on the CPU — i.e. at
small context / low `-ncmoe`. **Re-run `./bench.sh threads` if you change the
context or split substantially**, because the optimum moves with the CPU load.

`IK_CPU_LIST` is deliberately empty: pinning measured slower than letting Intel
Thread Director place all 18 threads itself.

---

## 3. Context and the KV cache

`IK_CTX=65536` by default; the model trains to 262 144.

**Measured, and more expensive than the architecture suggests.** The model's
sliding-window pattern is `[full, swa, swa, swa]` repeating, so on paper only 12
of 45 layers should need a full-length cache and the rest could cap at their
512-token window. In practice this ik_llama build allocates a **full-length KV
cache for all 45 layers** — the windowed layers are not given the reduced
buffer. So budget for every layer:

```
per token   45 layers × 2 (K+V) × 8 heads × 128 × 1.0625 B (q8_0)  ≈  96 KiB
  → at  32 768 tokens   ≈  3.0 GiB
  → at  65 536 tokens   ≈  6.0 GiB   (measured: 6120 MiB allocated)
  → at 131 072 tokens   ≈ 12.0 GiB
  → at 262 144 tokens   ≈ 24.0 GiB
```

That is **~2.3 expert layers of VRAM at 65 536**, not the "under two layers for
256k" this section originally (wrongly) claimed. Context is one of the more
expensive things you buy here, and it trades directly against how many experts
fit on the GPU — every doubling of context is roughly one more expert layer
exiled to the CPU.

This is why `--fit` and a fixed `-ncmoe` disagree about the split: `--fit`
reserves for this full cache and is conservative; a hand-picked `-ncmoe` that
loads fine in `llama-bench` (which allocates only a tiny cache) can then OOM in
the server once the real 6 GiB cache is added. **Always validate a chosen
`-ncmoe` by starting the actual server at the real context**, not just by
benchmarking it — see §1 and [BENCHMARKING.md](BENCHMARKING.md).

`IK_CTK=IK_CTV=q8_0` halves the cache against `f16`, so it is doing real work
here: `f16` would make the 65 536 cache ~12 GiB. Head size is 128 and
`q8_0`/`q8_0` is in ik_llama's default flash-attention kernel set, so it needs
no special build flags. If VRAM is tight, lowering `IK_CTX` frees experts fast:
32 768 gives back ~3 GiB, more than a whole expert layer.

---

## 4. Batch sizes

```bash
IK_BATCH=4096
IK_UBATCH=1024
```

`-ub` is the real knob: it sets how many tokens are prefilled in one pass.
Bigger is faster for prompt processing, but the CUDA compute buffers scale with
it, and every GiB they take is a GiB not holding experts. 1024 is a reasonable
hybrid point. `./bench.sh batch` sweeps it — look for a setting that wins on
`pp` while not losing on `tg`, since a `tg` regression means you just paid for
prefill speed with an expert layer.

If a run dies during warmup with a CUDA OOM, halve `IK_UBATCH` first.

---

## 5. MoE-specific switches

| flag    | default | reasoning |
|---------|---------|-----------|
| `-fmoe` | on      | Fuses the `up` and `gate` projections into one op. On by default upstream; no reason to disable. |
| `-ooae` | on      | When the scheduler copies CPU-resident experts to the GPU for a batch, transfer *only the activated ones*. With 8 of 288 active this is a large saving at small batch sizes. It can lose slightly at very large batches where all 288 get touched anyway — `./bench.sh batch` will show it. |
| `-thp`  | on      | Transparent huge pages for CPU tensors: fewer TLB misses when walking ~26 GiB of scattered expert weights. |
| `-rtr`  | **off** | Repacks CPU experts into a row-interleaved layout the AVX2/VNNI kernels prefer. **Measured (ncmoe=22): prefill +16% (208→242 t/s), generation unchanged.** It helps the compute-bound GEMM of prefill but does nothing for bandwidth-bound generation — reorganising the bytes doesn't reduce how many are read per token. It forces `--no-mmap`, so every start re-reads the whole model (~114 GiB Q4 / ~195 GiB Q8) from disk. Worth it only for prefill-heavy workloads (long prompts, RAG); useless for chat. To keep the gain without the per-start cost, repack the file once offline with `llama-quantize --repack` (see the `step-3.7-flash-q4-r4` profile). |
| `-ser`  | unset   | "Smart expert reduction": `IK_SER=6,1` runs 6 experts instead of 8. Roughly 25% less CPU-side traffic. **It changes model output** — treat it as a quality/speed trade to evaluate deliberately, not a free win. |

---

## 6. Flags deliberately *not* set

- **`-mla`** — DeepSeek-style multi-head latent attention. `step35` uses plain
  GQA; the flag does not apply.
- **`--defer-experts`** — faults expert pages in lazily. Cuts cold-start time,
  but the cost reappears as stalls during the first responses. Off by default
  since a server starts once and runs for hours; turn it on if you restart
  constantly while experimenting.
- **`--mlock`** — with ~30 GiB of CPU experts and 224 GiB of RAM, the page cache
  keeps them resident anyway, and `mlock` needs a raised `ulimit -l`.
- **`-amb`** — caps attention scratch size. Useful when attention buffers blow
  up; with 8 KV heads and a 512-token window they do not.

---

## 7. What to change if the hardware changes

### The 128 GB / DDR5-7000 swap

You mentioned possibly switching to 128 GiB at 7000 MT/s. For this workload
that trade is **bad**, and it is worth being explicit about why:

- **Bandwidth gain is small.** 6333 → 7000 MT/s is +10.5%. It applies only to
  the ~11 layers of experts on the CPU, which are maybe a third of total token
  latency — so expect low single-digit percent end to end.
- **Capacity loss is not small.** Q4_K_XL needs ~25–30 GiB of CPU-side expert
  weights, which 128 GiB still holds. But `Q8_K_XL` needs ~145 GiB and would
  **stop fitting entirely**, and you would lose the page-cache headroom that
  makes restarts fast.

Keep the 224 GiB. If you want more generation speed, the lever is getting more
experts onto the GPU (context length, KV precision, `-ub`), not making the CPU
path 10% quicker.

### A different model

The one number to re-derive is **GiB per expert layer**:

```
(model file size − non-expert weights) ÷ number of MoE layers
```

Everything in §1 follows from it. `--fit` handles this automatically; you only
need the arithmetic when you want to reason about a trade before running it.

### A second GPU

`--n-cpu-moe` changes behaviour with more than one device — it distributes the
CPU-resident layers proportionally across GPUs rather than taking the first N.
`--fit` handles multi-GPU too. Set `IK_EXTRA_ARGS="-sm layer"` and re-run
`./bench.sh ncmoe`.

---

## 8. Recommended order of work

1. `./build.sh`
2. `./serve.sh --dry-run` — check the command looks right
3. `./bench.sh quick` — get a baseline
4. `./bench.sh ncmoe` — confirm or beat `--fit`'s choice
5. `./bench.sh threads` — confirm 6/18
6. `./bench.sh batch` — confirm 4096/1024
7. Write the winners into `config/models/step-3.7-flash-q4.env`
8. `./bench.sh sweep` — verify it holds up as context fills


## 6. CUDA toolkit and GPU architecture: neither matters here

Mainline llama.cpp built with CUDA 13.x loses most of its throughput on
Blackwell once the KV cache passes 8192 tokens - proven, with a container-based
CUDA 12.8 workaround, in
`~/development/multi-gpu-llm/doc/cuda-fa-blackwell.md`. The obvious worry was
that this toolkit inherits the same problem.

It does not. Measured on MXFP4, `--n-cpu-moe 16`, q8_0 KV, 24 threads,
changing exactly one variable at a time (16k context):

| CUDA | arch | pp | tg |
|------|------|---:|---:|
| 13.3 | 120a-real | 382.0 | 19.71 |
| 12.8 | 120a-real | 384.6 | 19.47 |
| 12.8 | 120-real  | 376.2 | 19.40 |

All three sit inside the run-to-run spread. **Build with whatever CUDA is
installed**; `build-cuda12.sh` exists for the comparison and as insurance, not
because it is needed. The arch-specific `120a` suffix is likewise worth ~2% at
most, which is noise at this sample size.

Why ik_llama escapes a bug that costs mainline 5x: the mainline collapse comes
from its flash-attention kernel selector falling into an MMA path that is slow
on sm_120, and this fork's MLA implementation does not use those kernels.

### What actually moved the number

An earlier comparison appeared to show CUDA 12.8 winning by 10% on prefill.
It did not - that run also went from 18 to 24 threads, and threads are the
whole story (see section 2). Two variables, one conclusion, wrong conclusion.
