# Measured results

Every measurement taken while building and tuning this toolkit, on the
[reference machine](../README.md#the-reference-machine). This is the evidence
base behind the defaults; the *reasoning* is in [TUNING.md](TUNING.md).

**Model under test: Step-3.7-Flash** (StepFun AI), Unsloth Dynamic GGUF, from
`unsloth/Step-3.7-Flash-GGUF`.

- Architecture: `step35` MoE — 45 layers, 288 experts (8 active per token)
- Params: ~220 B total / ~7.4 B active
- Trained context: 262 144

Tested in two quantizations, which share that architecture and differ only in
bytes-per-weight (the driver of the Q4-vs-Q8 speed gap in §2):

| quant       | on disk             | role                   |
|-------------|---------------------|------------------------|
| **Q4_K_XL** | ~114 GiB (4 shards) | default / daily driver |
| **Q8_K_XL** | ~195 GiB (6 shards) | quality reference      |

A **second model, DeepSeek-V4-Flash** (`deepseek4`: MLA + sparse attention,
~151 GiB Q8), is also covered — in the runtime comparison at [§3.1](#31-deepseek-v4-flash-deepseek4-mla--sparse-attention) and, after a fair
re-match, at [§6](#6-re-measurement-2026-08-10),
because it is the architecture class where ik_llama is expected to differ most.

**Dates:** 2026-07-24 / 25.
**Two measurement methods, don't mix them:**

- **`llama-bench`** — allocates almost no KV cache, so it shows the *weight-only*
  ceiling. Great for A/B, but a split that passes here can still OOM the server.
- **Real server** (`serve.sh`, `-c` set) — allocates the full KV cache, so its
  `-ncmoe` ceilings are the ones that actually apply in production. Generation
  numbers are measured on short prompts (cache near-empty) unless noted.

All runs: `-fa on`, `q8_0` KV unless noted, GPU otherwise idle.

---

## 1. Q4_K_XL (default model, ~114 GiB)

### 1.1 First working baseline — `--fit`, 65 536 ctx, 6/18 threads (llama-bench)

| test   | t/s  |
|--------|------|
| pp512  | ~209 |
| pp2048 | ~267 |
| pp8192 | ~265 |
| tg128  | ~27  |

This is the untuned starting point. `--fit` chose the split; everything below is
about beating it.

### 1.2 Expert-split sweep — `--n-cpu-moe`, small ctx, 18/6 threads (llama-bench)

Layers 0-2 are dense, so `-ncmoe N` puts `N-3` expert layers on the CPU. Lower N
= more experts on GPU = faster, until the GPU OOMs.

| `-ncmoe` | CPU expert layers | pp2048 | tg128    |
|----------|-------------------|--------|----------|
| `--fit`  | ~17               | 222    | 26.9     |
| 24       | 21                | 183    | 23.9     |
| 20       | 17                | 225    | 27.8     |
| 17       | 14                | 260    | 32.2     |
| 14       | 11                | 308    | 37.8     |
| 13       | 10                | 359    | 39.9     |
| 12       | 9                 | 389    | 42.3     |
| **11**   | 8                 | 384    | **46.0** |
| 10       | 7                 | — OOM  | — OOM    |

Takeaway: hand-picked `-ncmoe` beats `--fit` by a lot (46 vs 27 tok/s at small
ctx), because `--fit` leaves ~6 expert layers on the CPU unnecessarily. The
weight-only ceiling is `-ncmoe 11`.

### 1.3 Context × split at the **real server** (full KV cache)

This is what actually applies in production. The KV cache is large (see §4) and
scales with context, so higher context forces more experts onto the CPU.

| context | KV type | `-ncmoe` | result            | tg (tok/s) | VRAM used |
|---------|---------|----------|-------------------|------------|-----------|
| 262 144 | q8_0    | 20       | OOM               | —          | —         |
| 262 144 | q8_0    | 21       | loads (tight)     | 26.5       | 95.6 GiB  |
| 262 144 | q8_0    | **22**   | loads (default)   | **25.3**   | 93.0 GiB  |
| 262 144 | q4_0    | 16       | loads (VRAM full) | 33.2       | 97.1 GiB  |
| 262 144 | q4_0    | **18**   | loads (comfy)     | **30.1**   | 91.9 GiB  |
| 131 072 | q8_0    | 14, 15   | OOM               | —          | —         |
| 131 072 | q8_0    | 16       | loads (tight)     | 32.6       | 95.9 GiB  |
| 131 072 | q8_0    | **17**   | loads (safe)      | 28.2       | 93.3 GiB  |
| 65 536  | q8_0    | 14       | loads             | ~38 *      | —         |
| 65 536  | q8_0    | 11       | OOM at server     | —          | —         |

\* 65 536 tg from the §1.2 bench at the matching split. Note `-ncmoe 11` loads in
llama-bench but **OOMs the real server** at 65 536 — the difference is the 6 GiB
KV cache llama-bench doesn't allocate. This is why §1.3 exists.

### 1.4 Thread count — generation, `-ncmoe 22` (llama-bench)

| generation threads   | tg128    |
|----------------------|----------|
| 4                    | 24.5     |
| 6                    | 25.9     |
| 8                    | 27.2     |
| 12                   | 27.3     |
| 18                   | **27.5** |
| 6, pinned to P-cores | 24.3     |

Generation keeps improving to 18 threads (all cores) and P-core pinning is
*slower* — at this heavy split the E-cores do real work. (At small `-ncmoe` the
P-core-count rule returns.) Default changed 6 → 18.

### 1.5 Thread count — prefill, `-ncmoe 22` (llama-bench)

| prefill threads | pp2048 |
|-----------------|--------|
| 4               | 187    |
| 8               | 169 *  |
| 12              | 239    |
| 18              | 248    |

Prefill also scales to ~18 but plateaus (12→18 only +4%). \* the 8-thread dip is
the hybrid scheduler landing 2 slow E-cores as stragglers; not a typo.

### 1.6 Run-time repack `-rtr` — `-ncmoe 22` (llama-bench)

|        | rtr **off** | rtr **on** | Δ          |
|--------|-------------|------------|------------|
| pp2048 | 208         | 242        | **+16%**   |
| tg128  | 27.5        | 26.9       | ~unchanged |

`-rtr` helps prefill (compute-bound GEMM), not generation (bandwidth-bound). It
forces `--no-mmap` → re-reads the model each start.

### 1.7 Offline repack to `_R4` — `-ncmoe 22`, mmap on (llama-bench)

The `-rtr` gain baked into the file (`step-3.7-flash-q4-r4` profile), so mmap
stays on and startup is normal.

| variant                       | pp2048  | tg128    | notes                        |
|-------------------------------|---------|----------|------------------------------|
| plain Q4                      | 208     | 27.5     | baseline                     |
| **R4, layers 3-21 only**      | **246** | **26.5** | +18% prefill, works ✓        |
| R4, **all** experts (mistake) | 74      | **0.35** | `_R4` experts on GPU → fatal |

Lesson: only the **CPU-resident** expert layers may be `_R4`; an `_R4` expert on
the GPU collapses to 0.35 tok/s. The shipped R4 file repacks only layers 3-21
(the CPU side at `-ncmoe 22`) and is locked to that split.

### 1.8 Production default, end-to-end — real server, 262 144 / `-ncmoe 22` / 18 threads

| metric                       | value                          |
|------------------------------|--------------------------------|
| generation (short prompt)    | ~26 tok/s (26.3 / 26.6 / 26.3) |
| prefill (7 627-token prompt) | ~242 tok/s                     |
| VRAM used                    | 93 GiB (of 96)                 |
| coherent output              | yes (verified in Slovak)       |

---

## 2. Q8_K_XL (quality reference, ~195 GiB)

Real server, 262 144 ctx, `--fit`, q8_0 KV:

| metric                          | value                   |
|---------------------------------|-------------------------|
| generation                      | ~13 tok/s (12.9 / 13.1) |
| prefill (16 227-token prompt)   | ~141 tok/s              |
| VRAM used                       | 93 GiB                  |
| CPU expert layers (via `--fit`) | ~29 (vs ~22 for Q4)     |
| coherent output                 | yes (verified)          |

**Exactly half of Q4's generation speed** — Q8 forces ~2× the bytes over DDR5
per token (~29 CPU layers instead of ~22). It nearly fills the 224 GiB of RAM.
Use it as a quality reference, not a daily driver.

### Q4 vs Q8 side by side (262 144 ctx)

|            | Q4_K_XL    | Q8_K_XL    |
|------------|------------|------------|
| file size  | 114 GiB    | 195 GiB    |
| generation | ~26 tok/s  | ~13 tok/s  |
| prefill    | ~250 tok/s | ~141 tok/s |
| CPU layers | ~22        | ~29        |

---

## 3. ik_llama.cpp vs standard llama.cpp — was any of this worth it?

> **Superseded for DeepSeek by [§6](#6-re-measurement-2026-08-10).** The
> verdict below stands for Step-3.7-Flash, but the DeepSeek half of it
> ([§3.1](#31-deepseek-v4-flash-deepseek4-mla--sparse-attention)) was measured
> against an ik binary two weeks older than its own checkout, with f16 KV and
> default threads. Re-measured fairly, ik wins that model by ~20-26% and no
> longer crashes on long prefill.

The honest head-to-head. **Standard llama.cpp** = the CUDA build LM Studio and
Ollama bundle (LM Studio's `nvidia-cuda12-avx2-2.26.0`). Both runtimes ran the
**same Q4_K_XL model**, **identical flags** (`-ngl 99 -c 262144 -fa on`, q8_0 KV,
`-b 4096 -ub 1024`, `--n-cpu-moe 22`, 18 threads), and the **same prompts**,
back to back on the freed GPU.

| metric                     | ik_llama.cpp | standard llama.cpp (2.26.0) |
|----------------------------|--------------|-----------------------------|
| generation (400-tok gen)   | ~27 tok/s    | ~27 tok/s                   |
| prefill (7 021-tok prompt) | ~270 tok/s   | **~635 tok/s**              |

- **Generation is identical.** It is bandwidth-bound (reading CPU-resident
  experts from DDR5); both runtimes read the same bytes at the same speed, so
  ik_llama's MoE tricks have nothing to add. Identical generation ⇒ same number
  of experts on the CPU ⇒ a fair, matched comparison.
- **Standard llama.cpp prefills ~2.4× faster** (635 vs 270). Its recent CUDA
  MoE kernels handle the batched, expert-offloaded prefill much better. No
  ik_llama flag closed the gap: `-no-ooae` → 275, `-no-fmoe` → 262, default →
  267. Even the `_R4` repack (§1.7, ~+18% → ~315) stays well below 635.
- **Standard llama.cpp already supports `step35`** (verified in LM Studio 2.24-
  2.26 and Ollama), so it runs this model out of the box.

**Verdict for this model:** the ik_llama build effort did **not** buy a speed
win. For Step-3.7-Flash, LM Studio / Ollama (bundling standard llama.cpp) match
generation and beat prefill, with zero of the CUDA-13 / glibc build pain — their
binaries are prebuilt and just work on Blackwell.

**What the effort still produced, independent of runtime:**
1. The measured tradeoffs in this file (`-ncmoe` vs context vs KV, the DDR5
   bottleneck) — they apply to LM Studio/Ollama too; you can set `--n-cpu-moe`
   there and get the same generation speed.
2. A one-command, reproducible tuning harness and the Blackwell + CUDA-13 build
   recipe (relevant only if you build ik_llama *or* mainline llama.cpp from
   source on this exact box).
3. The `_R4` offline-repack path — ik_llama-only, but even it doesn't catch
   standard's prefill here.

**Where ik_llama.cpp *might* pay off:** DeepSeek-style architectures (MLA /
FlashMLA), pure-CPU inference, or its SOTA `IQ*_K` quant types. Step-3.7-Flash is
a plain GQA MoE, so none of those apply here — hence the tie/loss.

*Caveat:* one model, one config. This is the verdict for **this** setup, not a
universal ranking of the two projects.

### 3.1 DeepSeek-V4-Flash (deepseek4: MLA + sparse attention)

> **The runtime verdict here is superseded by [§6](#6-re-measurement-2026-08-10).**
> This section was measured against an ik_llama binary two weeks older than its
> own checkout (missing the DS4 fixes of 2026-08-07/08), with f16 KV and default
> threads while mainline ran its own tuned config. Re-measured fairly on the
> current build, **ik_llama wins this model** (+26% prefill, +19% generation)
> and the long-prefill crash is gone. Only the VRAM-fill / MLA facts below still
> hold — kept because they are runtime-independent.

MLA compresses the KV cache hard (2.75 GiB at 65k vs 6 GiB for step35), and
`--fit` fills VRAM to ~92-95 GiB across quants:

| quant / ctx    | KV cache | VRAM filled | generation |
|----------------|----------|-------------|------------|
| Q8 / 65 536    | 2.75 GiB | 93.7 GiB    | ~17 tok/s  |
| Q8 / 262 144   | 11 GiB   | 91.7 GiB    | ~13 tok/s  |
| MXFP4 / 65 536 | 2.75 GiB | 95.0 GiB    | ~20 tok/s  |

`--fit` filling VRAM to ~95 GiB is a genuine ik_llama convenience: LM Studio only
reached ~74 GiB (its coarse *GUI* slider, not a llama.cpp limit), and standard
`llama-server` needs a manual `--n-cpu-moe` because its own `--fit` tries to put
everything on the GPU and OOMs. The **speed / robustness verdict is in §6.**

---

## 4. Hardware sensitivity

All measured on this box; the conclusion throughout is that **memory bandwidth
is the wall** and compute upgrades barely move generation.

### 4.1 GPU power limit 300 W → 400 W

|             | 300 W | 400 W                  |
|-------------|-------|------------------------|
| tg (3 runs) | 25.9  | ~26.4 (26.3/26.6/26.3) |

Within noise. GPU draws only ~160 W / 23% util during generation, ~162 W / 58%
during prefill — never near even 300 W. Raising the cap has nothing to use.

### 4.2 PCIe traffic (PCIe 5.0 x16 = ~63 GB/s available)

| phase           | avg read | peak read |
|-----------------|----------|-----------|
| generation (tg) | 89 MB/s  | 134 MB/s  |
| prefill (pp)    | 670 MB/s | 7.4 GB/s  |

Generation barely touches PCIe (experts computed on CPU); prefill bursts to
7.4 GB/s (experts copied to GPU for batched matmul) but still uses ~12% of x16.
→ PCIe affects prefill only, and even x8 would barely dent it. Details in
[FAQ.md](FAQ.md).

### 4.3 CPU upgrade (18 → 24 cores, projected)

Not bought, but bounded by the thread-scaling data (§1.4-1.5): generation
plateaus at ~8 cores and prefill at ~12-18 on the *current* CPU, and a new CPU
keeps the same dual-channel DDR5. Estimate: **tg +0-5%, prefill +8-20%**.
Reasoning in [FAQ.md](FAQ.md) and [TUNING.md §7](TUNING.md).

---

## 5. KV cache size (measured at load, q8_0, all 45 layers)

The sliding-window layers are **not** given a reduced cache in this build — every
layer gets a full-length cache, so it is bigger than the architecture suggests:

| context | KV cache | ≈ expert layers of VRAM    |
|---------|----------|----------------------------|
| 32 768  | 3.0 GiB  | ~1.2                       |
| 65 536  | 6.0 GiB  | ~2.3 (measured: 6120 MiB)  |
| 131 072 | 12.0 GiB | ~4.6 (measured: 12240 MiB) |
| 262 144 | 24.0 GiB | ~9.2                       |

`q4_0` KV halves these. This is why context trades so directly against
generation speed (§1.3).

---

## Raw reports

Auto-saved `bench.sh` runs (tables + full hardware/git context) live in
[`../results/`](../results/) as `*-<mode>-<timestamp>.md` and `.raw.log`. The
numbers above that came from ad-hoc `llama-bench`/server runs during tuning are
consolidated here because that is their only permanent home.

---

## 6. Re-measurement (2026-08-10)

[§3.1](#31-deepseek-v4-flash-deepseek4-mla--sparse-attention) concluded that
ik_llama loses to mainline on DeepSeek-V4-Flash, the model class it is built
for. That conclusion was an artefact of an unfair comparison, and it reverses
when the comparison is fixed.

**What was wrong with it.** The ik binary on disk was built two weeks before
the checkout it was measured from (binary `31018dc`, tree `bd342d6`), missing
a burst of DS4-specific optimisations landed 2026-08-07/08 — one of them
titled *"Allow Q8_0 cache in the CUDA DSA implementation"*. It also ran f16 KV
and default threads while mainline ran its own tuned config.

Everything in this section was measured back to back on the same day, so the
numbers are internally consistent. (§1-§4 predate a RAM reconfiguration and a
CUDA upgrade; do not compare across sections.)

**Setup:** DeepSeek-V4-Flash-0731 MXFP4 (~146 GiB), `-ncmoe 16`, `-mla 3
-fidx`, q8_0 KV, 24 threads, GPU otherwise idle. Measured over HTTP with
`multi-gpu-llm/linux/scripts/benchmark-loaded-model.sh` so both engines are
driven identically.

### 6.1 Head-to-head, both at their best

| | mainline (CUDA 12.8, f16 KV) | ik_llama (q8_0 KV, 24t) |
|---|---:|---:|
| prefill @4k / 16k / 65k | ~305 | **366 / 385 / 362** |
| generation @4k / 16k / 65k | 16.4 | **19.8 / 19.5 / 17.7** |

**+26% prefill, +19% generation.** Part of the margin is capability rather
than speed: mainline **segfaults with `-ctk q8_0` on this model past ~4k
context** (reproduced twice — 4k alone gives 335 pp / 18.9 tg, 16k kills the
server), so it cannot run the configuration ik runs by default.

The `GGML_SCHED_MAX_SPLIT_INPUTS` abort from §3.1 is also gone: 65 536-token
prefills complete normally on the current build.

### 6.2 Threads — prefill scales, generation does not

| threads | pp512 | tg128 |
|--------:|------:|------:|
| 12 | 268.6 | 24.63 |
| 16 | 316.7 | 25.03 |
| 18 | 333.4 | 24.97 |
| 20 | 350.8 | 25.07 |
| 22 | 343.0 | 25.24 |
| **24** | **354.6** | 24.92 |

**+32% prefill from 12 to 24 threads; generation flat** (24.6-25.2, inside the
±0.35 run-to-run spread) because it is bound by DDR5 bandwidth, not cores.
This box has 24 cores, not the 18 assumed throughout §1 — an Arrow Lake-S
Ultra 7 270K, 8 P + 16 E. The old §1.4 table stopped at 18 and measured only
`tg`, the one metric threads do not move, so it read as "18 is enough".

### 6.3 CUDA toolkit and GPU architecture — both noise

One variable changed at a time, 16k context:

| CUDA | arch | threads | pp | tg |
|---|---|---:|---:|---:|
| 13.3 | `120-real` | 18 | 349 | 19.8 |
| 13.3 | `120a-real` | 24 | 382.0 | 19.71 |
| 12.8 | `120a-real` | 24 | 384.6 | 19.47 |
| 12.8 | `120-real` | 24 | 376.2 | 19.40 |

Rows 2-4 sit inside the run-to-run spread. **The CUDA version does not matter
and neither does the `a` architecture suffix** — the 349 → 385 improvement was
the thread count, nothing else. An earlier reading of this data credited CUDA
12.8 with ~10% of prefill; that comparison had changed threads at the same
time and was wrong.

This also means the **Blackwell CUDA 13.x collapse is mainline-specific**.
Mainline llama.cpp loses ~5× past 8192 context under CUDA 13.3 on `sm_120`
(documented in `~/development/multi-gpu-llm/doc/cuda-fa-blackwell.md`, with a
container-based CUDA 12.8 workaround); ik_llama's MLA path does not route
through the flash-attention kernels that misbehave, so it never sees it.
`build-cuda12.sh` remains available as insurance and as the record of this
check.

### 6.4 Revised verdict

| model class | winner |
|---|---|
| Step-3.7-Flash (plain GQA MoE) | **mainline** — 2.4× prefill, generation tied |
| DeepSeek-V4-Flash (MLA + DSA) | **ik_llama** — +26% prefill, +19% tg, q8_0 KV |

Which is, in the end, what ik_llama.cpp claims about itself: it is an
MLA/DeepSeek specialist, not a general speedup. The §3 verdict was right about
the model it tested and wrong to generalise from a stale binary.

---

## 7. DeepSeek-V4-Flash — the two real levers: fit-in-VRAM and MTP (2026-08-10)

§3.1 measured the two DeepSeek quants that *spill* to system RAM (MXFP4 145 GiB,
Q8 195 GiB), where generation is pinned by DDR5 bandwidth to ~20 tok/s. That is
the wrong quant class for this GPU. The 96 GiB of VRAM is 18× the bandwidth of
the DDR5 spill path, so the single biggest lever is **picking a quant that fits
entirely in VRAM** — and the second is **MTP** on top of it.

### 7.1 Fit-in-VRAM quants beat the spilling ones ~3.4×

Measured on the current build, `-mla 3 -fidx -fa on -ctk/-ctv q8_0`:

| quant                              | size    | VRAM used | gen @128k | gen @32k |
|------------------------------------|---------|-----------|-----------|----------|
| MXFP4 (spills to RAM)              | 145 GiB | ~91 GiB   | 20.4 t/s  | 22.4 t/s |
| antirez **IQ2XXS** (fits)          |  81 GiB | 84–90 GiB | **69 t/s**| **75 t/s** |
| antirez **L37-42-Q4K mix** (fits)  |  91 GiB | 93 GiB    | **52 t/s**| —        |

Fitting the whole model in VRAM is worth **~3.4×** over the bandwidth-bound
MXFP4. Both fit-in-VRAM quants stay coherent — 17×23=391 with working, a correct
one-line Rayleigh-scattering answer, valid Fibonacci code. The "higher quality"
91 GiB mix is **25 % slower** and did **not** visibly win coherence on these
prompts (it once drifted into Polish mid-answer), so **IQ2XXS 81 GiB is the sweet
spot**: fastest, most VRAM headroom, stable language. For comparison, 2× DGX
Spark on this model is 37–40 tok/s — this single GPU beats it on a fitting quant
**without** MTP.

### 7.2 MTP adds another ~25–30 % — with a `--fit` caveat

MTP (Multi-Token Prediction / NextN speculative decoding) is **not baked into any
quant** — every DeepSeek-V4 GGUF strips the predictor tail (`--spec-type mtp`
alone fails: *"target GGUF contains no MTP tail, provide a matching predictor-only
companion with -md"*). The predictor is a separate 5.5 GiB companion:
**`philpax/DeepSeek-V4-Flash-MTP-bf16.gguf`**, loaded with
`-md <file> -ngld 99 --spec-type mtp:n_max=1,p_min=0.0`.

Measured at 32 768 ctx, same placement OFF vs ON, drafts accepted both times:

| model (placement)          | MTP OFF | MTP ON  | speedup |
|----------------------------|---------|---------|---------|
| antirez IQ2XXS (`-ngl 99`) | 75 t/s  | **94 t/s** | **+25 %** |
| MXFP4 (`--n-cpu-moe 24`)   | 18.4 t/s| **24.0 t/s** | **+30 %** |

**Caveat that cost several failed loads:** `--fit` cannot co-account for the
draft model — with the target filling VRAM, the draft's own auto-fit fails with
*"Unable to auto-fit model"* no matter how large `--fit-margin` is. You must place
the target **manually** and leave the draft ~6 GiB: `-ngl 99` (no `--fit`) for a
quant that already fits, or an explicit `--n-cpu-moe N` for a spilling one.

### 7.3 Bottom line for this box

The fastest coherent DeepSeek-V4 setup measured here is **antirez IQ2XXS 81 GiB +
MTP = 94 tok/s** at 32k — ~2.4× a dual DGX Spark and ~4.6× the MXFP4-spill
baseline. Order of impact: **fit-in-VRAM (3.4×) ≫ MTP (1.25–1.30×) ≫ everything
else.**

---

## 8. `--fit-margin` at 128k — the margin was costing 7 % (2026-08-12)

The `deepseek-v4-flash` profile carries `IK_FIT_MARGIN=8192`, sized back when the
profile defaulted to 262 144 ctx. At 131 072 the MLA KV is ~5.5 GiB smaller, so
that margin just leaves VRAM empty — and every MiB of empty VRAM is a MiB of
routed experts pushed out to DDR5, which is exactly what caps generation here.

MXFP4 145.6 GiB, `-c 131072`, f16 KV, `--fit`, one 256-token generation each:

| `--fit-margin` | VRAM used | free    | generation |
|----------------|-----------|---------|------------|
| 8192 (old)     | 90 556 MiB| 7.3 GiB | 19.66 t/s  |
| **4096**       | 93 814 MiB| 4.0 GiB | **21.09 t/s** (+7 %) |
| 2048           | 97 074 MiB| 813 MiB | 21.53 t/s (+9 %)     |

2048 is rejected despite being fastest: 813 MiB free is not enough to absorb
allocator fragmentation or anything else touching the card.

**Verified at depth**, margin 4096: a **94 015-token prefill** ran at 282 t/s and
grew VRAM by only **278 MiB** (93 814 → 94 092), no OOM, then generated at
16.5 t/s from that depth. So the compute buffer at full batch is already
accounted for by the fit decision, and 4 GiB of headroom is comfortable.

Applied in `serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh`, gated on context —
at 262 144 the profile's 8192 still applies, since the DSA caches and compute
buffer are allocated *after* the fit decision and 4096 does not cover them there.

---

## 9. MXFP4 GPU+CPU — prefill and generation vs context depth (2026-08-12)

The depth numbers for the spilling MXFP4 quant were scattered across §3.1, §6.1
and §7.1 in three mutually incomparable configurations (q8_0 vs f16 KV,
`-ncmoe 16` vs `--fit`), and nothing existed at 256k. This is one consistent
sweep: f16 KV, `-mla 3 -fidx`, `--fit`, driven over HTTP against the real
`llama-server` so the measured configuration is exactly what the toolkit serves
(`llama-bench` cannot set `-fidx`). Raw data in
`results/deepseek-v4-flash-mxfp4-{depth,repeat,256k}-20260812.csv`.

Every prompt carries a unique prefix, so no deep measurement can silently reuse
the KV of a shallower one.

### 9.1 The wrapper's config — `-c 131072`, `--fit-margin 4096`

| depth (actual)  | prefill  | generation, 1st request | generation, steady |
|-----------------|---------:|------------------------:|-------------------:|
| 463             | 186–202  | 21.0 / 23.3             | **23.3** |
| 4 061           | 264–288  | 14.2 / 15.0             | **21.5** |
| 33 079          | 283–287  | 19.9 / 20.0             | **20.0** |
| 127 904         | 262–263  | 12.7 / 14.9             | **13.1** |

Two columns because they answer different questions. "1st request" is what you
feel when you paste a long prompt into an empty session (both runs shown);
"steady" is the median of three generations at that depth, the later two hitting
the prompt cache so only generation is timed.

**Prefill is remarkably flat** — 262–288 tok/s from 4k all the way to 128k. MLA
plus the DSA indexer mean depth costs almost nothing on the prefill side.
Generation is the part that decays, and it is DDR5-bound throughout.

**The 4k row is a real artefact, not noise, and it is not about depth.** The
first, uncached request at ~4k came in low three times out of three (14.2 in the
128k config, 10.5 in the 256k config, 15.0 in the repeat pass) and recovered to
21.5 on re-send. At 32k — a *larger* prefill — all three samples land within 2 %
of each other, so "generation right after a big prefill is slow" does not
explain it. What is suspicious is that 4096 is exactly `-b`: the prompt fits in
a single full batch, while every deeper prompt ends on a partial one. Untested
hypothesis, recorded so the next person does not re-derive it.

**Noise floor:** the same 128k point measured on two fresh servers gave 14.9 and
13.1 tok/s — ~12 % apart with identical settings. The model is 145.6 GiB and
does not fit RAM+VRAM, so how much of it happens to be in page cache varies
between runs. Do not read these tables to the decimal.

### 9.2 Configuring for 256k costs ~10–17 % everywhere

| depth (actual) | `-c 131072` (margin 4096) | `-c 262144` (margin 8192) |
|----------------|--------------------------:|--------------------------:|
| 463            | 186 pp / 21.0 tg          | 155 pp / 19.8 tg |
| 4 061          | 264 pp / 14.2 tg          | 250 pp / 10.5 tg |
| 33 079         | 283 pp / 19.9 tg          | 250 pp / 18.0 tg |
| 129 931        | 263 pp / 14.9 tg          | 233 pp / 12.4 tg |
| 259 837        | — (over window)           | 211 pp / 10.8 tg |

VRAM used is the same either way (93.8 vs 93.9 GiB) — the 262144 KV is ~8 GiB
larger, so that much *more* of the expert weights sits in DDR5 instead. You pay
for the big window at every depth, including shallow ones.

### 9.3 Context checkpoints and the prompt cache cost 54 GiB and 26 % at 256k

A 250k-token prefill at `-c 262144` was **OOM-killed by the kernel** on the first
attempt: `anon-rss 203 GiB` on a 215 GiB box (the server had served the 512 / 4k
/ 32k / 130k depths first, so the caches had accumulated across requests). Re-run
alone in a `systemd-run --scope -p MemoryMax=170G` cage:

| config at 259 837 tokens          | peak RSS | prefill | generation |
|-----------------------------------|---------:|--------:|-----------:|
| stock                             | 130.2 GiB| 210.5   | 10.82 |
| `-ctx-ckpt 0 --cache-ram 0`       | **76.0 GiB** | 217.4 | **13.68** |

Turning off the 32 per-slot context checkpoints and the 8 GiB prompt cache is a
**double win at this context: 54 GiB less RAM and 26 % more generation**. They
are worth keeping at ordinary depths (that is what makes the re-send in §9.1
free), but at 256k they are pure overhead on this box.

**Operational consequence:** a long-lived server at `-c 262144` accumulates that
state across requests — one prefill peaked at 130 GiB, the accumulated case died
at 203 GiB. If you run the 262144 config as a service, pass `-ctx-ckpt 0
--cache-ram 0` or cap it in a cgroup.
