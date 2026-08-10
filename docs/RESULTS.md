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
re-match, at [§5](#5-re-measurement-2026-08-10),
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

> **Superseded for DeepSeek by [§5](#5-re-measurement-2026-08-10).** The
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

The other model tested, and the architecture ik_llama is *supposed* to be best
at (MLA / FlashMLA). It is not — the measurement reversed the expectation.

**ik_llama, `--fit`, f16 MLA cache, `-mla 3 -fidx`** (Q8_K_XL ~151 GiB, later
deleted; MXFP4 ~146 GiB):

| quant / ctx      | KV cache | VRAM filled | generation |
|------------------|----------|-------------|------------|
| Q8 / 65 536      | 2.75 GiB | 93.7 GiB    | ~17 tok/s  |
| Q8 / 262 144     | 11 GiB   | 91.7 GiB    | ~13 tok/s  |
| MXFP4 / 65 536   | 2.75 GiB | 95.0 GiB    | ~20 tok/s  |

MLA compresses the KV cache hard (2.75 GiB at 65k vs 6 GiB for step35). `--fit`
fills VRAM to ~92-95 GiB — LM Studio only reached ~74 GiB, but that is LM
Studio's coarse *GUI* slider, **not** a llama.cpp limit, so credit `--fit`, not
ik_llama specifically.

**Head-to-head, MXFP4, identical split (`--n-cpu-moe 18`), 65536, same prompts:**

| | ik_llama.cpp | standard llama.cpp (2.26) |
|-------------------------|--------------|---------------------------|
| VRAM used               | 95.0 GiB     | 92.0 GiB                  |
| generation              | ~20.2 tok/s  | ~20.3 tok/s               |
| long prefill (5263 tok) | **CRASHES**  | **~400 tok/s**            |

- **Generation is identical** (~20 tok/s), same as the step35 result — it is
  bandwidth-bound.
- **ik_llama CRASHES on long prefill** every time:
  `GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed` — the DeepSeek
  Sparse Attention masks produce >32 tensors crossing the CPU/GPU split. Short
  prompts (chat) work; long prompts (RAG, docs) abort. Standard llama.cpp
  handles the same prefill fine at ~400 tok/s.
- Two genuine ik_llama conveniences remain: `--fit` auto-offloads MoE experts to
  hit ~95 GiB (standard's `--fit` does **not** — it tries to put everything on
  the GPU and OOMs, so standard needs a manual `--n-cpu-moe`); and MLA is on by
  default. But neither makes ik_llama faster, and the prefill crash makes it
  *less* reliable here.

**Verdict, both models:** for Step-3.7-Flash (GQA MoE) **and** DeepSeek-V4-Flash
(MLA + DSA), standard llama.cpp — bundled prebuilt in LM Studio / Ollama — is as
fast on generation and either faster (step35 prefill) or more robust (DeepSeek
prefill doesn't crash). The expectation that ik_llama would win on DeepSeek was
wrong. This toolkit's value is the tuning knowledge, the automation, and
`--fit`'s MoE convenience — not raw speed or reliability on these two models.

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

## 5. Re-measurement (2026-08-10)

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

### 5.1 Head-to-head, both at their best

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

### 5.2 Threads — prefill scales, generation does not

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

### 5.3 CUDA toolkit and GPU architecture — both noise

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

### 5.4 Revised verdict

| model class | winner |
|---|---|
| Step-3.7-Flash (plain GQA MoE) | **mainline** — 2.4× prefill, generation tied |
| DeepSeek-V4-Flash (MLA + DSA) | **ik_llama** — +26% prefill, +19% tg, q8_0 KV |

Which is, in the end, what ik_llama.cpp claims about itself: it is an
MLA/DeepSeek specialist, not a general speedup. The §3 verdict was right about
the model it tested and wrong to generalise from a stale binary.
