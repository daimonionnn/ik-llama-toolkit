# Measured results

Every measurement taken while building and tuning this toolkit, on the
[reference machine](../README.md#the-reference-machine). This is the evidence
base behind the defaults; the *reasoning* is in [TUNING.md](TUNING.md).

Two models are covered. **DeepSeek-V4-Flash is the primary one** — it is what
this box serves daily and what the current default profile runs; Step-3.7-Flash
was the toolkit's original tuning target and remains the second model.

**Model 1: DeepSeek-V4-Flash** (`deepseek4`: MLA + DeepSeek Sparse Attention) —
the architecture class ik_llama.cpp is built for, and where all the recent
tuning lives: [§6](#6-re-measurement-2026-08-10) (fair re-match vs mainline),
§7 (fit-in-VRAM + MTP, the 87–94 tok/s config), §8–§12 (fit-margin, depth
sweeps, 512k, the kvram placement that is now the default), §13 (ds4 head to
head), §14–§16 (what is left on MXFP4: MTP at both contexts, and prefill).

- Architecture: 43 layers, 256 experts (6 + 1 shared active), MLA latent KV,
  DSA indexer top-k 512 — trained context 1 048 576
- Params: ~284 B total
- Quants measured: **MXFP4** ~145.6 GiB (effectively lossless QAT; spills to
  DDR5 — the placement work in §8–§12 is about this), **antirez IQ2XXS**
  ~81 GiB (fits entirely in VRAM, §7), antirez L37-42-Q4K mix ~91 GiB, Q8_K_XL
  ~195 GiB (retired)

**Model 2: Step-3.7-Flash** (StepFun AI), Unsloth Dynamic GGUF, from
`unsloth/Step-3.7-Flash-GGUF` — §1–§5.

- Architecture: `step35` MoE — 45 layers, 288 experts (8 active per token),
  ~220 B total / ~7.4 B active, trained context 262 144
- Two quants, differing only in bytes-per-weight (the driver of the Q4-vs-Q8
  speed gap in §2): **Q4_K_XL** ~114 GiB and **Q8_K_XL** ~195 GiB

**Dates:** §1–§5 measured 2026-07-24/25, §6–§7 on 2026-08-10, §8–§13 on
2026-08-12, §14–§16 on 2026-08-13; §1–§4 predate a RAM reconfiguration and a
CUDA upgrade — do not compare numbers across those groups. Generation figures
are also only comparable at equal `max_tokens` — see §15.1.
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

Sections 9 and 10 were driven over HTTP against a live server rather than by
`bench.sh`, because `llama-bench` cannot set `-fidx` and would therefore measure
a configuration this toolkit does not ship. Those runs saved both their data and
the script that produced it, as `deepseek-v4-flash-mxfp4-*-20260812.{csv,sh}`.
`results/` is gitignored, so they are local to the machine that ran them.

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

---

## 10. 512k context, and the KV/expert trade that comes with it (2026-08-12)

The model is trained to 1 048 576 (yarn x16 over 65 536), so 512k is in range on
paper. It works, and getting there exposed a placement mistake that costs 22 %.

### 10.1 It runs — and `--fit` needs a bigger margin every time context doubles

`-c 524288` CUDA-OOMs at `--fit-margin 8192`; 12288 loads. The reason is visible
in the log: at 512k the compute buffer alone is **13.2 GiB** and the DSA caches
another **2.1 GiB** (`CSA K 1428 MiB`, `HCA K 42.5 MiB`, `LID K 672 MiB`), and
`--fit` accounts for neither. Measured with `-ctx-ckpt 0 --cache-ram 0`:

| depth   | prefill | generation | VRAM      | peak RSS |
|---------|--------:|-----------:|----------:|---------:|
| 400     | 121.5   | 16.74      | 93 920 MiB| 92.0 GiB |
| 499 909 | 161.8   | 10.08      | 94 208 MiB| 92.4 GiB |

The 500k prefill took 52 minutes. Prefill stays flat only to ~128k and then
bends: 288 (32k) → 262 (128k) → 210 (260k) → **162 (500k)**.

### 10.2 The KV does not have to live in VRAM — the experts should

At 512k the MLA KV is 21.5 GiB of VRAM, and every GiB of it is a GiB of experts
pushed into DDR5. So: move the KV to RAM and give the VRAM to the experts.

The bandwidth argument says this should lose badly — expert weights are read
sparsely (6+1 of 256 per token, ~2.7 % of ~132 GiB), while the KV is touched on
every token. **That argument is wrong for this model**, because DeepSeek Sparse
Attention only attends ~512 positions regardless of depth. Measured at 512k,
130k depth, all with `-ctx-ckpt 0 --cache-ram 0`:

| variant                          | GPU wt | DDR5 wt | KV        | prefill | generation |
|----------------------------------|-------:|--------:|-----------|--------:|-----------:|
| `--fit` baseline                 | 55.4   | 90.2    | 21.5 GPU  | 205.5   | 13.33 |
| `-nkvo` alone                    | 55.4   | 90.2    | 21.5 RAM  | 204.0   | 13.05 |
| `-ictk q8_0`                     | 55.4   | 90.2    | 21.5 GPU  | 197.3   | 12.35 |
| `-ctk/-ctv q8_0`, margin 20480   | 58.6   | 87.0    | 11.4 GPU  | 211.6   | 13.63 |
| `-nkvo --n-cpu-moe 21`           | 77.7   | 67.9    | 21.5 RAM  | 258.2   | 15.48 |
| same, repeated                   | 77.7   | 67.9    | 21.5 RAM  | 257.4   | 15.42 |
| **`-nkvo --n-cpu-moe 19`**       | **84.1**| **61.6**| 21.5 RAM | **277.9**| **16.30** |
| `-nkvo --n-cpu-moe 17`           | 90.5   | 55.2    | —         | CUDA OOM | — |

**+22 % generation and +35 % prefill** over the `--fit` baseline, and the RAM
bill does not move: 21.5 GiB of KV arrives while 22–29 GiB of experts leave.

Every row tracks one number — how much weight is left in DDR5 — which is what
you would expect if generation is DDR5-bandwidth-bound, and it explains the two
failures too:

* **`-nkvo` alone does nothing** (13.05 vs 13.33) because `--fit` does not hand
  the freed VRAM to anyone: the weight split was byte-identical with and without
  it (92 402 MiB CPU / 56 726 MiB GPU), leaving **31 GiB of VRAM idle**. The KV
  moved out for free — 2 % — but nothing moved in. `-nkvo` is only useful
  together with manual `--n-cpu-moe`.
* **q8_0 KV nearly cancels itself.** It halves the KV to 11.4 GiB and `--fit`
  does move 13 GiB of experts onto the GPU — but then the load OOMs on the
  13.2 GiB compute buffer, and the bigger margin needed to survive takes back
  8 GiB of exactly what was gained. Net +2 %.
* **`-ictk q8_0` is a dead end**: it touches only the 672 MiB `LID K` cache,
  saves 294 MiB, and costs 7 % to dequantise.

`--n-cpu-moe 17` fails on the compute buffer, so 19 is the ceiling at `-ub 512`.
A smaller micro-batch would shrink that 13.2 GiB buffer and might allow 17 or
15 — untested, and the obvious next thing to try.

**Noise:** the repeat of `--n-cpu-moe 21` on a fresh server landed within 0.4 %
(15.48 vs 15.42), so same-config repeats are tight. The ~12 % spread noted in
§9.1 comes from comparing across separate sessions with different page-cache
states, not from the measurement itself.

**Not yet tested at 131072**, where the KV is only 5.5 GiB and the prize is
correspondingly smaller — but `--fit` leaving VRAM on the table is a property of
`--fit`, not of 512k, so it is worth checking there too.

### 10.3 The two open questions, answered

**`-ub` is not the lever for the compute buffer.** Halving the micro-batch to
256 at 512k freed only **668 MiB** (VRAM 91 242 vs 91 910) — nowhere near the
~3.1 GiB `--n-cpu-moe 17` needs, and it still OOMs. The control run (same
`--n-cpu-moe 19`, only `-ub` changed) also shows what the micro-batch actually
does:

| 512k, `--n-cpu-moe 19` | prefill | generation |
|------------------------|--------:|-----------:|
| `-ub 512`              | 277.9   | 16.30 |
| `-ub 256`              | 218.9   | 16.30 |

**-21 % prefill, identical generation.** So `-ub` trades prefill only, and the
13.2 GiB compute buffer is driven by something else — `-b 4096` or the DSA
structures over 512k positions. `-b` is the untested candidate.

**At 131072 the trick barely pays.** The whole `-nkvo` + manual-placement win is
a 512k phenomenon, because it is proportional to how much VRAM the KV occupies —
21.5 GiB there, 5.4 GiB here:

| 131072, 120k depth      | GPU wt | DDR5 wt | prefill | generation |
|-------------------------|-------:|--------:|--------:|-----------:|
| `--fit` margin 4096     | 80.9   | 64.7    | 265.7   | 17.43 |
| `-nkvo --n-cpu-moe 18`  | 87.3   | 58.4    | 288.6   | 17.84 |
| `-nkvo --n-cpu-moe 16`  | 93.6   | 52.0    | CUDA OOM | — |

**+8.6 % prefill, +2.4 % generation** — the generation figure is inside the
measurement's resolution. Note also that at 131072 `--fit` is already doing a
decent job: 64.7 GiB left in DDR5, versus the 90.2 GiB it leaves at 512k. Not
worth trading `--fit`'s adaptivity for a hand-tuned `--n-cpu-moe` here.

### 10.4 A side finding about the wrapper's own config

The 131072 baseline above ran **lean** (`-ctx-ckpt 0 --cache-ram 0`) and did
17.43 tok/s at 120k depth. §9.1 measured the same context **stock** at 13.1–14.9
at 128k depth. That is a 17–33 % gap on the config
`serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh` actually ships.

**This is not a free win and the wrapper has deliberately not been changed.**
`--cache-ram 0` disables the prompt cache, and for multi-turn chat or an agent
that is exactly the structure that makes a re-send free — §9.1's repeat pass hit
it and skipped an 8-minute prefill entirely. The trade is faster tokens within a
turn against re-prefilling the whole conversation on the next one.

Untested split: checkpoints and prompt cache were always turned off together, so
which of the two the speed belongs to is unknown. Worth separating before
anything acts on this.

> **Since measured — and the guess above is wrong.** §11.3 separated them: the
> **context checkpoints** are what make a re-send free, not the prompt cache.
> Turning `-ctx-ckpt 0` on its own loses the reuse entirely. They cost ~1.5 %
> prefill and ~3 % generation, which is a bargain against re-prefilling the
> conversation. §11.2 also shows the caches cost only 3.5 % at 32k depth — the
> 26 % in §9.3 is a 256k figure, and that cost scales with context.

---

## 11. Prefill: +69 % on MXFP4, with nothing given up (2026-08-12)

Constraint for this whole section: **MXFP4 stays**. That rules out the IQ2XXS
quant (fits entirely in VRAM, no CPU experts at all — almost certainly the
fastest thing this box can do, but 2-bit) and `-ser` (changes output). Every
lever below is a placement or scheduling change; the arithmetic is untouched.

All runs at `-c 131072` with a 32k prompt.

### 11.1 What moved the needle

| config                                   | prefill | generation |
|------------------------------------------|--------:|-----------:|
| wrapper default (`--fit` 4096, ub 512)   | 293.1   | 20.09 |
| `-b 8192`                                | ~equal  | ~equal |
| `-no-ooae`                               | 296.2   | 20.21 |
| **`-rtr`**                               | **366.8** | 20.22 |
| `-rtr -nkvo --n-cpu-moe 18`              | 396.6   | 20.71 |
| `-rtr -nkvo --n-cpu-moe 17`              | 413.4   | 21.33 |
| `-rtr -nkvo` n18 `-ub 1024 / 2048 / 4096`| 455 / 485 / 497 | ~20.7 |
| **`-rtr -nkvo` n17 `-ub 2048`**          | **507.8 / 500.9 / 501.6** | **21.31 / 21.68 / 21.40** |

Three mechanisms, and they stack because they fix different things:

**`-rtr` (+25 % on its own).** Repacks the CPU-resident experts into the
row-interleaved layout the AVX2/VNNI kernels want — `Repacked 60 tensors`, so
MXFP4 does have an interleaved variant. §1.6 measured +16 % from this on
step-3.7; DeepSeek gains more, which fits, since far more weight sits on the CPU
here. It is a memory layout change over the same quantized values: output is
unaffected. Cost: it forces `--no-mmap`, so every start re-reads the model.

**`-nkvo` + manual `--n-cpu-moe` (+9 % more).** Same trade as §10.2 — the KV
leaves VRAM, the experts move in. `--n-cpu-moe 17` fits here (§10.3 jumped 18 →
16 and never tried it) and beats 18 on both axes.

**A bigger `-ub`, which `-nkvo` unlocks (+20 % more).** This is the part that
was hidden. Under `--fit`, `-ub 1024` does not merely miss — it **aborts** with
`CUDA error: the resource allocation failed` in `cublas_handle`, because the
compute buffer doubles from 3624 MiB to 7248 MiB. With `-nkvo` the attention
scratch follows the KV to host memory and that buffer collapses to **440 MiB**,
so doubling the micro-batch costs 440 MiB instead of 3.6 GiB and evicts no
experts at all. Gains flatten after 2048 (455 → 485 → 497).

Dead ends, recorded so they are not retried: **`-b` does nothing** for prefill
(the compute buffer tracks `-ub`, not `-b`) and `-ub` above `-b` is silently
clamped; **`-no-ooae` is noise** (+1 %); and **`-ictk q8_0` costs 7 %** (§10.2).

### 11.2 It survives with the prompt cache on — so it is shippable

Every sweep above ran with `-ctx-ckpt 0 --cache-ram 0`, baseline included, so
they compare cleanly against each other but not against what the wrapper ships.
Re-measured with the caches in their default state:

| config                        | prefill | generation | re-send |
|-------------------------------|--------:|-----------:|---------|
| wrapper as shipped            | 287.1   | 19.75      | cache hit |
| **winner, caches on**         | **484.1** | **20.98** | **cache hit** |
| winner, caches off            | 501.6   | 21.40      | full re-prefill |

**+69 % prefill and +6 % generation over what the wrapper does today, keeping
the free re-send.** The caches cost only 3.5 % of prefill at this depth — the
26 % measured in §9.3 was at 256k, where their cost scales with context.

Two extras worth naming: with the KV in host memory, **VRAM no longer grows with
context depth** (the `--fit` config crept 93 814 → 94 102 MiB between 512 and
130k tokens), and host RAM is unchanged, since the KV arriving is offset by the
experts leaving.

The price is robustness. Manual placement means no `--fit` adaptivity, and the
winner leaves only **1.6 GiB of VRAM free** — anything else touching the card
will break it. `-rtr` also forces `--no-mmap`: 30 s to start from a warm page
cache here, minutes from cold.

### 11.3 Correction to §10.4: the checkpoints are what make a re-send free

§10.4 guessed that context checkpoints were pure overhead and the prompt cache
was the valuable half. **Measured, that is backwards:**

| winner with…                          | prefill | generation | re-send |
|---------------------------------------|--------:|-----------:|---------|
| checkpoints + cache on                | 484.1   | 20.98      | **cache hit** |
| `-ctx-ckpt 0`, `--cache-ram` default  | 491.3   | 21.60      | **no reuse** |
| both off                              | 501.6   | 21.40      | no reuse |

Turning the checkpoints off is what loses the reuse. They cost ~1.5 % prefill
and ~3 % generation and buy back a full re-prefill — 33 000 tokens, i.e. ~66 s
per follow-up turn at 500 tok/s. Keep them on for anything conversational.

---

## 12. 262144 catches up, and the "4k artefact" is really a 1k–16k band (2026-08-12)

### 12.1 The kvram treatment pays even more at 256k

256k was the one context none of the tuning had reached (TODO item 1). Applied
and measured at 130k depth, same corpus as §10/§11:

| config at 262144                          | caches | GPU wt | DDR5 wt | prefill | generation |
|-------------------------------------------|--------|-------:|--------:|--------:|-----------:|
| `--fit` margin 8192, ub 512               | off    | 73.0   | 76.1    | 241.4   | 15.31 |
| `--fit` margin 4096                       | —      | —      | —       | **does not load** | — |
| `-rtr -nkvo --n-cpu-moe 18 -ub 2048`      | off    | 89.4   | 59.8    | **429.2** | **17.55** |
| same, `--n-cpu-moe 17`                    | —      | —      | —       | CUDA OOM | — |
| same n18, **caches on**                   | on     | 89.4   | 59.8    | **402.5** | 13.38 |

Against the stock §9.2 reference at this depth (232.6 pp / 12.36 tg):
**+73 % prefill, +8 % generation**, caches on. Shipped as
`deepseek-v4-flash-256k-kvram` + `serve-deepseek-v4-flash-mxfp4-kvram-256k.sh`.

Three structural notes:

* **Margin 4096 does not load at 262144** — the §8 wrapper gating at
  `ctx <= 131072` was correct. The compute buffer scales with context as well as
  `-ub`: 3.6 GiB @128k, 7.1 GiB @256k, 13.2 GiB @512k (all ub 512, `--fit`).
* **The gain is bigger than at 128k because `--fit` is worse here**: at margin
  8192 it leaves only 73.0 GiB of weights on the GPU. The manual split moves
  16.3 GiB of experts out of DDR5.
* **The caches cost real money at this context: −24 % generation** (17.55 →
  13.38; at 128k it was ~2 %). Both figures here are 64-token generations and
  read low in absolute terms — see §15.1. Checkpoint overhead scales with KV size (11 GiB
  here). Still shipped ON — a re-send at 130k depth would otherwise re-prefill
  for ~5 minutes — but for single-shot batch work over huge prompts, run the
  profile with `-ctx-ckpt 0 --cache-ram 0`.

### 12.2 The dip: not `-b`, not batch fullness — a depth band

TODO item 4 hypothesised the ~4k first-request dip was `4096 == -b`. Moving `-b`
killed that in one sweep, and the follow-ups killed its successor too:

| depth  | `-b` 2048 | `-b` 4096 (§9.1) | `-b` 8192 | kvram default (ub 2048) |
|-------:|-----------|------------------|-----------|--------------------------|
| ~0.5k  | —         | 21.0 vs 23.3     | —         | — |
| ~1k    | —         | —                | —         | **15.3 vs 23.9 DIP** |
| ~2k    | **14.7 vs 22.4 DIP** | —     | **14.4 vs 21.9 DIP** | — |
| ~4k    | **17.8 vs 21.6 DIP** | **14.2 vs 21.5 DIP** | **14.5 vs 21.4 DIP** | **19.0 vs 23.0 DIP** |
| ~8k    | **13.9 vs 22.3 DIP** | —     | **14.5 vs 21.6 DIP** | — |
| ~16k   | —         | —                | —         | **18.6 vs 22.5 DIP** |
| ~32k   | —         | 19.9 vs 20.0     | —         | — |
| ~130k  | —         | 12.7 vs 13.1     | —         | — |

(each cell: first uncached request vs re-send of the identical prompt)

What survives the data: **an uncached request at roughly 1k–16k depth generates
15–35 % slower than the same request re-sent**, at every `-b` tried, under both
`--fit` and the kvram config — including the shipped default. Clean at ≥32k.
The lower edge is below 1k (the old ~0.5k row gapped ~10 %, previously read as
noise); the upper edge is between 16 327 and 33 079.

Killed hypotheses, so nobody re-walks them: `4096 == -b` (dip indifferent to
`-b`), and "last logical batch nearly full" (b 8192 @ 2k = 24 % full, dips
hard). Mechanism unknown — the re-send restoring the same depth from a context
checkpoint is fast, so it is not the attention cost of that depth; something
about the state a fresh prefill leaves behind differs from a restored one.
Practical impact: the first response of a session whose prompt lands in that
band runs at ~80 % generation speed, once.

---

## 13. ds4 (DwarfStar) vs ik_llama on the same quant (2026-08-12)

TODO item 5: benchmark antirez's [ds4](https://github.com/antirez/ds4) against
ik_llama on the **same** file — the antirez IQ2XXS ~81 GiB quant this box
already runs, written by ds4's own author. It took four local patches to get a
session to start; with those, **ds4 prefills ~1.5–1.8× faster than ik_llama and
generates ~0.7–0.9× as fast.**

### 13.1 Head to head, IQ2XXS, no speculative decoding on either side

ds4 via `ds4-bench` (incremental interval prefill, 128 greedy tokens per
frontier); ik_llama via HTTP against the live server, `-ngl 99` (no `--fit`),
f16 KV, same depths. Neither side used MTP/DSpark.

| depth  | ds4 prefill | ik prefill | ratio | ds4 gen | ik gen | ratio |
|-------:|------------:|-----------:|------:|--------:|-------:|------:|
| 2 048  | **1 853**   | 1 270      | 1.46× | 72.7    | **79.5** | 0.91× |
| 4 096  | **2 065**   | 1 423      | 1.45× | 52.1    | **72.6** | 0.72× |
| 8 192  | **2 207**   | 1 403      | 1.57× | 50.3    | **71.2** | 0.71× |
| 16 384 | **2 176**   | 1 395      | 1.56× | 51.1    | **68.4** | 0.75× |
| 32 768 | **2 144**   | 1 305      | 1.64× | 46.5    | **63.0** | 0.74× |
| 65 536 | **2 076**   | 1 166      | 1.78× | 44.8    | **55.9** | 0.80× |

ik generation is the steady (re-sent) figure, so the §12.2 dip band does not
distort it. Note ds4's prefill advantage *grows* with depth — it is essentially
flat at ~2 100 tok/s from 4k to 65k, while ik decays 1 423 → 1 166.

**Incidentally, this is also the first prefill measurement of the fit-in-VRAM
quant on ik_llama** (§7 only ever recorded generation): 1 270–1 166 tok/s versus
484 tok/s for the best MXFP4 config in §11. Prefill suffers from CPU offload
even more than generation does — 2.6×.

Verdict for this box: neither engine wins outright. ik_llama keeps the
generation crown (and with MTP reaches 87–94 tok/s, which ds4 has no equivalent
of here since DSpark needs its own support GGUF). ds4 owns prefill by a wide,
depth-proof margin. For RAG or long-document work the difference is stark: a
65k-token prompt costs ~31 s on ds4 and ~56 s on ik_llama.

### 13.2 Four bugs stood between "builds" and "runs"

The build itself is clean — `make cuda CUDA_ARCH=sm_120a CUDA_HOME=/usr/local/cuda-13.3`,
zero warnings, `DS4_CUDA_HAVE_MXF4=1`. Getting a *session* took a chain of four
fixes, each hiding the next. The patch is committed in this repo as
[`docs/external/ds4-blackwell-discrete-fixes.patch`](external/ds4-blackwell-discrete-fixes.patch)
and lives on branch `local/blackwell-discrete-fixes` in `~/development/ds4`,
together with `run-cuda-local.sh`, which carries the working invocation. Fixes
1–2 are hardware-portable and would suit upstream as-is; 3–4 want real design
decisions rather than the env-var escape hatches used here.

1. **`cudaHostRegisterReadOnly` is unsupported on this driver.** Verified
   directly: `cudaDevAttrHostRegisterReadOnlySupported = 0` on driver 595.84.
   ds4 registers the model mapping with `Mapped|ReadOnly` and has no fallback,
   so the no-copy path dies with *"operation not supported"*.
2. **The mapping was `r--s` — read-only and SHARED.** On Linux ds4 still takes
   the Metal branch (`MAP_SHARED`, `PROT_READ`), and a read-only mapping cannot
   be pinned at all without the ReadOnly flag; retrying without it returns
   *"invalid argument"*. Standalone probes confirmed the rule: anonymous,
   private-RW file, and even O_DIRECT-backed mappings all pin fine; `PROT_READ`
   + `ReadOnly` is the only combination the driver rejects. Patch: on Linux map
   `MAP_PRIVATE | PROT_READ|PROT_WRITE` (copy-on-write, nothing ever writes) and
   retry registration without the flag. Registering the full 80.76 GiB then
   succeeds in ~20 s.
3. **A successful registration disables the device weight cache.**
   `cuda_model_range_ptr()` returns the host pointer whenever the mapping is
   registered — correct on a DGX Spark, where host memory *is* device memory,
   but on a discrete GPU it means PCIe zero-copy reads on every token:
   **0.54 tok/s**. Patch: honour `DS4_CUDA_WEIGHT_CACHE=1` to fall through to
   the arena cache.
4. **The arena allocator cannot achieve full residency.** Its chunked first-fit
   packing consumed ~1.5× the bytes it stored — at `--n` 256 MiB chunks the card
   was exhausted with only 64 GiB of 80.76 GiB cached — and the cache limit
   governs *span* bytes while arenas overshoot it (limit 81 GiB → 88 GiB of
   arenas). Patch: raise the 8 GiB chunk cap so the whole model lands in **one**
   arena.

**Partial residency is a cliff, not a slope**, which is why this had to be
chased to the end:

| resident | prefill | generation |
|---------:|--------:|-----------:|
| 0 (registered host only) | 60 | 0.54 |
| 60.0 GiB (74 %) | 93 | 1.33 |
| 76.0 GiB (94 %) | 153 | 2.70 |
| 80.0 GiB (99.1 %) | 2 082 | 20.5 |
| **80.76 GiB (100 %)** | **2 161** | **72.6** |

The last 536 MiB span is worth 20.5 → 72.6 tok/s. With MoE routing touching
6 of 256 experts per token, any host-resident weight is hit constantly.

Environment: RTX PRO 6000 Blackwell 96 GiB (sm_120), driver 595.84, CUDA 13.3,
ds4 @ upstream 84cc882 (2026-08-12), checkout at `~/development/ds4`.

**To reproduce**, from that checkout on the patched branch:

```sh
./run-cuda-local.sh bench          # the sweep in §13.1
./run-cuda-local.sh serve          # OpenAI/Anthropic server on :8091
```

The two environment variables it sets are not optional and have no sane default
here — `DS4_CUDA_WEIGHT_CACHE=1` (fix 3) and
`DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=84000` (fix 4). The ik_llama side of §13.1 is
`results/ik-iq2xxs-sweep-20260812.csv`, produced by a plain `-ngl 99` server on
the same file.

**Why it stays interesting** beyond the prefill number: a 1.2 GiB compressed KV
where ik's MLA cache needs 2.75 GiB at 65k, KV persisted to disk across
restarts, and 1M context.

---

## 14. MXFP4: what is left after §11 — MTP is, the rest is not (2026-08-13)

The lossless path had four untried levers. Measured at 131072 / 32k prompt /
caches on, against the shipped `deepseek-v4-flash-128k-kvram` default:

| config                         | prefill | generation |
|--------------------------------|--------:|-----------:|
| default (`--n-cpu-moe 17`)     | **499.2** | 21.26 |
| `-mqkv`                        | 494.0   | 21.30 |
| `-muge`                        | — crashes | — |
| n19, no MTP                    | 450.4   | 20.03 |
| **n19 + MTP**                  | 434.5   | **24.12** |
| n20 + MTP                      | 416.5   | 23.71 |

### 14.1 MTP is worth +20 %, and +13.5 % after paying for it

The 5.5 GiB predictor needs VRAM the n17 placement does not have (1.6 GiB
free), so it costs two placement steps. Isolating the two effects:

* placement n17 → n19 alone: **20.03** tok/s (−1.23 from the default)
* adding MTP at n19: **24.12** tok/s — **+20.4 %** over the same placement

Net against the shipped default: **+13.5 % generation for −13 % prefill.**
`--n-cpu-moe 20` is worse on both axes, so 19 is the optimum. Verified from the
load log that the target split is byte-identical with and without the draft
(86 102.93 / 63 026.00 MiB); the draft adds 4 597 MiB on the GPU and 1 010 MiB
on the host, so the comparison isolates MTP cleanly.

This is not a quality trade — speculative decoding verifies every drafted token
against the target — which is why it is worth having on the lossless path at
all. Shipped as `deepseek-v4-flash-128k-kvram-mtp` +
`serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh`, **not** as the default: §11
bought +69 % prefill and this hands 13 % of it back, so which one wins depends
on whether the workload reads or writes more.

For comparison, §7.2 measured MTP on MXFP4 at +30 % (18.4 → 24.0) with the old
`--n-cpu-moe 24` placement. The *absolute* ceiling barely moved (24.0 → 24.12);
what the kvram work bought was the same generation speed with far better
prefill.

### 14.2 `-mqkv` is a no-op here, `-muge` crashes the server

**`-mqkv`** (merge Q, K, V) lands inside noise: 494.0 / 21.30 against 499.2 /
21.26, with an identical memory split. Expected in hindsight — DeepSeek's MLA
attention projects through q/kv LoRA ranks rather than three parallel
projections, so there is nothing to merge.

**`-muge`** (merge up/gate expert projections) **aborts the server** — but only
when combined with `-rtr`, which every kvram profile carries. It walks all 43
layers and then dies:

```
merge_up_gate_exps: merging up/gate in layer 42
llama.cpp:7854: GGML_ASSERT(other_type == l.ffn_up_exps->type) failed
```

**First diagnosis here was wrong** — it blamed the quant for mixing gate/up
types. The assert one line above (`l.ffn_up_exps->type == l.ffn_gate_exps->type`)
*passes*, so those two do share a type. The real cause is a one-line table bug
in `src/llama-quantize.cpp`, where `interleaved_properties()` maps every
interleaved type back to its base:

```cpp
{ GGML_TYPE_Q4_0_R8,     { GGML_TYPE_Q4_0,  8} },
{ GGML_TYPE_IQ4_XS_R8,   { GGML_TYPE_IQ4_XS, 8} },
{ GGML_TYPE_MXFP4_R8,    { GGML_TYPE_MXFP4_R8, 8} },   // <- maps to itself
```

With `-rtr`, the merged `ffn_up_gate_exps` is repacked to `MXFP4_R8`; the check
maps it back expecting `MXFP4`, gets `MXFP4_R8`, and the comparison against
`ffn_up_exps->type` fails. **Verified both ways: `-muge` without `-rtr` loads
fine, and correcting the entry to `GGML_TYPE_MXFP4` fixes it with `-rtr` on.**
Reported upstream with the patch.

`-amb` was not reached; with the compute buffer already at 440 MiB under
`-nkvo` (§11.1) there is nothing left for it to cap.

---

## 15. MTP at 262144 — the best trade of the three contexts (2026-08-13)

§14.1 measured MTP at 131072. It cannot be extrapolated to 262144: the KV is
11.4 GiB instead of 5.4, the compute buffer 2.8 instead of 1.8, and the shipped
256k profile already sits at `--n-cpu-moe 18` because 17 OOMs there. So the
5.5 GiB predictor is paid for from a tighter budget. Measured at 130k depth,
caches on, 160 generated tokens:

| config                         | prefill | generation |
|--------------------------------|--------:|-----------:|
| 256k kvram profile (n18)       | **406.1** | 16.31 |
| n20, no MTP                    | 376.9   | 15.47 |
| **n20 + MTP**                  | 364.2   | **20.48** |
| n21 + MTP                      | — exited during startup | — |

**MTP alone is worth +32.4 %** (15.47 → 20.48 at identical placement); net
against the shipped profile, **+25.6 % generation for −10.3 % prefill**. That
beats the same treatment at 131072 (+20.4 % isolated, +13.5 % net) — deeper
context leaves generation more bandwidth-starved, so each accepted draft token
saves proportionally more.

Verified from the load log that the target split is byte-identical with and
without the draft (82 838.93 / 66 290.00 MiB), and that the draft accounts for
the entire 7.2 GiB VRAM difference (95 772 vs 88 360) — so the comparison
isolates MTP rather than the placement.

Shipped as `deepseek-v4-flash-256k-kvram-mtp` +
`serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh`.

`--n-cpu-moe 21` exited during startup with no error line and no kernel OOM,
after both the target and the draft KV had initialised. Not chased: a looser
placement is monotonically worse (n19 → n20 at 131072 lost on both axes), so
20 is the answer regardless.

### 15.1 A methodology correction: generation figures depend on `max_tokens`

The n18 baseline above reads 16.31 tok/s where §12.1 recorded **13.38** for what
is nominally the same configuration. The difference is not noise and not the
config — §12.1 generated 64 tokens, this section generates 160.

§12.2 established that the tokens right after a fresh prefill are slow. A
64-token measurement carries that slow start over a quarter of its sample; a
160-token one dilutes it. **So generation numbers are only comparable across
sections that used the same `max_tokens`**, and the shipped 256k profile is
meaningfully faster in ordinary use than §12.1's figure suggests. Prefill is
unaffected (406.1 vs 402.5 — within noise).

Affected: §12.1 (64 tokens) reads low against §11 and §14–§15 (160 tokens).
Within any one section the comparison is sound.

---

## 16. Prefill is exhausted; threads split into two independent laws (2026-08-13)

Prefill is the metric this box is optimised for, so §11's 499 tok/s got a
dedicated sweep at 131072 / 32k prompt / n17 kvram. **Nothing moved it**, and
the sweep found something else instead.

### 16.1 Every remaining prefill lever is spent

| config                  | prefill | generation |
|-------------------------|--------:|-----------:|
| **`-tb 24` (shipped)**  | **491.8** | 21.66 |
| `-tb 20`                | 469.8   | 21.41 |
| `-tb 16`                | 428.8   | 21.42 |
| `-tb 12`                | 380.7   | 21.50 |
| `-tb 8`                 | 332.9   | 21.36 |
| `-no-ooae`              | 490.3   | 20.73 |
| `-ub 3072`              | loads, then **CUDA OOM mid-prefill** | — |

* **Prefill threads are already maxed.** The curve is monotonic and still
  climbing at the top — 20 → 24 is worth 4.5 %, so prefill is not saturated even
  at all 24 cores. There are no more cores to give it.
* **`-no-ooae` is noise** (490.3 vs 491.8), which kills the TUNING note that it
  should lose at large batches: this is `-ub 2048`, four times the batch where it
  was last tested, and it still does not matter.
* **`-ub 3072` is over the line.** It loads at 97 118 of 97 887 MiB — 769 MiB
  free — and then dies mid-prefill with `CUDA error: out of memory` and a core
  dump. `-ub 2048` is a hard ceiling at n17.

So 491–499 tok/s is the ceiling for MXFP4 here. Beating it means fewer experts
in DDR5, and n17 is VRAM-bound. The remaining options are a smaller quant
(IQ2XXS does 1 270 tok/s, but at 2-bit) or a different engine (ds4 does ~2 100
on that same 2-bit file, §13).

### 16.2 Prefill follows cores; generation follows the thread-to-core ratio

Chasing an accidental result — pinning to the P-cores had left prefill alone but
lifted generation 20 % — separated two effects that had been conflated:

| threads | cores | prefill | generation |
|--------:|------:|--------:|-----------:|
| 24      | 24    | 494.3   | 21.21 |
| 16      | 22    | 491.8   | 21.44 |
| **8**   | **24**| **492.8** | **22.78** |
| 8       | 8     | 302.3   | 21.56 |
| 12      | 8     | 303.1   | 23.29 |
| 24      | 8     | 302.9   | **25.15** |

**Prefill tracks the core count alone** — 24 cores gives 492–494 whatever `-t`
is, 8 cores gives 302–303 — because `-t` and `-tb` are independent knobs.

**Generation is about core *homogeneity*, not core count, thread count or core
quality.** Three readings died on the way to this one, so the controls matter:

| cores used              | threads | prefill | generation |
|-------------------------|--------:|--------:|-----------:|
| 24 mixed (8 P + 16 E)   | 24      | **494.3** | 21.21 |
| 24 mixed                | 32      | 490.8   | 21.38 |
| 24 mixed                | 48      | 489.4   | 21.38 |
| **8 P-cores (0–7)**     | 24      | 303.5   | **25.33** |
| 8 P-cores               | 12      | 303.1   | 23.29 |
| 8 P-cores               | 8       | 302.3   | 21.56 |
| **16 E-cores (8–23)**   | 16      | 303.2   | **24.63** |
| 8 E-cores (16–23)       | 24      | 161.1   | 20.25 |
| 4 P-cores (0–3)         | 4       | 189.6   | 23.18 |

* **Not the ratio:** oversubscribing the mixed set does nothing — 32 and 48
  threads on 24 cores both give 21.38, same as 24.
* **Not the count:** 8 cores can beat 24 (25.33 vs 21.21).
* **Not "E-cores are slow":** *sixteen E-cores alone* give 24.63, beating all
  24 mixed cores despite having strictly less compute.

What fits every row is heterogeneity. ggml splits work evenly across threads and
waits for the slowest at each barrier, so on a mixed set the P-cores finish and
idle while the E-cores catch up. A homogeneous set — all-P or all-E — has no
straggler, which is why 16 E-cores beat 8P+16E. Oversubscribing *within* a
homogeneous P-core set then adds a little more (8 → 12 → 24 threads:
21.56 → 23.29 → 25.33) by covering the memory stalls of scattered expert reads.

**One row does not fit and is left standing rather than explained away:** 4
P-cores with 4 threads gives 23.18, beating 8 P-cores with 8 threads (21.56)
despite the same 1:1 ratio, the same homogeneity and half the compute — 7.5 %,
well outside the 2.1 % noise floor. Whatever else is going on, per-barrier
overhead evidently grows with thread count too, so the honest summary is that
generation is hurt by *both* heterogeneity and thread count, and helped by
oversubscribing a homogeneous set. The mechanism is not nailed down.

Prefill, by contrast, is straightforwardly compute-bound and heterogeneity does
not bother it: it tracks how much core there is. A P-core is **1.88× an E-core**
here (303.5 vs 161.1 at eight of each), 16 E-cores land exactly where 8 P-cores
do (303.2 vs 303.5), and all 24 mixed cores give 491.8 — so the 16 E-cores add
only 62 % on top of the 8 P-cores, i.e. together they are worth about five
P-cores. Had this part been 24 homogeneous P-cores, prefill would extrapolate to
~910 tok/s rather than 492.

That is the real prefill ceiling on this box, and no engine flag reaches it: it
is set by the P/E mix of the CPU.

**This corrects `default.env`,** which called generation "flat, bandwidth-bound"
from 12 to 24 threads. That was measured at the old `-ncmoe 16` placement with
~90 GiB of experts on the CPU; kvram leaves 56 GiB there and the behaviour
changed.

The free win: **`-t 8` with `-tb 24` gives ~+5 % generation and leaves prefill
untouched.** Calibrated across three independent sweeps today, the 24/24
baseline reads 21.21 / 21.66 / 21.26 — a 2.1 % spread, which is the cross-sweep
noise floor for generation. `-t 8` reads 22.78 and 22.16, so the honest figure
is ~+5 %, not the +7.4 % a single pair suggested. Pinning to 8 P-cores buys
more generation still (25.15) but costs 39 % of prefill, so it is a trade rather
than a win — wrong way round for this box.
