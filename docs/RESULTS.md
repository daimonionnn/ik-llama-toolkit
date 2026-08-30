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
**Software stack.** Everything below was measured on:

| | |
|---|---|
| OS | Ubuntu 26.04 LTS (`resolute`), kernel 7.0.0-29-generic, glibc 2.43 |
| NVIDIA driver | **595.84** (CUDA runtime 13.2) |
| CUDA toolkit | 13.3 (`nvcc` V13.3.73) |
| Host compiler | GCC 15.2.0 |
| ik_llama.cpp | `7ebbb906` (2026-08-10) for §8–§16; re-checked on `2cda8d2d` (2026-08-13) — 494.1 pp / 21.07 tg against the 499 / 21.3 baseline, unchanged |
| GPU | RTX PRO 6000 Blackwell Workstation, 96 GiB, `sm_120` |
| CPU | Intel Core Ultra 7 270K Plus — 8 P-cores (0–7) + 16 E-cores (8–23) |
| RAM | 256 GB DDR5-6667 (4×64), dual channel — **current**; 244 GiB visible |

The memory has changed four times during these measurements, and each section
states the kit it ran on: 224 GiB at 6267 (§20 and everything before it), 2×48
at 7400 (§25), 4×64 at 6400 (§26), then the same kit at 6667. Do not read the
row above onto an older section.

The driver version matters for more than provenance: §13 turns on
`cudaDevAttrHostRegisterReadOnlySupported` reading **0** here, and §16.2 on the
P/E core split. Re-measure rather than extrapolate if any of these change —
`./check-driver-change.sh --bench` does exactly that.

> **Driver 610.43.02 verified, 2026-08-13 — nothing moved.** Upgraded from
> 595.84 (CUDA UMD 13.2 → 13.3, matching the 13.3 toolkit) specifically to test
> whether a newer driver fixes the ds4 failure. It does not:
> `HostRegisterReadOnlySupported` still reads **0**, so that attribute is a
> property of the platform — a discrete PCIe card without ATS — rather than of
> the driver version, and §13's four causes stand unchanged. Throughput was
> unaffected: 485.5 pp / 21.04 tg against the 499 / 21.3 baseline, i.e. 2.7 %
> and 1.2 %, both inside the 2.1 % same-config noise floor (§16.2). **Every
> table below therefore still holds on 610.43.02.** No Xid or GSP errors in
> either direction, which is worth noting given the open-module hang reports for
> this card under sustained inference
> ([#1111](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1111),
> [#1259](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1259)).
>
> Note also that on Blackwell the **open** kernel modules are not a preference
> but the only option — the proprietary modules cannot initialise the card.

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
per token (~29 CPU layers instead of ~22). It nearly filled the 224 GiB of RAM
this was measured on; the box now has 256 GB, so it is less tight than the
number suggests. Use it as a quality reference, not a daily driver.

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
`multi-gpu-llm-toolkit/linux/scripts/benchmark-loaded-model.sh` so both engines are
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
(documented in `~/development/multi-gpu-llm-toolkit/doc/cuda-fa-blackwell.md`, with a
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

> **PARTLY SUPERSEDED by §21.** The `-rtr` component of this win holds only at
> small `-ub`. At `-ub 4096` on a gen5 x16 link, `-rtr` costs 64 % instead of
> gaining 25 %, because it blocks GPU offload of the host-resident experts.

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
and published as
[`daimonionnn/ds4@local/blackwell-discrete-fixes`](https://github.com/daimonionnn/ds4/tree/local/blackwell-discrete-fixes),
which also carries `run-cuda-local.sh` with the working invocation. Fixes
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

**`-muge`** (merge up/gate expert projections) **aborted the server** — but only
when combined with `-rtr`, which every kvram profile carries. **Fixed upstream
the same day** as
[`ee77f7ff` (#2306)](https://github.com/ikawrakow/ik_llama.cpp/commit/ee77f7ff);
verified here on that build with no local patch — 43 layers merged, 34 tensors
repacked, server up, no assert. What follows is the diagnosis as it stood. It walks all 43
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

---

## 17. How much of prefill is CPU work? (what a faster CPU could buy)

> **CORRECTED by §21.** The number below is real but describes the `-rtr` CPU
> path, not the machine. With `-rtr` off the experts run on the GPU and this
> share collapses. Do not use this section to justify buying a CPU.

Measured 2026-08-14, prompted by the question of whether to change platform.
§16 established that prefill is limited by the CPU rather than by memory; this
puts a number on it, which is what bounds any CPU upgrade.

### 17.1 Method

`--n-cpu-moe n` puts the expert FFN of n layers on the CPU. Sweeping n at a fixed
32k depth and timing prefill gives

    T(n) = T_other + n*t_cpu + (43-n)*t_gpu
         = [T_other + 43*t_gpu] + n*(t_cpu - t_gpu)

so a line fitted to time-per-token against n has slope `(t_cpu - t_gpu)`. Since
`t_gpu > 0` the slope understates the true per-layer CPU cost, making the CPU
share a **lower bound** — the safe direction for a purchasing decision.

Held fixed: 32 920-token prompts, `-ub 1024`, `-rtr`, `-nkvo`, MXFP4, thread
count. 2 repeats per point. `-ub 1024` rather than the profile's 2048 because
2048 is the prime suspect for the NaN abort (§18 / TODO 9) and a crash mid-sweep
would waste the run; the difference cancels out of a slope.

### 17.2 Result

| `--n-cpu-moe` | prefill at 32k | µs/token |
|---:|---:|---:|
| 17 | 459.5 | 2176 |
| 18 | 435.8 | 2295 |
| 19 | 406.2 | 2462 |
| 20 | 406.4 | 2461 |

    per-layer CPU cost   102.03 us/token   (fit R2 = 0.898)
    at n=17              2195 us/token  ->  456 tok/s
    of which CPU experts 1734 us  =  79 %      LOWER BOUND
    everything else      21 %

**Prefill is overwhelmingly CPU work.**

### 17.3 Where the fit is weak, and why it probably reads low

R² = 0.898 on four points is not a good fit, and the reason is visible in the
table: n=19 and n=20 are identical (406.2 vs 406.4). Beyond ~19 CPU-resident
layers the linear model breaks — most likely the CPU becomes memory-bandwidth
bound there, so further layers stop costing proportionally.

If so, the relevant slope for the region actually shipped (n=17) is the one from
the first pair, 118 µs/layer, which puts the CPU share at **92 %**. The honest
interval is **79–92 %**, nearer the top.

### 17.4 What it implies

| CPU part faster by | prefill | vs today |
|---|---:|---:|
| 1.5x | 618 tok/s | +36 % |
| 2x | 753 tok/s | +65 % |
| 3x | 962 tok/s | +111 % |
| infinite (Amdahl) | 2170 tok/s | +376 % |

Two consequences worth recording:

* Generation is not on this curve at all — it is bandwidth-bound, so a faster CPU
  does close to nothing for it. Puget Systems measured a 16 % token-generation
  drop from moving this platform's RAM 7200 -> 5600 MT/s, which is the same
  finding from the other side.
* The ceiling is high enough that plausible CPU upgrades are not capped by it,
  so the question is entirely whether a given CPU exploits the path. ik_llama's
  `MXFP4_R8` GEMM lives in `iqk_gemm_legacy_quants.cpp`, which carries 14
  `HAVE_FANCY_SIMD` branches; `iqk_common.h` switches loads from `__m256i` to
  `__m512i` under it. So AVX-512 hardware would use genuinely wider code —
  unlike vanilla llama.cpp, which largely does not.

### 17.5 A better experiment, not yet run

Varying **thread count** rather than layer placement would fit Amdahl directly
and avoid the placement confound entirely: fewer threads is a direct simulation
of a slower CPU. §16 has partial data. Worth doing before anyone spends money.

---

## 18. Micro-batch size: what `-ub` is worth, and what it does not touch

Measured 2026-08-14 with `depthbench.sh`, one fixed methodology on both sides:
`max_tokens` 160, temperature 0, a salt unique per request so nothing is served
from the prompt cache, 2+ repeats. Everything else identical -- MXFP4, `-rtr`,
`-nkvo`, `--n-cpu-moe 17`, `-b 4096`, f16 KV, `-c 131072`.

### 18.1 Prefill

| depth (tokens) | `-ub 1024` | `-ub 2048` | 2048 ahead by |
|---:|---:|---:|---:|
| 515 | 277.6 | — | |
| 1 027 | 356.2 | — | |
| 4 101 | 472.2 | — | |
| 32 701 | 462.9 | **485.3** | +4.8 % |
| 127 981 | 409.6 | **422.7** | +3.2 % |

Two things worth recording:

* **§11's 484.1 tok/s is confirmed.** The `-ub 2048` figure at 32k came out at
  485.3 under the new methodology — 0.25 % from the old number, measured months
  and several rebuilds apart. That validates §11 and the method at once.
* **Prefill peaks around 4k and shallow prompts pay a fixed overhead.** At 515
  tokens the rate is 277.6 tok/s, only 59 % of the 4k peak. Anything that
  measures prefill on short prompts will understate this configuration badly.

Repeatability differed between the two: 1.2 % spread at `-ub 1024`, **6.5 %** at
2048 (496.3 / 492.1 / 488.7 / 464.1 across four samples, the last lowest). Against
the ~2 % noise floor of §16 that is worth noting rather than explaining away.

### 18.2 Generation is untouched

| depth | `-ub 1024` | `-ub 2048` |
|---:|---:|---:|
| 32 701 | 20.07 t/s | 20.4-21.1 |
| 127 981 | 17.48 t/s | 17.2-17.4 |

Identical within noise, as expected: micro-batching governs how prompt tokens are
chunked, and generation processes one token at a time. So the whole `-ub` question
is a prefill trade, with nothing at stake for interactive token rate.

### 18.3 The whole `-ub` range, and why 3072 does not load

Recovered 2026-08-14 from logs of 2026-08-12, which had not been written up:

| `-ub` | `--n-cpu-moe` | CUDA0 compute buffer | prefill at 32k |
|---:|---:|---:|---:|
| 512 | 17 | 440.00 MiB | — |
| 1024 | 17 | 880 MiB | 462.9 |
| 2048 | 17 | 1760.01 MiB | 485.3 |
| 3072 | 17 | 2640.01 MiB | **CUDA OOM** |
| 4096 | **18** | 3520.02 MiB | **494.1** (497.2 / 491.0) |

`-ub 4096` runs, and is ~1.8 % ahead of 2048 — but not on equal terms: it needs
`--n-cpu-moe 18`, because 3520 MiB of compute buffer does not fit beside 17 layers
of experts. `-ub 3072` at `--n-cpu-moe 17` aborts for the same reason. That it is
still slightly ahead while carrying a worse placement suggests the micro-batch
gain has not run out by 4096, but the two effects are confounded here and a clean
comparison would need 4096 and 2048 at the same `n`.

**The compute buffer is exactly linear in `-ub`:**

    440/512 = 1760/2048 = 2640/3072 = 3520/4096 = 0.859 MiB per unit of -ub

Four points, no deviation. So whether a given `-ub` fits can be computed rather
than discovered by trying it. This is the same quantity TODO item 10 is about:
`-rtr` shrinks it (1760 vs 2496 MiB at `-ub` 2048), which is what makes
`--n-cpu-moe 17` reachable at all.

Those two runs prefilled 32 699 tokens each, so they say nothing about the abort
question — a third of the lower edge of the band.

### 18.4 What this cost buys

Dropping to `-ub 1024` costs 3-5 % of prefill. Whether that is worth paying is
the open question of TODO item 9: five NaN-logits aborts all happened at
`-ub 2048`, but a deliberate reversion run reached 397 077 prefilled tokens at
2048 without one, so the association is not established. The reproduction attempt
now runs under interactive traffic, which is the one condition the clean runs did
not share.

---

## 19. The NaN-logits abort: `-ub 2048` is implicated (2026-08-13/14)

Under sustained use the server aborts in `llama-sampling.cpp:745` with every
logit `nan`:

    =============================== Failed to sample token
    Data has been stored in probabilities.txt

Six occurrences over two days. This section records what the bisect established,
because most of it is negative results that are worth not repeating.

### 19.1 It follows prefilled VOLUME, not uptime

Time-to-abort varied 24x across runs — 259 min down to 11 — and moved inversely
with the prefill rate, while the tokens prefilled before each abort stayed inside
one band. That is what made the investigation tractable: exposure could be raised
deliberately to get answers in minutes instead of hours.

### 19.2 What was ruled out

Each by a run that removed exactly one thing and aborted anyway:

| suspect | how it was cleared | abort at |
|---|---|---:|
| prompt cache | `--cache-ram 0`, provably inert in the log | 94 372 |
| context checkpoints | `-ctx-ckpt 0`, zero created | 134 124 |
| run-time repack | `-rtr` off | 188 265 |
| hardware | no Xid, MCE or ECC events in the journal | — |
| driver | reproduced on 595.84 and 610.43.02 | — |
| the build | identical binary across aborting and clean runs | — |

The whole context-state caching layer is therefore innocent, and so is the
repack. Those flags were restored.

### 19.3 What survived

| `-ub` | traffic | runs | prefilled tokens | outcome |
|---:|---|---:|---|---|
| 2048 | interactive | 6 | 94k-331k | **all abort** |
| 2048 | synthetic | 1 | 397 077 | clean, but stopped rather than survived |
| 1024 | mixed | 1 | 532 148 | clean |
| 512 | interactive | 2 | 528 577 / 541 369 | clean |

Six of six at `-ub 2048`; none at anything smaller. The two clean 512 runs were
real interactive use, so **1.07 M tokens of the same workload that aborts six
times at 2048 pass without one**.

### 19.4 How the reversion test nearly misled us

`-ub 1024` was made the default once the first five aborts lined up against three
clean runs. That partition was not a controlled result, so 2048 was deliberately
put back. It then ran **397 077 prefilled tokens under benchmark load with no
abort** — past the 94k-295k band — which read as a refutation and was reported as
one.

It was not. Driven by interactive traffic instead, the same setting aborted at
**331 269** tokens: past the old upper edge, hence the false all-clear a minute
earlier. The lesson is that the band's upper edge was an artefact of the sample,
not a boundary.

### 19.5 The most specific lead: partial micro-batches

The two load shapes differ in a way that matters. Interactive traffic is mostly
short prompts — median 1 800 tokens in the crash-6 run, 12 of 23 below 2048,
i.e. a **single partial micro-batch**. The benchmark's eight huge uniform prompts
fill nearly every micro-batch, and did not reproduce it.

If the fault is in a partial micro-batch path at large `-ub`, that asymmetry is
exactly what would be seen. It is the first lead in this investigation that points
at code rather than at another thing to bisect. Untested; the experiment is a
generator of many short varied prompts at `-ub 2048`.

### 19.6 Shipping position

`-ub 1024`, costing 3-5 % of prefill (§18). `-ub 4096` is faster still (§18.3)
and is the same suspect, only more so. Mechanism unknown, nothing filed upstream
yet — see TODO item 9 for the open threads.

---

## 20. PCIe 5.0 x8 -> x16: +5 % prefill, and my transfer model was wrong (2026-08-17)

The Radeon 9700 was moved to a chipset PCIe 4.0 x4 slot, which let the RTX PRO
6000 come up at its full x16 (`pcie.link.width.max` went 8 -> 16). Re-measured
with `tools/depthbench.sh`, same `-ub 1024`, same depths, same methodology — only
the machine changed.

| depth | prefill x8 | prefill x16 | delta | generation x8 | generation x16 |
|---:|---:|---:|---:|---:|---:|
| ~515 | 277.6 | **286.4** | +3.2 % | 21.55 | 21.34 |
| ~1 020 | 356.2 | **384.4** | +7.9 % | 21.41 | 21.63 |
| ~4 100 | 472.2 | **478.0** | +1.2 % | 21.47 | 21.70 |
| ~32 700 | 462.9 | **486.4** | +5.1 % | 20.07 | 20.32 |
| ~128 000 | 409.6 | **436.7** | +6.6 % | 17.48 | **17.97** |

Prefill spreads within each point were 0.2-1.4 %, so everything except the 4k row
is comfortably outside noise. Mean prefill gain **+4.8 %**.

### 20.1 The bandwidth model was right and still predicted the wrong answer

Before the change I estimated the gain at "well under 1 %", from traffic volume:
activations cross the bus, not weights — `n_embd` 4096 at f16 is 8 KiB per token,
times 17 CPU-resident layers, times two directions, is ~272 KiB per token. For a
2048-token micro-batch that is 557 MB, ~22 ms at gen5 x8, against 70.7 s for a
32k prefill. **0.5 % of the time, so doubling the width could save at most 0.25 %.**

That arithmetic still checks out, which is exactly why the result is interesting:
**bandwidth cannot explain a 5 % gain**, so the bus was not costing what it moves.
The remaining candidate is latency and synchronisation — every layer boundary is
a point where one side waits for the other, 17 of them each way per micro-batch,
and what matters there is how fast a round trip retires rather than how much fits
through.

The generation column is the supporting evidence. Generation had been completely
insensitive to `-ub` (§18.2) because it processes one token at a time, so data
volume across the bus is negligible. It moved anyway — +2.8 % at 128k — which is
what a per-round-trip cost would do and what a bandwidth cost would not.

### 20.2 Two variables moved at once

The Radeon did not merely vacate lanes; it moved from a CPU-attached slot to the
chipset. So this measures "RTX at x16 **and** the second GPU off the CPU root
complex" as one change. The improvement is real and is what the machine now does,
but attributing it specifically to link width would be more than the data
supports.

### 20.3 The `-ub 1024` stability tax is now paid off

`-ub 1024` at x16 reaches **486.4 tok/s at 32k, above the 485.3 that `-ub 2048`
managed at x8** (§18.1). The configuration that has never aborted is now also
faster than the one that aborted six times. Whether `-ub 2048` would gain the
same ~5 % is untested and not worth testing on the shipped profile.

---

## 21. `-rtr` was costing 3x prefill — two strategies, and we had picked the wrong one (2026-08-17)

Prompted by a comparison the user ran: `multi-gpu-llm-toolkit` (upstream
llama.cpp) reached **1565 tok/s at 4k** on the *same MXFP4 file*, against 478
here. A 3x gap is not tuning, so something structural had to differ. It did.

### 21.1 What the other toolkit does differently

Its CUDA-only profile is `-c 131072 --n-cpu-moe 18 -b 8192 -ub 4096 -ngl 99 -fa on`
— **more** expert layers in host RAM than ours (18 vs 17), and no `-nkvo`. Its own
`doc/performance-model.md` states the mechanism plainly: with `--op-offload` on
(the default) *llama.cpp does not compute CPU-hosted expert layers on the CPU —
it ships them to the GPU*.

ik_llama has the same machinery: `ggml-backend.cpp:2558` initialises the
`op_offload` mask to `0xffffffff`, i.e. on for every op. **We were disabling it
by accident.** `-rtr` repacks host-resident experts into `MXFP4_R8`, and that type
exists only under `ggml/src/iqk/` — there is no CUDA kernel for it. A repacked
tensor therefore cannot be offloaded, and the CPU must do the GEMM.

So these are two strategies, not a flag and an improvement:

| | `-rtr` + small `-ub` | no `-rtr` + large `-ub` |
|---|---|---|
| experts computed on | CPU, AVX2/VNNI | **GPU** |
| crosses PCIe | activations | weights, amortised over the micro-batch |
| sensitive to PCIe width | barely | strongly |
| wins when | narrow link, strong CPU | wide link, strong GPU |

### 21.2 Measured

`tools/depthbench.sh`, `-rtr` off, `--n-cpu-moe 18`, `-b 8192 -ub 4096`, otherwise
the shipped profile (`-nkvo` kept), gen5 x16:

| depth | shipped (`-rtr`, ub 1024) | no `-rtr`, ub 4096 | factor | generation |
|---:|---:|---:|---:|---|
| 4 101 | 478.0 | **1 375.7** | 2.88x | 20.74 (was 21.70) |
| 16 194 | — | **1 562.8** | — | 20.27 |
| 32 701 | 486.4 | **1 529.0** | 3.14x | 19.55 (was 20.32) |
| 127 981 | 436.7 | **1 145.4** | 2.62x | 17.58 (was 17.97) |

Spreads 0.1-3.0 %. Generation is 1-4 % *lower* — a cheap price for 3x prefill.
Nothing is traded on quality: same weights, same quant, same arithmetic, just
executed on the other processor.

This puts ik_llama at 88 % of the other toolkit at 4k and 90 % at 16k, with no
tuning of this regime at all. The remaining gap is most likely `-nkvo`, which we
still carry and they do not.

### 21.3 What this invalidates

Three earlier conclusions in this file were measured correctly and interpreted
wrongly. They are left in place, with this section as the correction:

* **§11: "`-rtr` is worth +25 % prefill".** True at `-ub 512`, where streaming
  weights cannot amortise. At `-ub 4096` the same flag costs **-64 %**. It is a
  regime, not a property.
* **§17: "79-92 % of prefill is CPU work", and the CPU-upgrade advice built on
  it.** The measurement was sound; the framing was not. That share was a
  consequence of `-rtr` forcing the CPU path, not a property of the machine. The
  honest answer to "would a faster CPU help?" turns out to be **no — stop using
  `-rtr` instead**. The Amdahl ceiling derived there describes a configuration we
  should not be running.
* **§20's premise.** The PCIe x8 -> x16 change gave only +4.8 % *because* `-rtr`
  had made prefill insensitive to the link. In this regime the link is central,
  which is exactly why the other toolkit gained ~30 % from the same rewiring.

There is a lesson worth keeping: every one of those measurements was repeatable
and internally consistent, and all three still misled, because they characterised
a self-inflicted constraint. A cross-check against a different tool on the same
file found in one afternoon what a week of internal bisecting did not.

### 21.4 Not yet done

* `-nkvo` off, KV on the GPU, retuned `--n-cpu-moe` — the likely remaining 10 %.
* The whole `-ub 2048` abort investigation (§19) was conducted in the `-rtr` CPU
  path. Whether it reproduces in the streaming regime is unknown and must be
  re-established before this becomes the default.
* `--n-cpu-moe` and `-ub` want a fresh sweep here; 18/4096 was copied from the
  other toolkit, not tuned.

---

## 22. Tuning the `gpu-experts` regime: 486 -> 1794 tok/s at 32k (2026-08-17)

§21 found the regime; this is what four sweeps made of it. Every arm is a fresh
server, `tools/sweep.sh`, two repeats, `max_tokens` 160, temperature 0, unique
salt per request. Baseline reproduced across five independent runs to within
0.7 % (1366.0 / 1370.8 / 1375.0 / 1375.7 at 4k), so differences above ~1 % are
real.

### 22.1 Result

| | borrowed start | **tuned** |
|---|---|---|
| `--n-cpu-moe` | 18 | **19** |
| `-ub` / `-b` | 4096 / 8192 | **8192 / 8192** |
| `-nkvo` | on | on (forced, see 22.3) |
| `-t` / `-tb` | 24 / 24 | 24 / 24 (already right) |

| depth | kvram profile | gpu-experts, tuned | factor |
|---:|---:|---:|---:|
| 4 101 | 478.0 | 1 329.4 | **2.78x** |
| 32 701 | 486.4 | **1 793.6** | **3.69x** |
| 127 981 | 436.7 | 1 327.7 | **3.04x** |

Generation is 3-7 % lower (19.98 / 19.14 / 17.20 against 21.70 / 20.32 / 17.97).

### 22.2 `--n-cpu-moe` wants the floor, and the floor is set by `-ub`

Monotonic, with no flattening: each layer pushed to host RAM costs **1.9 %
prefill and 2.8 % generation** (n18 1528.4 -> n20 1457.4 -> n24 1353.6 at 32k).
That is a different shape from §17, where the same knob cost ~102 us/token and
the curve flattened above 19 — there the constraint was CPU GEMM, here it is
bytes over the link every batch, so nothing saturates.

Generation suffers more than prefill because a single decoded token has nothing
to amortise the transfer over.

### 22.3 `-nkvo` is not a choice here

With the KV back on the GPU the compute buffer goes **3520 -> 28 992 MiB** at
`-ub 4096` (8.24x — the attention scratch follows the KV home, the same ratio the
kvram profile recorded at `-ub 512`). `--n-cpu-moe` 19, 20 and 22 all fail to
load. Fitting it would need `-ub` under ~1150, which discards the amortisation
the whole strategy rests on. `-amb 512` changes neither the allocation (identical
28 992.18 MiB request) nor the speed (identical to baseline within 0.4 %); it does
not reach this path.

This explains the residual gap to `multi-gpu-llm-toolkit` rather than closing it.
That toolkit fits `--n-cpu-moe 18` **with KV on the GPU** and `-ub 4096` in the
same 96 GiB, so upstream's attention scratch is far smaller than ik_llama's. An
engine difference, not a flag.

### 22.4 `-ub` pays, `-b` pays only at depth, and neither can be trusted to fit

| `-ub` / `-b` / `n` | pp 4k | pp 32k | pp 128k |
|---|---:|---:|---:|
| 2048 / 8192 / 18 | 1029.3 | 1122.6 | — |
| 4096 / 8192 / 18 | **1370.8** | 1520.7 | 1145.4 |
| 4096 / 4096 / 18 | 1371.9 | 1435.9 | — |
| 6144 / 8192 / 18 | — | **OOM at depth** | — |
| **8192 / 8192 / 19** | 1329.4 | **1793.6** | **1327.7** |
| 12288 / 12288 / 20 | — | **OOM at depth** | — |
| 16384 / 16384 / 21 | — | **OOM at depth** | — |

* **The optimum depends on depth.** `-ub 4096` wins at 4k by 3.4 %; `-ub 8192`
  wins by 17.3 % at 32k and 15.9 % at 128k. At 4k a 4101-token prompt is a single
  micro-batch either way, so the larger value shows only its cost.
* **`-b` matters, but only at depth** — 0.08 % at 4k, +5.9 % at 32k for 8192 over
  4096, because a 32k prompt is four logical batches instead of eight. §2 called
  `-b` inert; it measured the `-rtr` path, where CPU GEMM buried the difference.
* **Loading is not fitting.** `-ub 6144` at n18 started, reported its 5280 MiB
  buffer, served a 4k prompt, then died in `cuMemCreate` at 32k. The reported
  compute buffer follows 0.859 MiB per unit of `-ub`, but the runtime allocator
  wants far more: 12288 had 4493 MiB of computed headroom and still died, while
  8192 survived on 4747. **Any fit check must run at depth**, which is why
  `tools/sweep.sh` measures 32k and treats a mid-run death as a recorded result.
  These are CUDA OOMs, not the NaN abort of §19.

### 22.5 Threads split exactly along the mechanism

`-tb` is **inert from 4 to 24** (0.03 % at 32k between 24 and 8) — a third
independent confirmation that prefill no longer runs on the CPU, after §21's
kernel reading and §20's PCIe sensitivity. §16 measured +32 % for prefill threads
in the `-rtr` path; here the knob does nothing.

`-t` still matters: 8 threads costs 4.1 % of generation at 4k and 6.0 % at 32k.
Decode ships little across the link, so host-side work still shows. Both stay at
24; `-tb` can be lowered to 8 for free if the CPU is wanted elsewhere.

### 22.6 Still owed

A stability soak. §19's six aborts were all in the `-rtr` path at `-ub 2048`;
this profile is a different code path at `-ub 8192` and nothing is known about it.
Until it has taken 300-500k prefilled tokens of real traffic without an abort, the
kvram profile stays the shipped default.

---

## 23. Head to head with `multi-gpu-llm-toolkit` — it depends on depth (2026-08-17)

The comparison that started §21, now measured on the tuned profile at the depths
the other toolkit publishes. Same MXFP4 file, same machine, same PCIe 5.0 x16.

Against that toolkit's **final** configuration (`--cuda-only`, `-b 8192 -ub 4096`,
`--n-cpu-moe 18`, `-t 24 -tb 24`, 600 W, memory offset +3009 MT/s):

| depth | ik_llama `gpu-experts` (tuned) | multi-gpu-llm-toolkit (final) | |
|---:|---:|---:|---|
| ~4k | 1 347.5 | **1 760.8** | theirs **+31 %** |
| 16 384 | **1 887.1** | *(1 731.6, pre-overclock run)* | not comparable |
| 32 768 | 1 804.1 | *not published* | |
| 65k | *not measured* | 1 505.3 | |
| ~128k | **1 327.7** | 1 179.9 | **ours +12.5 %** |

Generation at 128k: ours 17.20, theirs 16.66 — **ours +3.2 %**.

Our four measurements of the 32k point across the day were 1792.9 / 1793.6 /
1795.3 / 1804.1 — 0.6 % spread — so these are not noise.

**The hardware settings are not a confound.** Their final run raises the card to
600 W and applies a +3009 MT/s memory offset. Both are already in effect here:
`power.limit` is 600 W (which is also this card's maximum and default), and
`clocks.max.memory` reports 15 505 MHz against a stock supported maximum of
14 001 — a difference of 1504 MHz, i.e. 3008 MT/s, the same offset. So these
numbers are measured on the same silicon in the same state.

**The curves have different shapes, and that is the real finding.** Theirs falls
monotonically with depth (1760.8 -> 1505.3 -> 1179.9); ours rises then falls
(1347.5 -> 1887.1 -> 1804.1 -> 1327.7). Same cause as the crossover below: `-ub
8192` needs depth before it earns its keep, so we are behind on short prompts and
ahead on long ones.

### 23.1 Why the crossover

Their config is `--n-cpu-moe 18 -b 8192 -ub 4096`; ours tuned to `19 / 8192 /
8192`. At a 4k prompt our micro-batch is twice the whole prompt, so we get none
of the amortisation and pay only its cost — the extra expert layer that `-ub 8192`
forces into host RAM, at -1.9 % (§22.2). By 128k the larger micro-batch is working
and the order reverses.

A shallow-prompt variant is therefore available if it is ever wanted: `-ub 4096`
at `--n-cpu-moe 18` measured 1370.8 at 4k, narrowing their lead to 12 % rather
than closing it.

### 23.2 What still separates them, and what does not

`-nkvo` is the remaining structural difference (§22.3): they run attention on the
GPU, we cannot, because ik_llama's attention scratch is 8.24x larger and forces
the KV to host RAM. That is an engine difference, not a tuning gap, and it is the
most plausible source of their 4k advantage.

Caveats worth stating rather than burying: their harness predicts 128 tokens and
ours 160, and the prompt content differs. Neither should move prefill much, but a
few percent either way is not worth arguing over.

### 23.3 The honest summary

For raw prefill on this box the two are now comparable, each ahead at different
depths, after starting 3x apart. The gap that mattered was never tuning — it was
that this toolkit was computing experts on eight AVX2 cores while the other
shipped them to a Blackwell (§21).

---

## 24. The abort was very likely an upstream f16 overflow in DSA (2026-08-17)

> **Superseded by §32.** The fix went in, and the abort came back anyway on
> real traffic at an unchanged rate. The reasoning below is left intact
> because it is where the mistake is legible.

Seven aborts were chased across two days and five configuration variables. The
answer appears to have been sitting in upstream since the afternoon of the first
one.

### 24.1 What was missed

Our checkout was `2cda8d2d`, 2026-08-13 11:45. Three and a half hours later
upstream merged:

    ff141691  2026-08-13 15:22  Use f32 accumulation in CUDA DSA implementation (#2311)
    7cd62a3e  2026-08-15 09:41  More principled CUDA DSA (#2315)
                                "Just do V*softmax(K*Q) in f32 precision"

DSA is DeepSeek Sparse Attention — this model's attention. f16 saturates at
65 504; an accumulation that overflows gives `inf`, and `inf - inf` or `inf/inf`
gives NaN, which then propagates through everything downstream.

**That mechanism matches every observation:**

* **All logits NaN, never some.** All seven `probabilities.txt` dumps have 40 of
  40 candidates NaN, with identical token ids (38, 22, 10, 34 …) — the degenerate
  order left by a sort whose comparisons are all false. Not one bad expert; a
  poisoned tensor propagated to the output.
* **It tracks prefilled volume**, not uptime or requests: more attention computed
  is more chances to overflow.
* **It happened in both regimes.** §21's two strategies differ in where the
  *experts* are computed. Attention does not care, and the aborts did not either
  — which is exactly why every configuration bisect came back negative.

### 24.2 Verification so far

`tools/stress.sh` was written to reproduce the abort quickly by driving the one
thing the crashing runs had and the clean benchmarks did not: prompts far shorter
than `-ub`, i.e. a partial micro-batch per request, plus conversations that grow
so checkpoints are built.

On the updated build, at `-ub 8192`:

    123 requests, 610 103 prefilled tokens, 13 minutes, no abort

That is 1.8x the latest abort (331 269) and ~55 000 tokens/min, roughly 25x the
rate of real traffic — which is what makes the tool useful: a question that took
half a day now takes minutes.

**And it costs nothing.** Re-measured on the new build:

| depth | old build | new build (f32 DSA) | |
|---:|---:|---:|---:|
| 4k | 1347.5 | 1335.4 | -0.9 % |
| 16k | 1887.1 | 1867.2 | -1.1 % |
| 32k | 1804.1 | 1801.7 | -0.1 % |
| 128k | 1327.7 | 1330.9 | +0.2 % |

All inside the noise floor. Plausibly because DSA attends only ~512 positions
regardless of depth, so the precision change lands on a small share of the work.

### 24.3 What this is not

One clean synthetic run against seven aborts. The build changed, so the run
clears **this build**, and says nothing about the configuration hypotheses that
were live before it — a distinction `tools/stress.sh` now prints in its own
output, because getting it backwards is exactly the error that made `-ub 2048`
look exonerated for an hour on 2026-08-14.

What is still owed is real traffic: Hermes, over a day, at the rate that produced
seven aborts in two.

**That debt was paid on 2026-08-19 and the answer was no** (§32). Note what this
paragraph got right and still lost: it named the correct outstanding test, and
the profile shipped as default anyway on the strength of the synthetic run.

### 24.4 The lesson, which is not a technical one

Every configuration variable was bisected — prompt cache, checkpoints, `-rtr`,
`-ub`, placement, driver, hardware — with runs that were individually sound. The
one thing never checked was whether the engine had moved. `git log HEAD..origin`
costs seconds and would have surfaced it on day one.

Two of the three big findings of these two days came from looking outside the
repository: this one, and §21, which came from the user comparing against another
toolkit. The internal bisecting produced correct measurements and wrong
conclusions.

---

## 25. DDR5 6267 -> 7400: generation +5.5 %, prefill +2.7 % (2026-08-17)

Four DIMMs at 6267 MT/s replaced by two at 7400. Same tool, same depths, same
methodology, same build — only the memory changed.

| depth | pp 6267 | pp 7400 | | tg 6267 | tg 7400 | |
|---:|---:|---:|---:|---:|---:|---:|
| 4k | 1335.4 | **1382.1** | +3.5 % | 20.07 | **21.23** | +5.8 % |
| 16k | 1867.2 | **1909.7** | +2.3 % | 19.85 | **21.07** | +6.1 % |
| 32k | 1801.7 | **1845.8** | +2.4 % | 19.18 | **20.22** | +5.4 % |
| 128k | 1330.9 | **1362.2** | +2.4 % | 17.17 | **18.00** | +4.8 % |

Generation gains roughly twice what prefill does, at every depth.

### 25.1 The size of the gain is the interesting part

Dual channel at 7400 is ~118.4 GB/s against ~100.3 for four DIMMs at 6267 —
**+18 % theoretical**. Generation gained 5.5 %.

That is what the mechanism predicts. At `--n-cpu-moe 19`, 19 of 43 layers keep
their experts in host RAM: 44 % of the expert reads. The rest are already in VRAM
and do not care how fast the DDR5 is. So

    0.44 x (1 - 1/1.18) = 6.7 % expected, 5.5 % measured

with the shortfall being everything the model ignores — attention, sampling,
per-token overhead. Close enough to treat the mechanism as confirmed from a third
direction, after §21's kernel reading and §22.5's inert `-tb`.

**Prefill moving at all is worth noting**, since in this profile the GPU computes
the experts. It still has to *read* them out of host RAM once per micro-batch
before shipping them across PCIe, so faster memory helps that read — just far
less than it helps decode, which re-reads per token with nothing to amortise
over.

### 25.2 Capacity

RAM dropped 224 -> 122 GiB with the swap. Every profile still fits:

| profile | `--n-cpu-moe` | host weights | free |
|---|---:|---:|---:|
| gpu-experts (default) | 19 | 60.6 GiB | 61.4 |
| kvram 128k | 17 | 54.2 | 67.8 |
| 512k (plus ~21.5 GiB KV in RAM) | 19 | 60.6 | ~40 |

Even the sweep's `--n-cpu-moe 24` arm (76.5 GiB) leaves 45 GiB. The headroom that
was lost was never being used.

---

## Appendix: primary data

Sections §18 onwards were produced by `tools/depthbench.sh` and `tools/sweep.sh`,
which write a self-describing report per run — the configuration in it is read
back from the server log, not from what was requested. Those reports are in the
repository so the numbers quoted above can be checked against their source:

| section | report in `results/` |
|---|---|
| §18 depth curve at `-ub 1024` | `depthbench-ub1024-20260814-043225.md` |
| §18 the 1027-token point | `depthbench-ub1024-20260814-044737.md` |
| §18 independent 32k re-measure | `depthbench-ub1024-20260814-044910.md` |
| §18.1 `-ub 2048` at 32k | `depthbench-ub2048-20260814-055007.md` |
| §20 PCIe x8 -> x16 | `depthbench-ub1024-20260817-082855.md` |
| §21 `-rtr` off, first measurement | `depthbench-ub4096-20260817-162505.md` |
| §22.3 `-nkvo` off | `sweep-…-20260817-165008.md` |
| §22.3 `-amb 512` | `sweep-…-20260817-165522.md` |
| §22.2 `--n-cpu-moe` 14-24 | `sweep-…-20260817-165941.md` |
| §22.2 `--n-cpu-moe 17` boundary | `sweep-…-20260817-170811.md` |
| §22.4 `-ub` / `-b` | `sweep-…-20260817-170943.md` |
| §22.4 `-ub` 12288 / 16384 | `sweep-…-20260817-172100.md` |
| §22.5 threads | `sweep-…-20260817-173130.md` |
| §23 head-to-head depths | `sweep-…-20260817-185348.md` |
| §24 updated build | `sweep-…-20260817-193021.md` |
| §25 DDR5-7400 | `sweep-…-20260817-232449.md` |
| §34 DDR5-6667, under gdb | `depthbench-ub8192-20260820-102351.md` |
| §34 DDR5-6667, `IK_GDB=0` control | `depthbench-ub8192-20260820-103046.md` |
| §37 probes + scan, graphs off | `depthbench-ub8192-20260821-160945.md` |
| §37 probes + scan, graphs on | `depthbench-ub8192-20260821-161648.md` |

(`sweep-…` is `sweep-deepseek-v4-flash-gpu-experts-128k-`.)

The rest of `results/` is ignored by git: it is output from ad-hoc scripts that no
longer exist, backing §8-§16, and nothing cites it by name.

One of these carries a correction. `depthbench-ub2048-20260814-055007.md`
originally attributed its own interruption to the NaN abort; the server had in
fact been stopped by hand. A dropped connection looks identical to a crash from
the client side, so the tool no longer guesses — see the note in that file.

---

## 26. 256 GB DDR5-6400: no measurable speed, and the bandwidth law holds (2026-08-18)

Third memory configuration on the same machine, same build, same profile, same
tool. This one is 4 x 64 GB at 6400 MT/s, replacing 4 x 56 GB at 6267.

| depth | 6267 (§20) | **6400** | delta | 7400 dual (§25) |
|---:|---:|---:|---:|---:|
| 4k | 1335.4 / 20.07 | **1354.8 / 20.05** | +1.5 % / 0 % | 1382.1 / 21.23 |
| 16k | 1867.2 / 19.85 | **1888.0 / 19.84** | +1.1 % / 0 % | 1909.7 / 21.07 |
| 32k | 1801.7 / 19.18 | **1802.1 / 19.18** | 0 % / 0 % | 1845.8 / 20.22 |
| 128k | 1330.9 / 17.17 | **1335.1 / 17.14** | +0.3 % / -0.2 % | 1362.2 / 18.00 |

Prefill +0.7 % on average, generation exactly zero. Both inside the noise floor,
which is what §25's model predicted: 6400 against 6267 is +2 % of bandwidth, and
only 44 % of the expert reads come from host RAM, so the ceiling on the gain was
0.9 % before anything was measured.

### 26.1 Three points now, and generation tracks bandwidth alone

| MT/s | bandwidth | tg at 32k |
|---:|---:|---:|
| 6267 | 100.3 GB/s | 19.18 |
| 6400 | 102.4 | **19.18** |
| 7400 (dual DIMM) | 118.4 | 20.22 |

+2 % of bandwidth bought 0 %; +18 % bought 5.4 %. Nothing else in the memory
configuration moved the number — not capacity, not rank count, not the move from
four DIMMs to two. §12.1 says the same about latency from the other side: the
tighter kit at the same 7400 measured within 1 % on every depth.

**So for this workload, memory is a single-variable problem: MT/s.** CAS latency
and capacity can be chosen freely, and what capacity buys is context and
`--n-cpu-moe` headroom rather than speed.

### 26.2 Stability

`tools/stress.sh`, the shape that used to abort: **406 384 prefilled tokens, 84
requests, 9 minutes, clean.** That is 1.2x the latest historical abort. Zero
`Failed to sample`, zero CUDA errors, and zero machine checks or Xid events in
the journal afterwards — the EDAC and rasdaemon lines at boot are the subsystem
initialising, not faults. (One of them matches a naive `edac` grep only because
"redaction" contains the string.)

### 26.3 What 256 GB actually costs

`free` reports 244.75 GiB, not 256, and the missing 11 GiB is accounted for:

    physically installed   256.00 GiB
    e820 usable            253.27 GiB   firmware took 2.73 (PCIe MMIO, ACPI, SMM)
    after kernel reserve   244.43 GiB   kernel took 8.84
    MemTotal               244.75 GiB

The largest single item is visible in the kernel command line:
`crashkernel=…,128G-:4096M` reserves **4 GiB** outright, plus 256 MB low, because
the rule scales with installed RAM. The rest is mostly the `struct page` array —
67 M pages at 64 bytes is ~4.3 GiB, and it grows with capacity by construction.

`crashkernel=no` would return 4.25 GiB at the cost of kernel crash dumps. Worth
knowing but not recommended by default; the practical ceiling is ~249 GiB.

---

## 27. Clearing the TODO backlog: two old mysteries were `-rtr` all along (2026-08-19)

Five open items measured in one chain. Two of them turned out not to exist any
more, one confirms something I had wrongly cast doubt on, one profile converts,
and one refuses to.

### 27.1 `-rtr` really does shrink the compute buffer (item 10 — CONFIRMED)

Controlled this time: `--n-cpu-moe 18` and `-ub 2048` held fixed, only `-rtr`
varying.

| `-rtr` | GPU weights | compute buffer |
|---:|---:|---:|
| on | 89 366.93 MiB | **1 760.01 MiB** |
| off | 89 366.93 MiB | **2 496.01 MiB** |

Byte-identical weights, +736 MiB (+42 %) without the repack. **The original claim
holds and my later doubt was wrong** — I had inferred from the `gpu-experts`
profile reporting exactly `0.859 x -ub` without `-rtr` that the flag was
irrelevant, but that reading came from a different `-ub`, `-b` and `--n-cpu-moe`,
none of them controlled.

§26.3 below explains the loose end that misled me: the per-unit rate is not a
constant.

### 27.2 The compute buffer scales with CONTEXT too, not only `-ub`

Discovered while the 256k conversion kept running out of memory:

| context | `-ub` | compute buffer | per unit |
|---:|---:|---:|---:|
| 131072 | 8192 | 7 040 MiB | 0.859 |
| **262144** | 8192 | **11 136 MiB** | **1.359** |

So §18.3's "0.859 MiB per unit of `-ub`" is a **131072-only** law. Any headroom
arithmetic has to be redone per context, which is exactly the mistake that cost
two OOM arms below.

### 27.3 The 256k profile converts, and the trade is better than it looks (12.2)

| depth | baseline `-rtr` n18 ub2048 | **gpu-experts n21 ub8192** | prefill | generation |
|---:|---:|---:|---:|---:|
| 4k | 451.2 / 20.29 | **1 100.6 / 18.12** | 2.44x | -10.7 % |
| 32k | 468.2 / 19.32 | **1 636.9 / 17.43** | 3.50x | -9.8 % |
| 128k | 429.3 / 17.11 | **1 258.8 / 15.67** | 2.93x | -8.4 % |

The generation cost is roughly twice the 131072 profile's, and it is arithmetic
rather than a property of streaming: n21 against n18 is three more expert layers
in host RAM at §22.2's measured -2.8 % each, i.e. -8.4 % of the -10 %. The extra
layers are what `-ub 8192` costs at this context.

**Percentages mislead here.** On a 128k prompt:

    prefill    298 s -> 102 s     saves 196 seconds
    a 500-token answer  29.2 s -> 31.9 s     costs 2.7 seconds

Three minutes of waiting for the first token against under three seconds of
slower typing, on a profile that exists for long contexts. Shipped: `--n-cpu-moe 21`,
`-ub 8192`, `-b 8192`.

`--n-cpu-moe` 19 does not load and 20 dies at depth in `cuMemCreate`, both for the
reason in §27.2.

### 27.4 The MTP profiles cannot be converted — they do not fit (12.3)

Both arms failed before producing a number: `n21` at `-ub 8192` would not load,
`n20` at `-ub 4096` loaded and died at depth. CUDA OOM in both cases, not the NaN
abort.

The draft model is the reason, and I underestimated it: 5.5 GB of weights **plus
its own KV cache** (`-cd 8192`), all resident in VRAM, leaving too little for a
large micro-batch. I had expected this experiment to fail on generation; it fails
one step earlier, on fitting.

They stay on `-rtr`, and deservedly: MTP measures **27.01 tg at 4k and 26.51 at
32k**, against 20.29 for the plain profile — **+33 %**, the highest generation
recorded on this model. Trading that for prefill would defeat the point of the
profile.

### 27.5 The "dip band" is gone (item 4)

§12 recorded generation collapsing to 14-15 t/s in a 1k-16k band, against 21-24
outside it, and the mechanism was never found. Measured again in the streaming
regime:

| depth | 523 | 1 019 | 2 055 | 4 099 | 7 991 | 16 195 | 32 737 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tg | 20.06 | 20.52 | 20.14 | 20.45 | 20.37 | 20.17 | 19.38 |

Flat to within 2.3 % across the whole band. **The dip belonged to the `-rtr` CPU
path.** Closed without ever being explained, because the configuration that caused
it is no longer used.

(Prefill peaked at 16 195 tokens with **1 912.2 tok/s**, the highest figure
measured on this machine.)

### 27.6 The 4-core anomaly does not reproduce (item 6)

| `-t` | pp 4k | tg 4k | pp 32k | tg 32k |
|---:|---:|---:|---:|---:|
| 24 | 1365.8 | 20.39 | **1821.0** | 19.50 |
| 16 | 1371.8 | **20.65** | 1761.6 | **20.50** |
| 8 | 1375.7 | 19.65 | 1764.8 | 18.89 |
| 4 | 1368.9 | 16.36 | 1761.8 | 15.84 |

§16 measured 4 threads producing an unexplained **23.18 t/s** — higher than
configurations with far more cores. Here generation falls monotonically as threads
are removed, with no anomaly anywhere. Same verdict as §27.5: it was a property of
the path that no longer runs.

Prefill is flat across the whole range (1366-1376 at 4k), a fourth independent
confirmation that it does not touch the CPU (§22.5).

`-t 16` and `-t 24` are within noise of each other and neither is clearly better —
16 wins generation at 32k, 24 wins prefill there. Left at 24.

---

## 28. The compute buffer is a max(), not a sum — and item 10 dissolves (2026-08-19)

No new run: nine measurements already in the logs, sorted, answer the question
§27.1 left open.

| context | `-ub` | `-rtr` | measured | `rate x ub` | |
|---:|---:|---:|---:|---:|---|
| 131072 | 2048 | **on** | 1 760 | 1 760 | ✓ |
| 131072 | 2048 | off | **2 496** | 1 760 | ✗ |
| 131072 | 4096 | off | 3 520 | 3 520 | ✓ |
| 131072 | 8192 | off | 7 040 | 7 040 | ✓ |
| 131072 | 12288 | off | 10 560 | 10 555 | ✓ |
| 131072 | 16384 | off | 14 080 | 14 073 | ✓ |
| 262144 | 2048 | on | 2 784 | 2 783 | ✓ |
| 262144 | 8192 | off | 11 136 | 11 133 | ✓ |

Every point fits

    compute buffer = max( rate(context) x ub , floor(-rtr) )

with `rate` = 0.859 MiB/unit at 131072 and 1.359 at 262144, and a **floor of
~2496 MiB that exists only when `-rtr` is off**.

### 28.1 What that means

**`-rtr` does not shrink the buffer.** Without it the GPU computes the
host-resident experts (§21), which needs somewhere to stage those tensors — and
staging space is a fixed size, independent of how many tokens are in the
micro-batch. With `-rtr` those ops run on the CPU and the staging area is not
needed at all.

So this was never a separate phenomenon: it is §21's mechanism seen from the
allocator's side.

It also explains why the effect appeared and vanished depending on where one
looked. At `-ub 2048` the floor (2496) exceeds the proportional term (1760) and
`-rtr` is clearly visible; above `-ub` ~2900 the proportional term takes over and
the two configurations report identical buffers. **That is exactly what misled me
on 2026-08-17** into recording item 10 as doubtful: the comparison used
`-ub 4096`, where the floor is invisible.

### 28.2 What is left, and why it does not matter

The floor's size (~2496 MiB) is not derived from anything, and whether it scales
with context is unmeasured — there is no 262144 point at small `-ub` with `-rtr`
off. Both are irrelevant in practice: every shipped profile runs `-ub` well above
the crossover, so the floor never decides anything. Item 10 is closed on that
basis rather than on a complete model.

---

## 29. The 512k profile: 5.3x prefill, and the two levers finally conflict (2026-08-19)

Converted and swept the same way as the others. It is the largest gain measured
on any single profile — and the first context where `-ub` and `--n-cpu-moe` cannot
both have what they want.

### 29.1 Result

| depth | baseline `n19 ub512` | **tuned `n25 ub8192`** | factor |
|---:|---:|---:|---:|
| 4k | 309.9 / 20.34 | **1 068.9 / 16.91** | **3.4x** |
| 32k | 323.1 / 19.26 | **1 721.3 / 16.26** | **5.3x** |
| 128k | 301.7 / 17.32 | **1 334.9 / 14.86** | **4.4x** |

The old profile's prefill was flat at 300-320 tok/s regardless of depth, which is
what `-ub 512` does: the micro-batch is too small for the weight streaming to
amortise over, so depth changes nothing.

### 29.2 Where the levers fight

At 524288 the compute buffer is **21 376 MiB at `-ub 8192`** — three times the
131072 figure for the same batch. So a large micro-batch has to be paid for with
expert layers pushed to host RAM, and both sides were measured:

| arm | pp 4k / 32k / 128k | tg 4k / 32k / 128k |
|---|---:|---:|
| **n25 / ub 8192** | 1069 / **1721** / **1335** | 16.91 / 16.26 / 14.86 |
| n22 / ub 4096 | **1112** / 1454 / 1141 | **18.54** / **17.71** / **16.06** |
| n23 / ub 4096 | 1096 / 1429 / 1123 | 17.95 / 17.25 / 15.67 |

`-ub 4096` wins generation by ~9 % and loses prefill by ~17 %. At 32k that is
nearly a wash in wall-clock — 3.5 s of prefill against 2.6 s of answer — but the
prefill difference scales with prompt length and the generation difference does
not:

    512k prompt:   prefill  383 s  vs  449 s     -> 66 s
    500-token answer:        33.6 s vs 31.1 s    -> 2.5 s

For a profile that exists to hold half a million tokens, that resolves in favour
of the batch. **Shipped: `--n-cpu-moe 25`, `-ub 8192`, `-b 8192`.**

n23 confirms §22.2's rule from the other side: at fixed `-ub`, fewer layers is
strictly better (n22 beats n23 on both axes), so `--n-cpu-moe` belongs at the
floor of what fits.

### 29.3 The floor is higher than arithmetic predicts, again

Predicted `n23`; measured that n22 and n23 do not load at all, and **n24 loads,
serves 4k, then dies at 32k** with 6728 MiB of computed headroom. n25 leaves 9992
and survives.

The required headroom grows with context: 4747 MiB sufficed at 131072, 7176 at
262144, and 6728 was not enough at 524288. §22.4's warning that "loading is not
fitting" applies more strongly the longer the context.

### 29.4 The buffer's growth is not linear, so 1M cannot be extrapolated

| context | buffer at `-ub 8192` | per unit |
|---:|---:|---:|
| 131072 | 7 040 | 0.859 |
| 262144 | 11 136 | 1.359 |
| 524288 | **21 376** | **2.609** |

The increments are 4 096 and 10 240 MiB — the second is 2.5x the first, so the
two-point line fitted before this run **under-predicted 524288 by 10.6 %** and
would under-predict 1 048 576 by more. Extrapolating the same slope suggests
~41 900 MiB and `--n-cpu-moe` near 30, i.e. 30 of 43 layers in host RAM, but that
is a guess resting on a curve that has already bent once. A 1M profile would have
to be measured.

### 29.5 An oddity worth recording

At both 32k and 128k the 524288 profile out-prefills the 262144 one
(1721 vs 1637, 1335 vs 1259) despite carrying four more expert layers in RAM,
which §22.2 says should cost ~7.6 %.

The plausible difference is that the 512k profile ships `-ctx-ckpt 0
--cache-ram 0` — memory discipline for a context where a single checkpoint is
~1744 MiB — so it never pauses to build checkpoints during prefill, while the 256k
profile does. §11.3 measured what checkpoints buy on re-send; this suggests they
also cost something to create. **Untested**: one run of the 256k profile with
`-ctx-ckpt 0` would settle it.

---

## 30. Context checkpoints cost 11-27 % of prefill (2026-08-19)

§29.5 noticed the 524288 profile out-prefilling the 262144 one despite carrying
four more expert layers in host RAM, and guessed at checkpoints. It was
checkpoints, and the size of the effect is larger than the anomaly that revealed
it.

### 30.1 Measured on `deepseek-v4-flash-gpu-experts-256k`

| depth | checkpoints on | `-ctx-ckpt 0` | prefill | generation |
|---:|---:|---:|---:|---:|
| 4k | 1102.8 / 18.04 | **1398.4 / 19.66** | **+26.8 %** | +9.0 % |
| 32k | 1635.5 / 17.40 | **1881.0 / 18.72** | **+15.0 %** | +7.6 % |
| 128k | 1254.7 / 15.68 | **1396.1 / 16.90** | **+11.3 %** | +7.8 % |

The control arm reproduced the profile's own figures to 0.2 %, so these are real.

**The prompt cache is free.** A third arm with `-ctx-ckpt 0 --cache-ram 0`
measured 1392.0 / 1872.2 / 1397.7 — identical to dropping checkpoints alone.
Every bit of the cost is checkpoint creation; `--cache-ram` can stay on without
paying anything for it.

The prefill gain shrinks with depth (27 -> 15 -> 11 %) while the generation gain
stays near 8 %. That fits the mechanism: a checkpoint is a fixed slab of copying
(~872 MiB at 131072, ~1744 at 262144), so it is a large share of a shallow
prompt's work and a smaller share of a deep one — while a token being generated
pays the same toll however deep the context is.

**1881 tok/s at 32k is the highest prefill measured on this machine**, higher
than the 131072 profile's 1830 with checkpoints on. So this is not a 262144
finding: every profile in the repository is paying it.

### 30.2 What this does NOT mean

It is not an argument for turning checkpoints off. `tools/depthbench.sh` salts
every prompt uniquely so nothing is ever reused — by construction it measures
their **cost with none of their benefit**.

§11.3 measured the other side: checkpoints, not the prompt cache, are what make a
re-send free. So the trade is

* **cost** 11-27 % of prefill throughput on every prompt;
* **benefit** up to 100 % of prefill on any prompt whose prefix repeats.

For agent traffic — a fixed system prompt, a conversation that grows by one turn
at a time — the prefix repeats constantly and the benefit is far larger than the
toll. For one-shot passes over long documents that share nothing, checkpoints are
pure loss.

A rough decision rule: if more than about a fifth of prefilled tokens would be
served from a checkpoint, keep them.

### 30.3 Consequently

Nothing is changed on that basis alone, because the answer depends on a workload
property this repository has never measured — the actual prefix-reuse rate of
Hermes traffic. That is now TODO item 14, and it is measurable from the server
logs of a normal working day rather than by another benchmark.

The one place the decision is already clear is the 524288 profile, which ships
`-ctx-ckpt 0` for memory reasons (a checkpoint there is ~1744 MiB and 32 of them
would be ~55 GiB). §29.5's "oddity" is now explained and is not an oddity: that
profile was simply not paying a toll the others were.

---

## 31. Checkpoints at 524288 cost twice what they cost at 262144 (2026-08-19)

§30 priced checkpoint creation on the 262144 profile. Repeated on 524288, where a
checkpoint is 3487 MiB rather than 1743.

| `-ctx-ckpt` | pp 4k | pp 32k | pp 128k | tg 32k |
|---:|---:|---:|---:|---:|
| **0** | **1060.4** | **1717.9** | **1330.6** | **16.42** |
| 4 | 764.9 | 1352.7 | 1098.6 | 14.25 |
| 8 | 769.1 | 1340.8 | 1098.1 | 14.34 |
| 32 | 765.7 | 1343.3 | — | 14.26 |

Cost of having them at all: **-27.9 % prefill at 4k, -21.3 % at 32k, -17.4 % at
128k, and a flat -13 % of generation** at every depth. Roughly double §30's
figures for 262144, which is what a checkpoint twice the size should cost.

### 31.1 `-ctx-ckpt N` controls memory, not overhead

4, 8 and 32 are indistinguishable — within 0.9 % on every one of the six points
they share. An eightfold range in the limit changes nothing.

So checkpoints are created on a fixed schedule and `N` only decides **how many are
retained**. Lowering it to save time does not work; lowering it saves RAM. Since
only 4.6 % of checkpoints are ever restored (§30.2 / TODO 14), a low N costs
almost no hit rate — but it also buys no throughput.

### 31.2 The 32-checkpoint arm OOM-killed the machine

The `-ctx-ckpt 32` arm never finished its 128k point. At 524288, 32 checkpoints
are 109 GiB; with 80.7 GiB of host weights and 21.5 of KV that is 211 GiB of 244,
and the kernel took the server at **231 GiB RSS** — along with the user's editor:

    Out of memory: Killed process 613298 (llama-server) anon-rss:242085500kB
    app-code-18973.scope: Failed with result 'oom-kill'

My own arithmetic had said 211 of 244 GiB "fits", and it does — **it fits the
model, not the desktop**. A headroom calculation for a workstation has to leave
room for the workstation. That arm should have carried a `--cache-ram` ceiling or
not been run.

### 31.3 What ships

* **524288**: `-ctx-ckpt 0`. Double the cost of 262144 and a demonstrated memory
  hazard, against a benefit that is no larger.
* **262144**: checkpoints kept, with `--cache-ram 32768` added — roughly 19
  checkpoints, ~32 GiB. They earn 8.4 : 1 on real traffic (TODO 14) and now have
  the ceiling they always needed. That ceiling only works because of the local
  patch in `docs/external/local-cache-limit.patch`; upstream does not enforce
  `--cache-ram` when the cache holds a single prompt, which is the bug filed as a
  comment on ikawrakow/ik_llama.cpp#2320.
* **131072**: unchanged, checkpoints on at defaults.


---

## 32. The DSA fix did not stop the abort (2026-08-19)

§24 concluded the NaN abort was "very likely" an upstream f16 overflow in DSA,
fixed by `ff141691`. Real traffic disagrees. That section stays as written —
including its reasoning, which still looks sound — because the correction is
worth more next to the claim than in place of it.

### 32.1 What happened

195 minutes of Hermes traffic on `8337e4cd`, 422 tasks, the shipped
`gpu-experts-128k` profile. Aborted at depth 16 934:

    slot create_check: created context checkpoint 18 of 32 (pos_max = 16933, ...)
    kv cache rm [p0, end) ... p0=16934
    =============================== Failed to sample token
    llama-sampling.cpp:745: Fatal error

Same signature as the seven before it: 40 of 40 candidates NaN, same degenerate
token order. Saved as `crash9`.

Checking the other logs on this build turned up a **second** abort nobody had
noticed, on 2026-08-18 at depth 92 008, immediately after a checkpoint restore.
Saved as `crash8`; its `probabilities.txt` was already overwritten by crash9,
since that file is written to the working directory under a fixed name.

### 32.2 The fix is genuinely in the build

Not a stale binary, and not the wrong commit:

    git -C ik_llama.cpp merge-base --is-ancestor ff141691 HEAD   -> yes
    binary is 56 s newer than the last source change

### 32.3 The rate did not move

| build | serving time | tasks | aborts | rate |
|---|---:|---:|---:|---|
| `2cda8d2d` / `7ebbb906`, before the fix | 23.0 h | 518 | 7 | 1 per 3.3 h |
| `8337e4cd`, with the fix | 7.4 h | 634 | 2 | 1 per 3.7 h |

Within noise of each other. **Two events cannot measure a rate** — the interval
around 1-per-3.7 h is enormous, and this table cannot exclude a large real
improvement. What it does exclude is the thing §24 needed: that the abort was
gone. It is not.

### 32.4 What this rules in and out

The abort now spans **both** builds, **both** expert strategies (`-rtr` on for
five, off for four) and `-ub` 1024, 2048 and 8192. Every configuration bisect in
§19 came back negative, and the engine change that was supposed to explain that
does not.

Two patterns were tested against this log and **both failed**, which is the only
reason they are recorded — each looked convincing from the crash excerpts alone:

* *A checkpoint operation immediately precedes every abort.* True, and nearly
  vacuous: 293 checkpoints were created across 422 tasks. Almost anything is
  immediately preceded by a checkpoint operation here.
* *The prompt cache reports `n_past != n_past_prompt` before the abort.* Present
  in four of the older excerpts and it looks like an off-by-N. But it occurs in
  12 % of ordinary cache restores, and crash9 does not have it (15 803 = 15 803).

### 32.5 Standing

The abort is **open**, not fixed. `stress.sh` clearing 610 k tokens (§24.2)
turned out to be a synthetic run agreeing with a hypothesis rather than testing
it; the honest reading now is that it does not reproduce the abort, so it cannot
clear a build either.

Nothing here is a reason to move off the profile: it is roughly one abort per
several hours of heavy use, the server restarts, and the alternative costs 3.8x
prefill. But `default.env` should stop describing the cause as understood.

Worth filing upstream, with crash8/crash9 attached — the earlier reports were
against a build that has since been fixed, so they were arguably answered.

---

## 33. A backtrace at last, and the abort made survivable (2026-08-20)

### 33.1 The tenth abort, and the first stack

Abort #10 came 5.8 minutes after a restart, on the second task, at depth 29 228 —
the fastest yet. Same signature as the other nine: 40 of 40 candidates NaN, same
degenerate token order. The rate on the fixed build is now 1 per 2.5 h across
7.6 h, which is if anything worse than before #2311, though three events still
measure nothing.

It was the first caught with `IK_GDB=1`, so there is finally a stack:

    #5  ggml_abort (file="llama-sampling.cpp", line=745) at ggml.c:266
    #6  llama_sample_token_with_rng_impl at llama-sampling.cpp:745
    #8  llama_sampling_sample_impl (idx=4) at common/sampling.cpp:556
    #10 server_context::process_batch_tokens (n_batch=8192) at server-context.cpp:4791
    #11 server_context::update_slots at server-context.cpp:4998

**It does not locate the cause**, and was never going to: `llama_decode` has
returned by then. It confirms only what the dumps already said — the logits
arrive poisoned, the sampler is the messenger.

### 33.2 What it did show

`server-context.cpp:4790` already wraps the sampling call in try/catch: log
"sampling failed, releasing slot", return a 500 for that request, release the
slot, carry on. **That handler has never been reachable.** `GGML_ABORT` →
`ggml_abort` → `abort()` is not a C++ exception; frame #4 in the stack is
`__GI_abort`. A condition the server knows how to survive was killing the
process, and every other in-flight request with it.

`docs/external/local-sampler-throw.patch` makes it throw instead. Verified by
calling the sampler directly with 40 NaN candidates, exactly what the dumps show:
the exception is caught, the process lives.

The dump filename is now unique per abort. It was a fixed path in the working
directory, so each abort destroyed the previous evidence — which is how the dump
from abort #8 was lost.

**The test caught a bug in the patch before it shipped.** The first version keyed
the name on seconds plus a static counter; the counter is per-process, so two
servers aborting in the same second would collide. Three separate runs produced
one file. Switched to microseconds: three runs, three files.

### 33.3 Surviving is not enough: the poisoned state has to go

`slot.release()` resets the slot but keeps `cache_tokens`, the KV cells and the
checkpoints. So the state that had just produced NaN would be reused by the next
request carrying the same prefix — and a retry is exactly what an agent client
does with a 500. Either it aborts again and the agent spins, or it does not and
the answer is quietly degraded. The second is worse, because nothing reports it.

The handler now erases that slot's state with the same sequence the `SLOT_ERASE`
task uses: `llama_kv_cache_seq_rm`, `cache_tokens.keep_first(0)`, checkpoints and
data cleared, sampler reset. Cost is one full prefill on the retry.

### 33.4 Verified against a running server, not just compiled

A temporary `IK_TEST_NAN_AT` hook was added to fire the failure path on the Nth
sample, 11 requests were driven through a real server, and the hook was then
removed and the binary checked to confirm nothing of it remained.

    ERR sampling failed, releasing slot   id_task=254 error="...all candidate logits are NaN (dump: ...)"
    ERR task error                        id_task=254
    INFO request ... status=500
    ERR NaN logits -- dropped this slot's cached state, the next request will
        prefill from scratch   id_slot=0 cache_tokens_erased=45 checkpoints_erased=0
    INFO slot released                    n_past=0 n_cache_tokens=0

`n_past=0, n_cache_tokens=0` is the proof the cache actually went. The next
request then shows `cache_size = 0` and `kv cache rm p0=0` — a prefill from
scratch, not from the poisoned state. Answers after the failure were correct
("Paris", "42"). Tally: 10 × 200, 1 × 500, process alive.

    grep -a 'NaN logits' logs/server-*.log

One thing a patch cannot repair: if this fires mid-generation with
`stream: true`, tokens already sent stay sent, so the client gets a truncated
answer followed by an error.

The first attempt at this test measured nothing — at `temperature 0` the sampler
takes the greedy branch and never reaches `llama_sample_token_with_rng`, so the
hook never fired and four requests all returned 200. The real aborts all come
through the temperature branch.

### 33.5 It fired in production the same morning, twice

Two aborts during three hours of ordinary Hermes work, 2026-08-20 at 09:54 and
10:08 (crash11, crash12 — same 40-token signature as the other ten). **The server
survived both.**

| | 09:54 | 10:08 |
|---|---:|---:|
| `cache_tokens_erased` | 44 872 | 29 556 |
| `checkpoints_erased` | 32 | 25 |
| next request | `cache_size = 0`, 44 864 tok at 1648 tok/s, **200** | `cache_size = 0`, 29 558 tok at 1723 tok/s, **200** |

So the retry re-prefilled from scratch at full speed and returned a correct
answer. That is the whole design working end to end on traffic nobody staged:
one 500, one re-prefill, no restart, no lost session. Before the patch this
morning would have been two dead servers.

### 33.6 And it refutes the VRAM hypothesis

`tools/vramwatch.sh` was sampling every second across both. Free VRAM at the
abort second: **727 MiB**, against 705–729 MiB over 1694 samples spanning the
whole session. Flat. No dip, no exhaustion, and `nvidia-smi --query-compute-apps`
shows llama-server alone on the card the entire time — no squatter.

Combined with the absence of any `out of memory` / `cuMemCreate` / `CUDA error`
line in all twelve crash logs, and with the fact that the run holding the LARGEST
headroom (4829 MiB) aborted too, memory pressure is out.

### 33.7 What this is not

It does not fix, diagnose or hide the NaN. §32 stands and TODO 9 stays open. It
converts "the server dies every few hours" into "one request returned a 500 and
the next one prefilled from scratch", which is worth having whether or not the
cause is ever found, and prejudges nothing about what the cause is.

---

## 34. DDR5-6400 → 6667: nothing measurable, and a lesson about the noise floor (2026-08-20)

Fourth memory configuration, same 4 × 64 GB kit as §26 pushed from 6400 to
6667 MT/s. Same tool, same profile, same depths, `-r 2`. Two full runs were taken
because the first disagreed with §26 and the disagreement turned out to be the
interesting part.

### 34.1 Prefill

| depth | 6400 (§26) | 6667 +gdb | 6667 −gdb | gdb effect | 6667 vs 6400 |
|---:|---:|---:|---:|---:|---:|
| 4k | 1354.8 | 1354.3 | **1366.3** | +0.9 % | +0.8 % |
| 16k | 1888.0 | 1899.3 | **1901.0** | +0.1 % | +0.7 % |
| 32k | 1802.1 | 1812.9 | **1813.5** | −0.0 % | +0.6 % |
| 128k | 1335.1 | 1340.2 | **1339.8** | −0.0 % | +0.4 % |

+0.6 % on average, and unusually tight: every depth lands between +0.4 and
+0.8 %. Small enough to be nothing, consistent enough to be worth stating as
"not negative".

### 34.2 Generation

| depth | 6400 (§26) | 6667 +gdb | 6667 −gdb | gdb effect | 6667 vs 6400 |
|---:|---:|---:|---:|---:|---:|
| 4k | 20.05 | 19.58 | **19.97** | +2.0 % | −0.4 % |
| 16k | 19.84 | 19.35 | **19.46** | +0.6 % | −1.9 % |
| 32k | 19.18 | 18.77 | **18.59** | **−1.0 %** | −3.1 % |
| 128k | 17.14 | 16.77 | **16.97** | +1.2 % | −1.0 % |

−1.6 % on average with a −0.4 to −3.1 % spread. That spread is the finding, not
the average.

### 34.3 Two things this run corrected

**gdb is not costing anything.** `serve.sh` runs the server under gdb since §33
and the comment there says the overhead is expected-but-unmeasured, so it was the
first suspect. It is not: the effect changes sign — at 32k the run WITHOUT gdb is
1.0 % slower — and on prefill it is +0.2 % across the board. `IK_GDB=1` stays on
by default, now with a measurement behind it rather than an assumption.

**The reported spread understates the real noise floor.** Two runs of the same
configuration, same binary, half an hour apart, differ by up to 2 % on
generation. `depthbench` reported within-run spreads of 0.5–1.1 % for those same
points. So **between-run variance is larger than the within-run figure the tool
prints**, and §16's ~2 % noise floor applies to *sessions*, not repeats.

That matters retroactively: comparing one fresh run against a number stored days
earlier — which is what §25, §26 and this section all do — cannot resolve
anything under a few percent. The first read of this run called −2.3 % across
four depths "too regular to be noise". The control run showed it was.

Prefill does not have this problem: ±0.2 % between runs against generation's
±1.4 %. It is bulk GPU work; generation is memory-bound and shares the machine
with everything else.

### 34.4 The answer

**The overclock bought nothing that can be measured here.** Prefill +0.6 %,
generation inside the noise. Consistent with §26 finding zero for 6267 → 6400.

§25's model — generation tracks bandwidth alone — predicts +1.3 % of generation
for 6667's +4.2 % of bandwidth. That is below what this method resolves, so the
model is neither confirmed nor contradicted.

One hypothesis worth recording, untested: §25's +5.5 % came from a **two-DIMM**
kit at 7400. 6267, 6400 and 6667 are all four-DIMM, and a memory controller runs
looser at 2 DIMMs per channel. If that is what caps the four-DIMM configurations,
MT/s alone would not predict across the two, and §26.1's single-variable law
would hold only within a DIMM count. Settling it needs 2 vs 4 DIMMs at the same
MT/s, which needs hardware not on hand.

---

## 35. ~~The abort is an f16 overflow in the K·Q GEMM~~ — MEASURED AND WRONG (2026-08-20)

A confident, well-supported hypothesis, killed in ten minutes by measuring the
one number it depended on. Kept in full because the reasoning is worth reading
and the mistake is worth not repeating.

### 35.1 The measurement that ended it

The claim below was that the raw, unscaled `K·Q` dot products overflow f16
(65504) in `dsa_attn.cu` line 347. `kq16` is a `half` buffer, so any overflow is
already `inf` by the time it can be read — the GEMM was therefore made to write
f32, the magnitudes measured, and a kernel converted down for the rest:

    max_abs = 366.4    (f16 ceiling 65504, headroom 179x)
    over    = 0        of 5 905 580 032 values

**Two orders of magnitude below the ceiling, across nearly six billion values,
covering both short prompts and a 6000-line one.** Nothing there overflows. The
hypothesis is dead, and both the diagnosis and the patch built on it are
withdrawn.

(The probe run ended in a CUDA error — `cudaStreamSynchronize` and
`cudaMemcpyFromSymbol` are not allowed during CUDA graph capture. That is a bug
in the instrumentation, not a finding; the data it collected first is valid.)

### 35.2 Why it was worth measuring rather than soaking

The plan was an 11-hour soak on real traffic. That would have "confirmed" this by
absence of aborts, and been wrong — exactly the §24 failure again, one section
after writing that §24 was a standing reminder not to repeat it. The measurement
cost one build and ten minutes and answered in the other direction.

**Generalising: prefer measuring the quantity a hypothesis depends on over
waiting to see whether the symptom returns.** Absence-of-symptom evidence is slow
and confirms whatever you already believe.

### 35.3 What was right, and still stands

Two findings from the comparison survive, independent of the overflow claim.

**Mainline llama.cpp has no such abort in its code at all.** `Failed to sample
token`, the dump and the `GGML_ABORT` are ik_llama additions:

| | ik_llama | mainline |
|---|---|---|
| pick a token | cumulative array + `upper_bound` | `std::discrete_distribution` |
| given all-NaN logits | finds nothing → **aborts, dumps** | returns an index → **carries on silently** |

So mainline not crashing is not evidence that mainline does not produce NaN. It
is evidence that it would not say so. This matters for anyone treating it as the
reference.

**#2311 did leave one GEMM in f16 accumulation** — line 347, `K·Q`, against 360
and 367 which it converted. That remains true. It is simply not where the NaN
comes from, since the values there have 179x of headroom.

### 35.4 The original reasoning, for the record

§32 left the NaN abort open with the cause unknown. Comparing against mainline
llama.cpp — which Matt runs in `multi-gpu-llm-toolkit` and which does not show
this — found a mechanism that fits every observation.

#### The premise, as first stated

"Mainline does not have this problem" turns out to mean something weaker than it
sounds. **Mainline has no such abort in its code at all.** The `Failed to sample
token` message, the `probabilities.txt` dump and the `GGML_ABORT` are ik_llama
additions. The two sample differently:

| | ik_llama | mainline |
|---|---|---|
| pick a token | manual cumulative array + `upper_bound` | `std::discrete_distribution` |
| given all-NaN logits | finds nothing → **aborts, dumps** | returns some index → **carries on silently** |

So mainline not crashing is not evidence that mainline does not produce NaN. It
is evidence that it would not tell you. Worth remembering before treating it as
the reference implementation.

#### The mechanism (refuted by 35.1)

DSA attention in `ggml-cuda/dsa_attn.cu` runs two GEMMs:

| line | computes | accumulation | fixed by #2311? |
|---|---|---|---|
| 347 | `K·Q` → scores | **f16** | **no** |
| 360, 367 | `V·softmax(KQ)` | f32 | yes |

`cublasHgemmStridedBatched` is f16 in, **f16 accumulate**, f16 out. The reduction
runs over `n_embd_head_k = 512`. And `scale` is not applied there — the softmax
below does it, at `scale*__half2float(x[ix])`. So the **raw, unscaled** dot
product is what lands in an accumulator that saturates at 65504.

One overflow poisons everything downstream:

    kq16 = inf  ->  val = scale*inf = inf  ->  max_val = inf
                ->  expf(val - max_val) = expf(inf - inf) = nan

which gives **all** logits NaN, not some — the signature in all twelve dumps.

It also explains the rest of §32: why #2311 did not help (it fixed the *other*
GEMM), why mainline does not hit it (WMMA fragment with a `float` accumulator in
`ggml-cuda/lightning-indexer.cu`), why it tracks prefilled volume, and why every
configuration bisect came back negative — this is in attention, which does not
care where the experts are computed.

#### The patch, and its cost (withdrawn)

Same change #2311 made to the other GEMM: `cublasGemmStridedBatchedEx` with
`CUDA_R_32F` compute. (The patch file was removed when the hypothesis was refuted; the change is the same one #2311 made to the other GEMM.)

| depth | before | with f32 K·Q | prefill | generation |
|---:|---|---|---:|---:|
| 4k | 1363.1 / 19.60 | 1371.5 / 20.03 | +0.6 % | +2.2 % |
| 16k | 1908.9 / 19.42 | 1926.6 / 19.79 | +0.9 % | +1.9 % |
| 32k | 1825.8 / 18.66 | 1830.1 / 18.70 | +0.2 % | +0.2 % |
| 128k | 1349.1 / 16.95 | 1352.3 / 17.14 | +0.2 % | +1.1 % |

Every depth neutral-to-positive. **Read this as "costs nothing", not "is
faster"**: generation's between-run noise is ±1.4 % (§34.3) and the +1.36 %
average sits inside it. Prefill's +0.5 % is marginally above its ±0.2 %, which is
suggestive at best. Plausibly `CUBLAS_GEMM_DEFAULT_TENSOR_OP` asks for tensor
cores explicitly where `Hgemm` left the choice to cuBLAS.

#### A first correction: the patch was narrower than claimed

`kq16` is `ggml_cuda_pool_alloc<half>`: the GEMM's **output buffer is f16
whatever the compute type**. `CUDA_R_32F` changes how partial sums are
accumulated; the result is still converted to f16 on the way out.

| | accumulate | store | protects against |
|---|---|---|---|
| original `Hgemm` | f16 | f16 | nothing |
| the patch here | f32 | **f16** | overflow in *partial sums* only |
| an actual fix | f32 | f32 | both |

So if the full dot product exceeds 65504, `inf` still appears. The patch only
helps where a running sum transiently leaves f16 range while the final value
would fit — real with split-k tensor-core accumulation, but much narrower than
§35.2 suggests on its own. Written down because it was stated too broadly first
and the code says otherwise.

Which makes the measurement below the thing that decides it, rather than the
soak.

#### What was not established — and turned out to be false

**No overflow has been caught in the act.** What is established is that the
mechanism is present, unguarded, and consistent with every observation. The proof
would be hours of the traffic that used to abort, coming back clean — which is
now running, and which §24 is a standing reminder not to call early.

§24 made exactly this mistake once: a correct-looking mechanism, a clean
synthetic run, and a conclusion that did not survive real traffic. The difference
this time is that the mechanism is a specific unfixed line rather than an
inference about a commit — but that was true in §24 as well.

---

## 36. The NaN surfaces at `FLASH_ATTN_EXT` layer 0 — and arrives from the KV cache (2026-08-21)

Thirteen aborts had been chased through configuration bisects, an upstream commit
audit and one refuted overflow hypothesis (§35), all without ever seeing where the
NaN came from — because the abort is reported by the sampler, and `llama_decode`
has long returned by then.

A probe in `ggml_cuda_compute_forward` now checks each node's output and records
the first that contains a NaN, reading the flag back after graph capture ends
(`IK_NAN_CHECK=1`). Three aborts later it answers.

### 36.1 The accounting is exact

486 hits across the run, and they partition perfectly:

* **3 × 162**, three identical blocks
* **zero hits before the first `FLASH_ATTN_EXT`** — there is no second origin
* 3 aborts, 3 origins, one-to-one

Each block:

    node 14   FLASH_ATTN_EXT   fattn-0        <- origin
    node  0   MUL_MAT          qr-1
    node  0   MUL_MAT          qr-2
     ...                                       161 nodes, layer by layer
    node  0   MUL_MULTI_ADD    ffn_moe_out-42
              -> all logits NaN -> abort

So one poisoned tensor in layer 0 propagates through all 43 layers to the logits.
That is why the dumps show **all** candidates NaN and never some.

### 36.2 It is not where anyone was looking

Every hypothesis so far pointed at DSA (`dsa_attn.cu`) — §24's f16 accumulation,
§35's K·Q overflow, the sm_120 branch divergence of #2317. The origin is ordinary
flash attention, `GGML_OP_FLASH_ATTN_EXT`, in the *first* layer.

    op     : FLASH_ATTN_EXT
    tensor : fattn-0                         f32   512 x 64 x 4223
    src[0] : q_rope-0                        f32   512 x 4223 x 64
    src[1] : raw_k-0                         f16   512 x 4352 x 1
    src[2] : raw_k-0                         f16   512 x 4352 x 1
    src[3] : dsv4_raw_mask_padded-0          f16   4352 x 4224 x 1
    src[4] : blk.0.attn_sinks.weight         f32   64

### 36.3 What this does NOT yet establish

**Whether that op produces the NaN or merely reads it.** The probe checks node
*outputs*. `raw_k-0` comes from the KV cache, which no node in this graph
produces, so it was never in view — an op propagating someone else's NaN looks
identical to the op that made it.

The suspicion is concrete rather than theoretical: `SET_ROWS csa_k_write-2`, a KV
cache *write*, appears among the propagation hits. And a poisoned cache would fit
what §33 already showed — that dropping the slot's cache after an abort stopped
the immediate recurrence.

The probe has been extended to check the inputs of the offending node and print
`(clean)` or `<== ALREADY NaN` against each. The next abort answers it.

### 36.4 Answered overnight: it reads the NaN, it does not make it

§36.3 asked whether that op produces the NaN or merely reads it. The probe was
extended to check the inputs of the offending node, and the fourteenth abort
answered on the first try:

    op     : FLASH_ATTN_EXT   fattn-0        f32   512 x 64 x 1378
    src[0] : q_rope-0                        f32   (clean)
    src[1] : raw_k-0                         f16   <== ALREADY NaN
    src[2] : raw_k-0                         f16   <== ALREADY NaN
    src[3] : dsv4_raw_mask_padded-0          f16   <== ALREADY NaN
    src[4] : blk.0.attn_sinks.weight         f32   (clean)

**So the heading this section was first given was wrong.** Flash attention is
where the poison becomes visible — the first node in the graph that reads from
the cache — not where it comes from. The query and the model weight are clean, so
nothing is wrong with the input to the model.

### 36.5 And it does not appear to be computed at all

The stronger result is what the probe did **not** see. Across 4.6 h and 33
requests it logged exactly one block of 162 hits, the one at the abort. No node
output was ever NaN before that. Yet the cache was already poisoned when that
graph began.

If something had *computed* a NaN, the probe would have caught it as a node
output. It did not. That points at memory that was never written rather than a
value that was miscalculated — and both poisoned tensors are padded past the end
of the real data:

    prompt        1378 tokens
    raw_k-0       512 x 1536      158 elements beyond
    mask_padded   1536 x 1392     the name says it

The probe now reports the flat index, row and column of the first NaN and how
many there are. If they all sit past the logical end, uninitialised padding is
established. That is the next abort's job.

Recorded as a hypothesis, not a finding — §35 is one section away as a reminder
of what a convincing mechanism is worth before it is measured.

### 36.6 Measured: it is NOT uninitialised padding

The fifteenth abort answered §36.5 on the first try, and the answer is no.

    prompt 2414 tokens, raw_k-0 is 512 x 2560

    raw_k-0   first NaN at flat 36 (row 0, col 36)   93 119 of 1 310 720
    mask      first NaN at flat 86 (row 0, col 86)   95 671 of 6 184 960

`row = flat/ne0` with `ne0 = 512`, the head dimension. So the first NaN sits in
**token 0's key vector, at dimension 36** — the very start of real data, as far
from the padding as it is possible to be.

The counts say the same thing independently:

| | elements | share |
|---|---:|---:|
| whole tensor | 1 310 720 | |
| real data | 1 235 968 | 94.3 % |
| padding | 74 752 | 5.7 % |
| **NaN** | **93 119** | **7.1 %** |

**There is more NaN than there is padding** — 18 367 elements more. Even a wholly
corrupt pad region could not account for it. The poison is scattered through the
real data, about 7 % of the tensor.

Static analysis had already made it unlikely, and is recorded because it cost
nothing and narrowed the field before the abort arrived:

* `dsv4_pad_raw_mask_to` fills with `-INFINITY` — and its `Oops, padding mask`
  printf appears **zero** times in any log, so it never ran at all.
* `dsv4_pad_mask_tokens` also fills `-INFINITY`, which is inf, not NaN.
* `dsv4_pad_raw_k_to` zeroes its pad via `ggml_scale(row0, 0.0f)`. That would
  propagate rather than clear a NaN or inf in row 0 — but it needs row 0 to be
  bad already, which is the question, not the answer.

### 36.7 What that leaves

The KV cache for layer 0 is wholesale poisoned before the graph that reads it
begins, and no node output was NaN earlier in that graph. So it is persistent
state that went bad, not arithmetic in this forward pass.

One observation, deliberately marked as weak. The abort followed a checkpoint
creation at `pos_max = 75190` and a `kv cache rm p0=75191`. §32.4 dismissed
checkpoint-adjacency as vacuous, and on frequency alone it still is — 151
creations in this run. What has changed is that the poison is now known to live
**in the KV cache**, which is exactly what checkpoints read and write. That moves
it from coincidence to a mechanism worth testing, and no further.

Against it, twice over. There was **no restore** anywhere near the abort — 13 in
the whole run, none in the preceding 60 lines. And the invalidation path, which
the count of 132 made look like the remaining candidate, does not touch the KV
cache at all: `server-context.cpp:3736` only erases entries from
`slot.server_cached_prompt.checkpoints`, a host-side vector of saved copies.

So of the three checkpoint operations, creation reads the cache, invalidation
drops host copies, and restore writes the cache but did not run. **None of them
is a route by which the cache gets NaN in it.** The checkpoint lead is close to
dead on the same evidence that raised it; it is left written down only so the
next person does not re-raise it.

§35 remains one section away. This is a lead, not a finding.

### 36.8 `raw_k` is a VIEW of the cache, not a computation — which explains §36.5

§36.5 made much of the probe never seeing a node output go NaN, and read that as
evidence for memory never written. The real explanation is duller and does not
need CUDA graphs. In `dsv4_raw_get_k`:

```c
if (n_stream == 1 && lctx->kv_self.n == raw_k_read_idxs->ne[0]) {
    return ggml_view_3d(ctx, cache, n_embd_head, n_head_kv, n_kv, ...);
}
```

On this path `raw_k` is **a `ggml_view_3d` straight into `kv_self.k_l[il]`**. A
view is not a node; nothing is computed, so there is no node output for the probe
to flag. The NaN is physically in the KV cache and `raw_k` is a window onto it.

That resolves the paradox without the graph-replay hypothesis. It does not
resolve where the NaN comes from — it only says the producer was never going to
show up in a scan of node outputs, so §36.5's "not computed at all" was reading
too much into a blind spot.

The other branch of the same function is worth noting for later: it gathers rows
with `ggml_get_rows(cache_2d, raw_k_read_idxs)` over `n_kv = max(256,
GGML_PAD(n_kv_visible, 256))` rows — more rows than are visible, using whatever
indices sit in `raw_k_read_idxs` past the end. That path was **not** the one
taken here, and is recorded as somewhere to look, not as a claim.

### 36.9 The first KV scan was measuring nothing — a correction

The scan added to `llama_decode_internal` reported a clean cache through an abort
in which attention demonstrably read NaN out of that very tensor. Both cannot be
true, and the scan was wrong: it covered `max(kv.head, kv.used)` cells, but
DeepSeek-V4 compacts and windows rows (`kv.row_count`, `size_swa`, `head_swa`),
so `head`/`used` does not describe layer 0's occupancy at all.

It now scans the whole tensor, counts NaN inside and beyond the live rows
separately, and prints a heartbeat:

    IK_KV_SCAN: alive, decode #1, scanned 67108864 elems (131072 live rows), 0 NaN (0 live)

The heartbeat exists because the first version could not distinguish *found
nothing* from *never ran*, and that distinction cost an abort's worth of
evidence. Any probe that reports only on failure needs one.

Cost: 67 M f16 reads per decode. Acceptable while hunting, not otherwise.

### 36.10 Why this took thirteen aborts

The three configuration variables that were bisected (`-rtr`, `-ub`, placement)
could never have found it: none of them touches layer-0 attention. §24's lesson
was to look outside the repository; this section's is narrower and more useful —
**instrument the failing path rather than bisecting the configuration around it.**
The probe cost one build and answered on the first abort it saw.

---

## 37. What the NaN hunt costs, and CUDA graphs are not it (2026-08-21)

The diagnostic build carries three things the shipped one does not: the node NaN
probe (§36.1), the KV cache scan (§36.9) and `GGML_CUDA_DISABLE_GRAPHS=1`. Two
arms, same tool and depths, `-r 2`, against the reference run from earlier the
same day.

| depth | reference | probes + scan, graphs **off** | probes + scan, graphs **on** |
|---:|---|---|---|
| 4k | 1363.1 / 19.60 | 1310.4 / **13.17** | 1285.2 / 13.42 |
| 32k | 1825.8 / 18.66 | 1735.2 / **12.79** | 1700.0 / 12.98 |

Against the reference: prefill −3.9 % and −5.0 %, generation **−32.8 % and
−31.5 %**.

### 37.1 CUDA graphs are worth about 2 %, not 30

| depth | graphs on vs off |
|---|---|
| 4k | prefill −1.9 %, generation **+1.9 %** |
| 32k | prefill −2.0 %, generation **+1.5 %** |

Enabling graphs buys 1.5–2 % of generation and makes prefill marginally worse,
which is inside the 3.2 % within-run spread at 4k. So turning them off — done to
remove the probe's blind spot during replay — costs about two percent of
generation, not the thirty that the diagnostic build loses overall.

That also disposes of the idea that graph replay is expensive enough to matter
here either way.

### 37.2 The scan dominates, and the asymmetry says why

The NaN probe alone was measured at −5.3 % prefill and −7.0 % generation when it
was added. Graphs account for ~2 %. The remaining ~25 points of generation are
the KV scan.

The split between prefill and generation is the tell: the scan runs **once per
decode**, reading 67 M f16 values. Prefill amortises that over thousands of
tokens in a batch; generation pays it in full for every single token. Hence 5 %
against 32 %.

Two cheaper variants exist — scan every Nth decode, or scan only during prefill,
since the poisoning shows up while a prompt is being processed. Neither was
adopted: Matt's call was to keep the full scan while the cause is open, and take
the generation hit.

---

## 38. Two aborts, two layers, one suspect: `CUDA0#raw_k-N` (2026-08-21)

The corrected scan (§36.9) ran through three aborts and produced a contradiction
worth more than either half on its own.

### 38.1 The host cache is clean; the device copy is not

    IK_KV_SCAN   layer-0 K cache, whole tensor, every decode:  0 NaN, ever
    IK_NAN_CHECK 348 blocks across the same run

The scan is not silently broken — its heartbeat prints every 500 decodes and
reports 67 108 864 elements scanned. So the host-resident KV cache was clean
entering the decode in which attention read NaN.

### 38.2 The origin of each abort names the same kind of tensor

Only the **first** block of each burst is the origin; the rest is propagation,
and counting all of them is what made an earlier pass conclude that poisoned
tensors are mostly not scheduler copies. Restricted to origins:

| burst | origin op | poisoned input | clean inputs |
|---|---|---|---|
| 1 | `FLASH_ATTN_EXT`, layer 0 | `CUDA0#raw_k-0 (view) (permuted)#0` | `q_rope-0`, `blk.0.attn_sinks.weight` |
| 2 | `CONCAT`, layer 4 | `CUDA0#raw_k-4 (view)#0` | `csa_k-4` |

Different op, different layer, same shape of culprit: **`CUDA0#raw_k-N (view)#N`**,
which the `%s#%s#%d` naming in `ggml-backend.cpp` identifies as a scheduler-made
copy of a tensor into the CUDA backend. Everything computed natively on the
device is clean.

### 38.3 Where that points, and what is not yet shown

Two facts sit next to each other:

* `dsv4_raw_get_k` returns `ggml_view_3d(cache, n_embd_head, n_head_kv, n_kv)`
  with `n_kv = max(256, GGML_PAD(n_kv_visible, 256))` — the view **deliberately
  extends past the live rows**, to give attention a stable 256-row-aligned shape.
* the CUDA copy is a flat `cudaMemcpyAsync(dst->data, src->data,
  ggml_nbytes(dst), ...)` (`ggml-cuda.cu:4483`) — bytes, no strides.

So the copy faithfully brings whatever occupies those beyond-live cache rows onto
the device. If they were never written, that is uninitialised memory.

**What contradicts the simple version:** the host scan covers the whole of
`k_l[0]`, beyond-live rows included, and never saw a NaN — while burst 1 is
layer 0. Either the poison is not in `k_l[0]` at all and `raw_k-0` resolves to a
different buffer, or it appears between the scan at the end of one decode and the
copy in the next.

Recorded as the state of the evidence, not as a mechanism. The next step is to
check the host tensor **at the moment the probe finds the device copy poisoned**,
rather than at the end of the previous decode — a direct comparison instead of an
inference across two points in time. §35 is the standing reminder of what happens
when a plausible mechanism is written up before that comparison is made.

### 38.4 Answered: the host cache is clean at the moment before the graph

§38.3 listed three ways the contradiction could resolve. The next abort picked
the third, and cleanly:

    IK_KV_SCAN [pre]  poisoned: 0
    IK_KV_SCAN [post] poisoned: 0
    IK_NAN_CHECK    : 162 blocks
    sampler abort   : 1

The `[pre]` scan runs after `ggml_backend_sched_alloc_graph` and immediately
before the graph is computed, so the gap in which the server can restore a
checkpoint, run `kv_cache_seq_rm` or defrag is now covered. The host-resident
`k_l[0]` was clean going in, clean coming out, and the device copy was NaN in
between.

Two ways of being wrong were checked and neither holds:

* **Wrong tensor.** All three branches in `build_deepseek4.cpp` build `raw_k`
  from `kv_self.k_l[il]`, including the `raw_compacted` and `read_idxs` paths.
  The scan is watching what attention reads.
* **A stale second copy.** `GGML_SCHED_MAX_COPIES` is 1, so `sched->n_copies` is
  1 whatever `pipeline_parallel` says. There is no older device buffer to read by
  mistake.

That leaves the transfer itself. `ggml-backend.cpp:2133` copies split inputs with
a flat `ggml_backend_tensor_set_async(..., input->data, 0, ggml_nbytes(input))` —
and `input` here is `raw_k-N (view) (permuted)`, not a contiguous tensor.

### 38.5 The probe that decides it

`IK_COPY_CHECK=1` scans the **exact bytes about to be sent, at the moment of
sending**, on the host side of that call. It ends the practice of comparing a
host scan at one point in time against a device state at another.

| outcome | conclusion |
|---|---|
| `HOST source already NaN` fires | the poison is in the cache and the whole-tensor scan is missing it — wrong offset or range |
| silent, while `IK_NAN_CHECK` still fires | the poison is introduced **by the host→device transfer** |

Verified not to fire on healthy traffic before being left to run.

### 38.6 Four instrumented copy paths, none of them carries `raw_k`

`IK_COPY_CHECK` was placed, in turn, on:

1. the `only_active_experts` + `BUFFER_USAGE_WEIGHTS` branch in
   `ggml_backend_sched_copy_inputs` — expert streaming, not the KV cache;
2. the general input-copy branch in the same function;
3. `ggml_backend_tensor_copy`;
4. `ggml_backend_tensor_copy_async` and `ggml_backend_tensor_set_async`.

Every one reports the same after a 35 007-token prefill:

    IK_COPY_CHECK: alive, 1 copies seen, 0 host-f16 scanned

So `CUDA0#raw_k-0` is not filled through any of them. The mechanism assumed in
§38.3 — a flat byte copy of a permuted view — is not what happens.

**The heartbeat is the only reason this is known.** Without it, four separate runs
would have been read as "the host source is clean", and the fourth would have
been written up. What the counter actually says is "this path is not used". §36.9
recorded that lesson after the first KV scan; it took four more misses before it
was applied without being reminded.

### 38.7 The possibility that fits everything

If nothing ever copies into `CUDA0#raw_k-N`, it holds whatever the allocator left
in that device memory. That would account for the whole picture at once: a clean
host cache, a poisoned device tensor, no node that computes it, and no copy that
can be caught in the act.

Cheap to test: log the names in `split->inputs[]` and see whether `raw_k-N` is
among them at all. If it is not, the tensor is allocated and never written.

Not tested yet — recorded as the next step, not as a finding.

### 38.8 The contradiction was mine, three times over

§38.6 reported four instrumented copy paths seeing no traffic, and §38.7 built a
hypothesis on it. Both were wrong, and for the same reason each time: **a probe
that had not run was read as a probe reporting zero.**

| # | what was read | what it meant |
|---|---|---|
| 1 | KV scan: no poisoning | scanned `max(head, used)` cells; DeepSeek-V4 compacts and windows rows, so that range describes nothing (§36.9) |
| 2 | branch tags: only `try_async` ever taken | two of three `str.replace` calls silently matched nothing — they had no `assert`, unlike every other edit |
| 3 | copy check: "1 copies seen" | the *first* heartbeat, printed at call #1; the next was due at 50 000 and short tests never reached it |

With the tags actually present, every input takes `try_async` **and then falls
back**: 253 distinct names through `tensor_copy`, 23 through `copy_early`. That
matches the static reading — `ggml_backend_buffer_is_cuda` compares against the
*device* buffer's `get_name`, and the KV cache is `CUDA_Host`, so
`cpy_tensor_async` returns false for all of them.

With a readable heartbeat, the copy check turns out to have been working all
along:

    IK_COPY_CHECK: alive, 20000 copies seen, 10646 host-f16 scanned

### 38.9 What is now actually established

`raw_k-N` reaches the device through:

    ggml_backend_tensor_copy(src, dst)
      -> ggml_backend_buffer_is_host(src->buffer)          // true, CUDA_Host
      -> ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src))

a flat byte copy of a **permuted view**, and the check on it reports nothing over
10 646 host-f16 copies of healthy traffic.

So the decisive comparison is finally armed rather than assumed: at the next
abort, either the host bytes are dirty at the moment of the copy — and the
whole-tensor scan is missing them — or they are clean and the poison is
introduced in transit.

**The lesson, since it cost more than any hypothesis here:** a probe that only
reports on failure is indistinguishable from a probe that never ran. Every one
needs a liveness signal, and every edit that inserts one needs to be verified
against the file rather than the script's own success message.

---

## 39. Found: the host KV cache is poisoned, in layers 3–42 (2026-08-22)

The seventeenth abort answered §38.9 on the first try, and the answer is the
first branch: **the host bytes are already NaN at the moment of the copy.**

    IK_COPY_CHECK [tensor_copy]: HOST source already NaN: raw_k-4 (view) -> CUDA0#raw_k-4 (view)#0,
                                 530432 of 655360 f16, first at 0

157 such reports in one abort, across **40 distinct layers**, and the shape of it
is what makes it a finding rather than another observation:

| layer | NaN | share | first at |
|---:|---:|---:|---:|
| 0, 1, 2 | — | — | never reported |
| 3 | 12 288 of 655 360 | 1.9 % | element 1024 (row 2) |
| 4 … 42 | **530 432 of 655 360** | **80.9 %** | element 0 (row 0) |

Byte-identical counts across thirty-nine layers. 655 360 f16 is 1280 rows of 512;
1036 rows are NaN and 244 are clean, the same split every time. Uniformity like
that is memory that was never written, not arithmetic that went wrong.

### 39.1 Why every previous scan missed it

`IK_KV_SCAN` reported a clean cache through four aborts. It was not broken and it
was not the wrong range this time — **it only ever scanned `k_l[0]`**, and layer 0
is one of the three layers that is never poisoned. `raw_k-0` does not appear in
any of the 157 reports.

So §38's contradiction — clean host, poisoned device — was never real. Both
probes were telling the truth about different layers.

### 39.2 The one correlation, stated carefully

The first poisoned copy follows a `restored context checkpoint` (task 57103,
`pos_max = 33124`) by 108 log lines. Restores are rare enough for that to be worth
noting: 15 in the whole run against 452 creations.

**But 12 of those 15 restores produced no poisoning at all.** So a restore is not
sufficient, and this is a lead rather than a mechanism. §32.4 dismissed
checkpoint-adjacency as vacuous on frequency grounds and was right to; what has
changed is that the poison is now known to be host-resident KV cache, which is
exactly what a restore writes, and that a checkpoint is ~872 MiB against a 5504
MiB cache — so a restore cannot be rewriting all of it.

### 39.3 What to do next

Scan **all** layers, not layer 0, and scan on both sides of a checkpoint restore.
That turns "a restore preceded it" into "this restore left these rows unwritten",
or kills the idea outright. The probe already exists; it needs its loop widened
and a hook either side of `apply_checkp`.

---

## 40. The causal chain, end to end — and one step left (2026-08-22)

The eighteenth abort, with the scan moved to layer 4, gives the whole sequence.

### 40.1 The KV poisoning is a consequence, and the arithmetic is exact

    IK_KV_SCAN [post] layer-4: CLEAN -> POISONED
      decode #5800: batch of 3426 tokens, kv head=56648 used=56648
      first NaN at cell 53222, 1 754 112 NaN (all inside live rows)

| | |
|---|---:|
| `kv.head` before the batch | 56648 − 3426 = **53222** |
| first poisoned cell | **53222** |
| batch size | **3426** tokens |
| poisoned cells | 1 754 112 / 512 = **3426** |

The NaN starts exactly where the batch started and there are exactly as many
poisoned cells as there were tokens. **This is not uninitialised memory** — §36.5
and §39 both leaned that way. The K values *computed for that batch* were NaN and
were duly stored.

### 40.2 Layer 0 is the origin, and it fails differently

| probe | side | layers reported |
|---|---|---|
| `IK_COPY_CHECK` | host, at copy time | 3 … 42 — **never 0** |
| `IK_NAN_CHECK` | device, as graph input | `CUDA0#raw_k-0` already NaN |

For layers 3–42 the host cache is poisoned, which §40.1 explains as the stored
result. For **layer 0 the host is clean and the device copy is NaN** — a
different failure, and the only one that is not accounted for by something
upstream of it.

Chronology confirms the direction. First event of the run: the NaN probe at layer
0. Then the copy check on layers 3+. Then the layer-4 cache flips to poisoned.

### 40.3 The chain

    CUDA0#raw_k-0 is NaN on the device, host source clean   <-- unexplained
      -> fattn-0 reads it, output NaN
      -> propagates through all 43 layers
      -> K computed for the batch is NaN, stored to host cache (layers 3-42)
      -> next decode reads poisoned host, aborts at the sampler

Every step but the first is now measured rather than argued.

### 40.4 What is NOT the explanation

`raw_k` alternates permuted / not — 0, 1, 3, 5, 7 permuted, 2, 4, 6 not — so
"permuted views are copied with a flat memcpy" does not separate the affected
layers from the unaffected ones and is not the mechanism, however plausible it
looked. Layers 0, 1 and 2 are the ones never poisoned in host memory, and what
distinguishes them is `compress_ratios = [0, 0, 4, 128, ...]`: layers 0 and 1 are
uncompressed.

### 40.5 The one remaining question

How does `CUDA0#raw_k-0` become NaN when the host bytes it is copied from are
clean? The next probe reads the **destination** immediately after the copy and
compares it with the source — the same instant, the same bytes, both sides.

### 40.6 For the upstream comment: it only shows up with the cache split across RAM

Worth stating plainly whenever this is reported, because it explains why almost
nobody else would hit it.

| KV cache | serving | prefilled tokens | aborts |
|---|---:|---:|---:|
| host RAM (`-nkvo`) | 74.1 h | 34 253 025 | **22** |
| VRAM | 19.8 h | 3 724 240 | **0** |

At the observed rate, 1.43 aborts would be expected in the VRAM runs and none
happened. That is a ~24 % coincidence, so it is **suggestive, not established** —
roughly three times the volume would be needed to claim it.

The mechanism is the better argument. With the cache in VRAM there is **no
host→device copy of `raw_k` at all**: attention reads it where it already lives.
The poison appears exactly on that boundary — host clean, device copy NaN — so
removing the transfer removes the place the fault occurs.

Which makes this a bug that only bites configurations running a model too large
for the card, with the KV cache in system RAM. Anyone with enough VRAM to hold
the whole cache would never see it, and that is a plausible reason it has not
been reported before.

**And `-nkvo` is not a tuning preference — it is forced.** Worth spelling out in
the report, because it decides whether the affected configuration reads as exotic
or as the only way to run the model at all.

The KV cache itself is small: 5504 MiB against 96 GiB of card. It is not there
because it does not fit. It is there because **the attention scratch follows it**.
With the cache on the GPU, §22.3 measured the CUDA compute buffer going from
3520 MiB to **28 992 MiB** — 8.2x — since the score matrix for
`tokens x kv_positions` is materialised where the cache lives. 86 103 MiB of
weights plus 28 992 MiB of scratch does not fit in 96 GiB, and the model fails to
load at `--n-cpu-moe` 19, 20 and 22 alike.

So: a table five times the size of the notebook it works from. Anyone running a
model larger than their card has no choice but `-nkvo`, which puts them on the
one code path where this bug lives. That is not a corner case for them — it is
the only configuration available.

---

## 41. A candidate cause: the allocator guard that never runs (2026-08-22)

§40.5 left one question: how does `CUDA0#raw_k-0` become NaN when the host bytes
it is copied from are clean and the copy is byte-perfect?

`IK_DST_CHECK` settled the copy itself. Reading the destination back immediately
after the transfer, over 50 reports and 1200 comparisons:

    src_nan=926208  dst_nan=926208  (first 0)  differing=0 of 1048576

Never once `src_nan=0` with `dst_nan>0`, and **never a single differing byte**.
The transfer is faithful. So the poison arrives after it.

### 41.1 The guard

Both places where the scheduler makes a device copy of a split input read:

```c
if (sched->n_copies > 1) {
    ggml_set_input(tensor_copy);
    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
}
```

`GGML_SCHED_MAX_COPIES` defaults to 1, so `sched->n_copies` is 1 and **this guard
never runs**. Its own comment says what that leaves open.

It fits every observation, including the ones that made no sense before: the host
is clean because nothing writes there; the copy is faithful; the device buffer is
NaN anyway; no node computed it, because it is a memory overwrite rather than
arithmetic; and it is intermittent because the allocator's layout depends on the
batch shape.

### 41.2 Turning it on wholesale is not possible

    compute buffer requested: 176 977 MiB   (against 7040)
    cudaMalloc failed: out of memory

25x. Narrowed to `raw_k` alone it costs **+536 MiB** — 7040 → 7576 — and the
model loads, serves, and survives 4k, 32k and 128k without OOM on ~180 MiB of
remaining headroom.

| depth | reference | with the raw_k guard | prefill | generation |
|---:|---|---|---:|---:|
| 4k | 1363.1 / 19.60 | 1336.4 / 19.92 | −2.0 % | +1.6 % |
| 32k | 1825.8 / 18.66 | 1798.3 / 18.94 | −1.5 % | +1.5 % |
| 128k | 1349.1 / 16.95 | 1352.5 / 17.06 | +0.3 % | +0.6 % |

−1.1 % of prefill on average, generation unchanged. `docs/external/local-rawk-alloc-guard.patch`,
switchable at runtime with `IK_RAWK_GUARD=0`.

### 41.3 Worth it even before it is proven

| | prefill lost | other cost |
|---|---:|---|
| with the guard | **1.1 %** | none |
| without | **3.9 %** | one 500 to the client, ~56 s of re-prefill |

The abort discards ~101 000 tokens of cache (measured across eight events:
97 080 to 109 893) at one abort per ~2.6 M prefilled tokens. So the guard is
cheaper than the fault it may prevent — provided it prevents it.

### 41.4 What the overlap check does and does not show

`IK_OVERLAP` walks every tensor with storage after allocation and reports
intersecting byte ranges:

| | compute buffer | `raw_k` overlaps |
|---|---:|---:|
| guard off (upstream) | 7040.03 | **73 423** |
| guard on | 7576.28 | **1** |

That confirms the guard does exactly what is intended. **It does not show a
bug.** Memory reuse is the graph allocator's entire purpose: tensors whose
lifetimes do not overlap are *supposed* to share storage. An intersecting address
range is normal operation, not a fault.

Demonstrating the fault needs a *lifetime* conflict — `raw_k`'s storage handed to
another tensor while `raw_k` is still to be read. Reading `CUDA0#raw_k-0` back
both immediately after the copy and again just before attention consumes it would
show that directly; a difference between the two is the proof.

Recorded as a candidate with strong circumstantial support. §35 is four sections
back and remains the reason for the wording.

### 41.5 The direct proof was attempted and did not arrive

Three probes, each fixing the flaw in the one before, and the honest answer is
that the lifetime conflict was **not** observed.

| attempt | result | why it did not answer |
|---|---|---|
| overlapping address ranges | 73 423 overlaps | sharing storage is what the allocator is *for* |
| same address, content changed | 32 of 1500 | the map was not cleared between graphs |
| same, cleared per graph | 30 of 3000 | reported `q_b-N` — a **different tensor** at a former `raw_k` address, i.e. correct reuse after last use |
| **same tensor, content changed** | **0 of 5500** | this is the right measurement, and it found nothing |

So on healthy traffic no `raw_k` copy is ever overwritten between the copy and
its use. The mechanism §41.1 proposes is not happening routinely.

That leaves two readings, and the evidence does not choose between them:

1. the conflict occurs only in the rare graph shapes that end in an abort — those
   three requests were ~100 k tokens against one abort per ~2.6 M;
2. it does not occur at all, and the guard's 1.1 % buys nothing.

Distinguishing them needs `IK_LIFETIME` running with the guard **off** until an
abort arrives. That is still better than waiting for absence: it would catch the
conflict at the moment it happens, which is positive evidence rather than the
negative kind §35 warns about.

Recorded here rather than quietly dropped because three of these four probes
produced numbers that looked like findings and were not. The pattern — a probe
answering a slightly different question than the one asked — has now cost more
time in this investigation than any wrong hypothesis.

---

## 42. PR #2347 does not stop it (2026-08-23)

`sayap` opened [#2347](https://github.com/ikawrakow/ik_llama.cpp/pull/2347) on the
issue. Applied here on top of `8337e4cd`, with the local `raw_k` guard **off** so
the PR is what is being tested.

    338 509 prefilled tokens, 1.8 h  ->  1 abort
    "failed to sample token: all candidate logits are NaN"

Same signature. That is sooner than the observed mean of one per ~2.6 M tokens,
so it is not a marginal survival — though one event is one event.

Cost, measured the same way as everything else:

| depth | reference | with #2347 | prefill | generation |
|---:|---|---|---:|---:|
| 4k | 1363.1 / 19.60 | 1336.4 / 20.00 | −2.0 % | +2.0 % |
| 32k | 1825.8 / 18.66 | 1773.2 / 18.79 | −2.9 % | +0.7 % |
| 128k | 1349.1 / 16.95 | 1319.6 / 16.93 | −2.2 % | −0.1 % |

−2.3 % of prefill, generation unchanged, no OOM at any depth despite KQ now
holding an f32 buffer.

### 42.1 The stated rationale does not match what is measured here

The PR describes "Q·K logits overflow fp16 (65504), saturating the KQ GEMM output
to inf/NaN". §35.1 measured that quantity directly, by making the GEMM write f32
and reading the magnitudes before the softmax:

    max_abs = 366.4    (f16 ceiling 65504, headroom 179x)
    over    = 0        of 5 905 580 032 values

Two orders of magnitude of headroom over nearly six billion values, covering
short prompts and a 6000-line one. Nothing there overflows on this box.

The PR's second change — `cublasSetStream(ctx.cublas_handle(), ctx.stream())`,
so the GEMMs stop racing kernels on the backend stream — fits the observations
much better (intermittent, host clean while the device copy is NaN, no node
producing it). It did not stop the abort either, but it is the half worth
keeping attention on.

### 42.2 And a note on why our own probes could never have found a race

Every probe here that read device memory back — `IK_DST_CHECK`, `IK_LIFETIME`,
the KV scan — does so through `ggml_backend_tensor_get`, which synchronises. So
does `GGML_CUDA_DISABLE_GRAPHS=1`, which most of these runs used. Both serialise
the streams and would **suppress** a race.

283 000 lifetime verifications found nothing (§41.5). If the cause is a stream
race, that result was never capable of being anything else. Worth recording as a
methodological limit rather than as evidence.

---

## 43. Now testing CUDA 12.8 (2026-08-23, running)

`sayap` also asked whether a CUDA 12.9 downgrade helps. It had never been tried:
every one of the 25 aborts recorded here is from a CUDA 13.3 build
(`libcudart.so.13`), so there was nothing to report either way.

There is prior reason to take it seriously beyond the suggestion. `build-cuda12.sh`
exists in this repo because a CUDA 13 build of *mainline* llama.cpp loses most of
its throughput on sm_120 once the KV cache passes 8192 tokens, while the same
source built with CUDA 12.8 does not (TUNING §9). A 12-vs-13 difference on
Blackwell is already established here for something else.

### 43.1 One variable, and it was verified rather than assumed

| | |
|---|---|
| binary | CUDA **12.9.2** — `libcudart.so.12.9.79`, `libcublas.so.12.9.2.10` |
| verified | `readlink /proc/PID/exe`, 44 libraries mapped from `build-cuda12`, cudart in the process maps |
| #2347 | **reverted** — proven not to fix it (§42), so not worth its 2.3 % |
| `raw_k` guard | **off** via `IK_RAWK_GUARD=0`, confirmed by compute buffer 7040 not 7576 |
| probes | all off |
| sampler patch | on — the abort must stay survivable and, more to the point, logged |

Testing two candidates at once would say nothing about which mattered, which is
why the guard is off despite being the cheaper of the two.

### 43.1a A trap in the container build, now fixed

`build-cuda12.sh` copied the container's CUDA libraries next to the binaries but
never removed what was there before. Building 12.9 over an existing 12.8 tree
left **both** sets in `bin/`:

    libcudart.so.12.8.57   libcublas.so.12.8.3.14
    libcudart.so.12.9.79   libcublas.so.12.9.2.10

The `libcudart.so.12` symlink did point at 12.9, so the right one would have
loaded — but that is a fact nobody had checked, and the whole point of this test
is that only one variable moves. The stale files were deleted and the script now
clears them before bundling.

Verified afterwards from `/proc/PID/maps` rather than from what was built:
`libcudart.so.12.9.79`, `libcublas.so.12.9.2.10`, `libcublasLt.so.12.9.2.10`.

### 43.2 How it reads

| outcome | conclusion |
|---|---|
| ~8 M prefilled tokens clean | CUDA 13 is implicated — reportable |
| an abort arrives | CUDA is not it; next candidate is the `raw_k` guard at 1.1 % |

The order was chosen on strength, not cost: §41.5 failed to find the lifetime
conflict the guard predicts, while the CUDA difference on this hardware is
already measured for another symptom.

### 43.3 Answered: CUDA 12.9 does not fix it either

    CUDA 12.9.79, #2347 reverted, raw_k guard off
    527 387 prefilled tokens, 3.0 h  ->  1 abort, same signature

Verified from `/proc/PID/maps` at the time (`libcudart.so.12.9.79`), with the
guard confirmed off by the 7040 MiB compute buffer and #2347 confirmed absent
from the source. One variable, and it was the one under test.

So both of `sayap`'s suggestions are answered and both are negative. The
CUDA 12-vs-13 difference on this box is real for throughput (TUNING §9) and has
no bearing on this abort.

Standing tally of candidates:

| candidate | result |
|---|---|
| f16 overflow in `K·Q` | refuted by measurement, §35.1 |
| PR #2347 | abort after 338 509 tokens, §42 |
| CUDA 12.9 | abort after 527 387 tokens |
| `raw_k` allocator guard | **now under test** — the direct proof failed (§41.5), but it is the last candidate and the cheapest at 1.1 % |

A note for the next configuration switch: starting the new server immediately
after stopping the old one failed with `cudaMalloc failed: out of memory` on the
86 103 MiB weight buffer. The card had not finished releasing. That is not a
property of the configuration being tested, and reads exactly like one.

### 43.4 And the guard does not fix it either — the list is exhausted

    CUDA 13.3, #2347 reverted, raw_k guard ON (compute buffer 7576 not 7040)
    244 849 prefilled tokens, 0.8 h  ->  1 abort, same signature

Sooner than any other configuration tested. §41's candidate is dead.

This is the outcome §41.5 pointed at: the direct probe looked for the lifetime
conflict the guard presumes and found **0 in 283 000 verifications**. That was
recorded at the time as weakening the hypothesis, and it was weakening it
correctly.

The guard is therefore reverted, and good riddance — it matched on the string
`"raw_k-"` inside the scheduler, which has no business knowing tensor names from
one model's architecture. It was a hypothesis test wearing the shape of a fix,
and it is worth being explicit that it never was one.

| candidate | result |
|---|---|
| f16 overflow in `K·Q` | refuted by measurement, §35.1 |
| PR #2347 | abort after 338 509 tokens, §42 |
| CUDA 12.9 | abort after 527 387 tokens, §43.3 |
| `raw_k` allocator guard | abort after 244 849 tokens |

Four candidates, four negatives, each measured rather than argued. The cause is
open and the list of ideas is empty. What survives from all of it is the chain in
§40 — every step measured except the first, which remains: **`CUDA0#raw_k-0` is
NaN on the device while the host bytes are clean and the copy is byte-perfect.**

And §42.2 still stands as the limit on everything tried since: every probe here
reads device memory back, which synchronises, so none of them could ever have
caught a race. If that is what this is, nothing in this investigation was capable
of finding it.

---

## 44. The verification sweep: my input evidence was polluted, and the fingerprint has an innocent explanation (2026-08-23)

Seven independent code readers verified each link of the foreign-f32-overwrite
mechanism, every claim cited to file:line. Two things came back that change the
picture more than any hypothesis so far.

### 44.1 The "inputs ALREADY NaN" annotations were read AFTER the graph finished

`IK_NAN_CHECK` flags node outputs in-stream (trustworthy), but the per-input
"ALREADY NaN" annotations are printed by `ik_nan_report` **after graph
completion** — a post-hoc readback. By then the input copies' memory has been
**legally freed and reused**: with `n_copies==1` a split-input copy is an op-NONE
node freed after its last consumer (ggml-alloc.c:653-667), and any later node
output can inherit its bytes. The overlap-log reconstruction shows exactly that:
the `dsv4_raw_mask_padded-0` region is recycled within the same graph to **layer
38-42 f32 outputs** (ffn_up_gate, kv_b, hca/csa state, attn_wo_a — full covers).

**So the 1.55 % fingerprint is real data with an innocent explanation**: f32
outputs of late layers legally occupying the freed mask region at readback time,
read as f16 → 1.561 % NaN. It is not the poison; it is what freed memory looks
like after the graph. §36.4, §38.2 and §40.2's "device copy is NaN while host is
clean" rested on those annotations and are hereby downgraded: the true content of
fattn-0's inputs *at read time* was never validly observed.

### 44.2 What still stands, on trustworthy probes only

* First NaN **node output** in stream order is `fattn-0` (`FLASH_ATTN_EXT`,
  layer 0) — in-stream probe, valid.
* Host KV cache: layers 0-2 clean, layers 3-42 poisoned for exactly the batch's
  cells — host-side reads, valid.
* Copy-time checks: host source clean, destination byte-perfect right after the
  copy (1200 healthy comparisons + abort-time reports) — valid.
* 283 000 synchronised copy-to-consume verifications: no change — valid but
  race-blind.

### 44.3 Verified code facts from the sweep (file:line in the workflow transcript)

* The split-input H2D copy is `cudaMemcpyAsync` on **`cudaStreamPerThread`** +
  sync of only that stream (ggml-cuda.cu:649-655). Compute runs on a
  `cudaStreamNonBlocking` stream (common.cuh:876-885). The copy has **zero
  device-side ordering** against compute, either direction.
* The only ordering is the scheduler's host-side `needs_sync` gate → full
  `cudaStreamSynchronize`; events are dead code since `n_copies==1`.
* **Mainline synchronises unconditionally before every non-user input copy**
  (mainline ggml-backend.cpp:1622-1627, 1717-1724). ik gates it. Upstream
  accepted sync-elision on this path **twice and reverted it twice** — 57819b8d4
  (Gerganov, 2026-03-12) and 86b94708f (NVIDIA, 2026-06-30). This exact class
  was judged unsafe upstream.
* `llama_decode_internal` **never drains after graph compute** — the post-logits
  sync is commented out (llama.cpp:6838-6839); token generation drains via
  `llama_get_logits`, **prefill ubatches do not**. Graphs N and N+1 overlap on
  device during prefill; per-ubatch exposure matches the abort being
  per-prefilled-token.
* Within one `copy_inputs` call, expert-weight uploads go on the **compute
  stream un-waited** while later inputs deposit via `cudaStreamPerThread`
  concurrently — a window mainline forbids.

### 44.4 The two hypotheses that survive

**H1 — the FLASH_ATTN_EXT kernel itself** intermittently produces NaN from clean
inputs on sm_120. Reconnects with #2317 (sm_120 takes different branches) — and
was never really tested: §35 measured the *DSA indexer GEMM*, a different op.

**H2 — a device-side race corrupts fattn's true inputs at read time**, invisible
to every probe used so far (all synchronise) and misattributed by the post-hoc
readback.

### 44.5 The discriminating experiment

Launch NaN-check kernels on the **inputs** of `FLASH_ATTN_EXT`, on the compute
stream, **immediately before the node** — stream-ordered, no synchronisation, so
races stay live; flags read after the graph as today. At the next abort:

| input flags | output flag | verdict |
|---|---|---|
| clean | NaN | **H1** — kernel bug, from clean inputs |
| NaN | NaN | **H2** — inputs corrupted before the read |

Companions, both free of behavioural change: a host-side live-interval assert in
galloc (names any plan violation deterministically), and — as the treatment test
if H2 wins — `k_set_sync = true`, one line, which restores mainline's ordering
semantics exactly.

### 44.6 The probe was suppressing the fault it hunts — Matt's observation, quantified

Matt noticed the recent no-probe configurations all aborted within a few hundred
thousand tokens while the probe run sailed past a million. Tokens-to-first-abort
for all 19 aborts since 08-20, split by the one shared factor:

| configuration | aborts | median to abort | max |
|---|---:|---:|---:|
| no probes (no added synchronisation) | 9 | **154 k** | 446 k |
| probes on (`ik_nan_report` synchronised per split) | 10 | **900 k** | 2.86 M |

No no-probe run survived past 446 k; half the probe runs exceeded 900 k. The
report ran once per split and synchronised every time — the probe serialised the
pipeline and throttled the fault. This is also independent evidence that the
fault is **concurrency-sensitive**, favouring H2-class mechanisms over a purely
deterministic kernel bug, and consistent with §44.3's finding that prefill
ubatches never drain between graphs.

Fixed by making the flag readback stream-ordered and asynchronous: the report
queues `cudaMemcpyFromSymbolAsync` into pinned memory and returns; only when a
latched flag is seen on a later call does it synchronise (the fault has already
happened by then, so that sync is free). Verdict latency: one split. The
concurrency window is restored to no-probe conditions, so the verdict should now
arrive on the no-probe timescale — a few hundred thousand tokens.

---

## 45. THE VERDICT: H2 — an asynchronous writer poisons device memory mid-graph (2026-08-24)

Three aborts on the input-probe build (456 k tokens, 2.0 h), 231 verdict lines
in three clusters, one per abort. Read chronologically:

**Cluster 2 is the conviction.** The first in-stream latch of the whole event is
node 35, op **CONCAT** (`csa_k_all-4`): its `src[0]` — `CUDA0#raw_k-4`, the
scheduler's device copy of host-cache K — carried NaN into the concat while the
other source was clean and **no earlier node in the graph had produced NaN**.
A concat only copies bytes. Therefore the `raw_k-4` device copy was poisoned
*after* its deposit and *before* the concat read it. This is §40's claim
resurrected with valid evidence: in-stream, at execution time, not a post-hoc
readback of freed memory. Cluster 3 repeats the pattern (`hca_k_all-7`).

**Cluster 1 fits the same writer.** `fattn-0` inputs verified clean by the probe
kernel immediately before the node, output NaN — an asynchronous writer landing
inside the probe→kernel window. A deterministic kernel bug (H1) cannot explain
clusters 2/3: no kernel malfunction at layer 0 puts NaN into layer 4's K copy
with every upstream producer clean.

**Verdict: H2.** A device-side write that is not ordered against the compute
stream corrupts input-copy regions while the graph runs. The mechanism verified
in §44.3 predicts exactly this: during prefill `llama_decode` never drains
(the post-compute sync is commented out, llama.cpp:6838), so while graph N's
kernels still run, the host prepares ubatch N+1 and deposits its inputs via
`cudaStreamPerThread` — which has **zero ordering** against the non-blocking
compute stream — into arena addresses graph N is still using.

### 45.1 The treatment test

One line in `llama_decode_internal`: drain the scheduler immediately before
`llama_set_inputs`, closing exactly the unordered-deposit window and nothing
else (one sync per ubatch — far coarser than the per-split serialisation that
throttled the fault in §44.6). Probes OFF for the soak: maximum fault pressure,
and the readout is abort/no-abort. No-probe runs abort at median 154 k, max
446 k; a clean run of several million tokens convicts the window.

### 45.2 Refuted in under an hour — the set_inputs window is not the writer

    drain before llama_set_inputs, probes off, CUDA graphs on
    282 701 tokens -> abort, then two more back-to-back during the re-prefills
    (an abort mid-prefill prints no timing line, hence the zero counts between)

First abort squarely inside the untreated distribution (median 154 k, max
446 k). The drain build was verified running: libllama.so newer than the last
source edit, the call present, process started after the link. Zero effect.

So the H2 verdict stands -- the in-stream evidence of raw_k-4 poisoned between
deposit and read is untouched by this -- but the writer is not the next
ubatch's input deposits. What §44.3 leaves as the remaining unordered window:
within one copy_inputs call, expert-weight uploads stream on the COMPUTE stream
un-waited while later inputs of the same split deposit via cudaStreamPerThread,
concurrently -- the gate is consumed by the first input and never re-arms.
Mainline re-drains before EVERY such deposit; ik elides it; upstream accepted
that elision twice and reverted it twice. The one-line test is
`k_set_sync = true` in ggml-backend.cpp, restoring mainline semantics exactly.
The drain from 45.1 comes out first -- one variable.

### 45.3 k_set_sync = true does not stop it either — but it may thin it

    mainline deposit ordering, probes off, CUDA graphs on
    727 682 tokens -> abort, then the usual two back-to-back on re-prefill

First abort 1.6x beyond the untreated maximum (446 k) and 4.7x the median --
suggestive of a partial effect (P ~4 % as tail luck, one sample), but partial is
not fixed. Both scheduler-ordering windows from §44.3 are now refuted as the
full explanation: the set_inputs drain (45.2, no effect at all) and the
intra-split deposit elision (this).

What today's origins say that nothing else has: every poisoning origin sits in
the expert-streaming layer range. Cluster 1: layer 0. Cluster 2: raw_k-4,
layer 4. Cluster 3: layer 7. Historical: raw_k-0 and mask-0, layer 0; host KV
poisoning enters at layer 3. With --n-cpu-moe 19, layers 0-18 stream their
expert weights from host per prefill batch -- layers 19+ never show an origin.
Expert weights are quantised blocks, i.e. random-ish bytes, which also fits
raw_k's 3-7 % NaN-as-f16 fraction better than the pure-f32 1.56 % the mask
region showed post-hoc. The expert-upload path itself (and its prefetch engine,
ggml_moe_prefetch_*) is now the prime suspect -- with k_set_sync=true its
uploads are drained before later deposits, but any write it issues OUTSIDE the
scheduler's deposit protocol is still unordered.

### 45.4 The expert-upload machinery, read to the bottom (three verified readers)

**The prefetch engine is exonerated.** ggml-moe-prefetch.cpp contains no cuda*
call, no memcpy, no memset -- its workers only run mincore() and
madvise(MADV_POPULATE_READ/WILLNEED/COLD), which move page residency, never
bytes. It cannot touch the KV cache either: it refuses tensors outside the
registered model-file mmaps, and the -nkvo cache is cudaMallocHost memory that
cannot lie inside a file mmap. One residual flagged, not verified: MADV_COLD is
issued on H2D source pages right after the async eval, potentially while the
copy is in flight -- harmless under classic staging, driver-dependent under HMM.

**The uploads cannot write out of bounds -- proven.** Every byte funnels through
ggml_backend_tensor_set_async, which carries an unconditional
GGML_ASSERT(offset + size <= ggml_nbytes) that has never fired; run boundaries
are clamped; the "+512 B padding" writes the next expert's own true bytes at
their correct offsets; the ids readback is host-synchronised on the producing
stream before any offset is computed. The residue is aliasing, not overrun:
with n_copies==1 the expert copies carry no OUTPUT flag, so galloc reuses their
exact bytes for later tensors -- raw_k and K-assembly among them -- and nothing
ever host-waits on the fire-and-forget uploads riding the compute stream.

**What -no-ooae removes is exactly the suspicious branch and nothing else:**
un-waited compute-stream deposits of expert SLICES sourced from pageable mmap
memory, replaced by whole-tensor uploads on cudaStreamPerThread with an
immediate sync -- the same boring primitive every other input uses. MoE matmuls
for layers 0-18 stay on the GPU during prefill (the offload decision never
reads only_active_experts), so the workload shape is preserved; prefill cost
measured at -6 %. The prefetch engine keeps running, which is fine -- it is a
page warmer, not a writer.

The running -no-ooae soak therefore asks a single clean question: is the only
fire-and-forget device writer in the system the poisoner? If aborts stop, the
bisect inside the branch is ready (keep OOAE, synchronise the uploads). If they
continue, the branch is exonerated wholesale and the candidates left are the
allocator itself (the live-interval assert from §44.5) and the pageable/HMM
residue above.

## 45.5 CONVICTED: the expert-upload branch — and the bisection (2026-08-24)

    -no-ooae, k_set_sync=true, probes off, CUDA graphs on
    2 468 081 prefilled tokens, 6.4 h, 90 requests -> ZERO aborts

Against the untreated rate measured over eleven no-probe runs (1 abort per
246 767 tokens, 2.71 M tokens of data), ten aborts were expected. P(0 by
chance) = **0.0045 %**. The expert-upload branch is the cause.

It fits every constraint gathered over eight days: prefill only (expert
streaming needs batch >= 32); origins only in layers 0-18 (the --n-cpu-moe
range) and never 19+; foreign quantised bytes rather than computed NaN;
sensitivity to added synchronisation; and §45.4's finding that these uploads are
the only fire-and-forget device writer in the tree, whose bytes galloc then
recycles for raw_k and the K-assembly tensors.

**Now bisecting.** OOAE is back ON and a single `ggml_backend_synchronize` is
added after the per-expert upload loop (env `IK_SYNC_EXPERT_UPLOADS`), keeping
the slicing and the ids identical and removing only the fire-and-forget
property. Prefill measured at 1707 t/s — indistinguishable from the 1709 t/s
baseline, so if this is the fix it is a free one.

* clean at 2.5 M -> asynchrony was the poison; the fix is ordering, and it costs nothing
* abort -> the slicing or the ids are at fault, and a sync is not the fix

### 45.6 Bisection answer: NOT the asynchrony — the slicing or the ids

    OOAE on, IK_SYNC_EXPERT_UPLOADS=1 (verified in /proc/PID/environ),
    build 21:36:42 / process 21:37:18
    134 804 tokens -> abort, then the usual two on re-prefill

Below the untreated median (154 k). Draining after the upload loop changes
nothing, so the fire-and-forget property is NOT the poison — which also
retires §44.3's whole ordering theory, since every ordering treatment has now
failed (set_inputs drain, k_set_sync, this).

What -no-ooae removes and this does not: the **slicing** (contiguous active-
expert ranges plus the 512-byte over-copy) and the **ids** driving it. §45.4
proved every byte lands inside input_cpy's allocation — the assert would have
fired otherwise — so the write is in-bounds, and the fault must be in WHAT is
written or WHAT IS NOT: with slicing, only active experts are deposited, and
the untouched remainder of input_cpy keeps whatever the previous tensor left
there. Under -no-ooae the whole tensor is written every time, overwriting all
of it.

That reframes the fault: not a stray writer at all, but stale bytes being READ
from regions this branch deliberately skips. It fits what the probes saw --
foreign quantised-looking bytes inside a device buffer whose live region was
byte-perfect -- and it fits galloc recycling expert-copy blocks into raw_k and
the K-assembly tensors. Next: test the reading side, i.e. whether a consumer
touches expert rows that were never deposited this pass.

### 45.7 The gaps are not read every time (weak negative)

Painting the whole destination with 0xFF before the sliced deposits (env
`IK_PAINT_EXPERT_GAPS`) produced no NaN, and the model answered correctly. The
paint demonstrably ran -- prefill collapsed from 1700 to 24 t/s under the extra
H2D traffic, and no CUDA error was logged.

Honest limit: this refutes "the consumer reads the gaps on every pass" only.
At 24 t/s a soak long enough to test "reads them rarely" is impractical, so the
rare variant stands untested. Recorded as weak evidence, not a refutation.

### 45.8 Isolating the slicing from the rest of the branch

-no-ooae changes four things at once: the ids readback (a D2H of the expert ids
plus the unique-expert bitmap), `ggml_moe_prefetch_wait`, a `needs_sync` side
effect on the ids backend, and the sliced deposits. §45.6 blamed the slicing on
reasoning alone, which is exactly the kind of step that produced §36-§40.

Now running with all of the branch intact and only the deposits replaced by a
single whole-tensor upload through the same primitive on the same stream
(`IK_WHOLE_EXPERT_UPLOAD`, verified in /proc/PID/environ, no other switch set,
OOAE on). Prefill 1577 t/s.

* clean at ~2.5 M -> the partial deposit is the cause
* abort -> the cause is the ids readback, prefetch_wait, or the needs_sync side effect

### 45.9 The slicing is NOT the cause — and §45.6 was wrong

    OOAE on, IK_WHOLE_EXPERT_UPLOAD=1 (verified in /proc/PID/environ),
    no other switch set
    290 713 tokens -> abort

Inside the untreated distribution. The deposits were whole-tensor, byte for byte
what -no-ooae uploads, through the same primitive on the same stream -- and it
still aborts. So the partial deposit is exonerated, and with it the §45.6 story
about stale bytes surviving in the gaps, which §45.7's paint test had already
failed to support.

That was a conclusion reached by reasoning rather than by measurement, and it
was wrong -- the same failure mode as §36-§40. Worth stating plainly: in this
investigation every hypothesis argued from code and refuted by experiment has
been wrong, and the only durable results have come from measurement.

**What is left in the branch**, now the sole remaining difference between the
aborting build and the 2.47 M-clean -no-ooae run:

| candidate | what it does |
|---|---|
| the ids readback | `ggml_backend_tensor_get_async(ids_backend, ids_tensor, ...)` + `ggml_backend_synchronize(ids_backend)` -- a D2H into a pageable `std::vector`, mid-copy_inputs |
| `ggml_moe_prefetch_wait(input)` | host-side wait on the page-warming pool |
| the `needs_sync` side effect | `needs_sync[ids_backend_id] = k_set_sync` at ggml-backend.cpp:2343 |

Note `ids_backend` defaults to `split_backend` and is only corrected when the
ids tensor is a LATER input of the same split -- so when the ids tensor lives on
another backend, this reads it through the wrong backend's stream. That is the
first thing to test, and it is testable by assertion rather than by soak.

### 45.10 The ids readback holds up under assertion — and the branch is smaller than assumed

Three tripwires on the ids readback (env `IK_IDS_CHECK`), with a liveness
heartbeat because a probe that only speaks on failure cannot be told apart from
one that never ran:

1. `ids_backend` defaulting to `split_backend` when the ids tensor lives elsewhere
2. the `ids` vector sized by `ggml_nbytes` while the loop indexes by `nb[1]`/`nb[0]`
3. an expert id outside `[0, n_expert)`

**201 verified readbacks, zero findings.** All three assumptions hold on the
live workload. So the ids readback is not obviously broken, and candidate 1 of
§45.9 is weakened (not eliminated -- 201 samples cannot exclude a rare event).

Two facts worth recording, both contradicting what I had assumed:

* The branch is entered for the ids of layers **24, 29, 36** -- not 0-18. With
  `-ncmoe 19` the host-resident expert weights are in the UPPER layers on this
  build, so §45.9's "origins cluster in 0-18 = the streaming range" reasoning
  had the range backwards. The origins (layers 0-7) and the streamed layers
  (24+) do not overlap at all.
* It runs ~65 times per prefill of a 4000-line prompt, not once per split.

The first point matters: it removes the layer-containment argument that made the
expert branch look guilty in the first place. What still stands is purely
experimental -- `-no-ooae` ran 2.47 M clean where 10 aborts were expected, and
that is a fact about the branch as a whole, not about any story told over it.

### 45.11 Prefetch eliminated for free — and the difference I had missed

`--prefetch-experts` is **not set in this profile** (default false,
common.h:458) and never has been. So the pool is never created,
`ggml_moe_prefetch_enabled()` is false and `ggml_moe_prefetch_wait` is a no-op.
Candidate 2 of §45.9 dies without a test, and the whole prefetch engine --
including §45.4's unverified MADV_COLD residual -- has been irrelevant all
along.

Candidate 3 (the `needs_sync` side effect) is also already answered: with
`k_set_sync = true` that line sets needs_sync to TRUE, the safe direction, and
§45.3 aborted anyway.

Which leaves nothing from my original list -- so the list was incomplete. The
difference I never enumerated is **the deposit primitive itself**:

| path | primitive | stream |
|---|---|---|
| OOAE (aborts) | `ggml_backend_tensor_set_async` | the **compute** stream, no wait |
| generic, used by -no-ooae (2.47 M clean) | `ggml_backend_tensor_set` | `cudaStreamPerThread` + immediate sync of that stream |

§45.9 swapped slicing for whole-tensor but kept `set_async`, so it never tested
this. §45.5 added a drain after the loop, which is not the same as depositing on
a different stream and waiting there.

Now running with only the primitive swapped -- slicing, ids and everything else
untouched (`IK_SYNC_DEPOSIT`, verified in /proc/PID/environ, no other switch
set, OOAE on). Prefill 1682 t/s.

* clean at ~2.5 M -> the compute-stream deposit is the mechanism, and the fix
  is to deposit like every other input does
* abort -> the branch differs from the generic path in some way still not
  enumerated, and enumerating it by reading is what has failed four times

### 45.12 The primitive is not it either — but something here reduces the rate

    OOAE on, IK_SYNC_DEPOSIT=1 (verified in /proc/PID/environ), no other switch
    1 abort after 1 328 753 tokens; 1 828 174 tokens total, 8.0 h

Depositing through `ggml_backend_tensor_set` -- cudaStreamPerThread with an
immediate sync of that stream, exactly what the generic path does -- still
aborts. So the deposit primitive is not the mechanism either.

The rate, though, is not the untreated rate: 7.4 aborts expected, 1 observed
(P(<=1) = 0.5 %). The same was true of `k_set_sync = true` (first abort at
727 k where the median is 154 k). Both treatments add synchronisation and both
appear to thin the fault without removing it -- the same dose-response §44.6
measured with the probes, where per-split synchronisation moved the median from
154 k to 900 k.

That pattern is now the most robust thing known about this fault after the
in-stream verdict, and it argues that `-no-ooae`'s 2.47 M clean run may itself
be a strong version of the same thinning rather than a cure. The honest reading
of §45.5 is therefore weaker than written there: 0 aborts in 2.47 M is
P = 0.0045 % against the UNTREATED rate, but against a rate already thinned
~5x by added synchronisation it is only ~P = 13 %, which is not significant at
all. **§45.5's conviction of the branch does not survive that correction.**

What survives, and it is not nothing:

* the in-stream verdict (§45): a device copy is poisoned between deposit and
  consumption, with every upstream producer clean
* the dose-response: every treatment that adds synchronisation reduces the rate,
  none eliminates it
* the layer facts (§45.10): origins in layers 0-7, streamed experts in 24+ --
  no overlap

The next step is not another single-variable soak. Each costs hours and the
last four have produced negatives with confidence intervals too wide to
separate "fixed" from "thinned". What is needed is either a deterministic
detector (the galloc live-interval assert from §44.5, which names a violation
the first time it happens) or a reproducer that raises the rate instead of
lowering it.

---

## 46. The checksum detector: measuring the fact itself (2026-08-25)

§45.12 ended with the realisation that single-variable soaks can no longer
answer anything: each costs hours and cannot separate "fixed" from "thinned",
because every added synchronisation thins the rate (the dose-response of
§44.6). And §45's verdict, on re-examination, was weaker than written: the
in-stream probe covered FLASH_ATTN_EXT only, so "raw_k-4 carried NaN into the
concat" rested on a post-hoc annotation -- the same invalid mechanism as §44.1.
What is actually established is only: *some split-input copy changes between
deposit and consumption*.

So measure exactly that. `IK_SUM_CHECK`:

* **mark** -- right after each split-input deposit (all three copy_inputs
  paths), a kernel on the compute stream sums the copy's bytes into a slot
* **verify** -- right before every consumer, a kernel re-sums and a third
  compares, latching the first mismatch (consumer node, src index, both sums)
* no synchronisation while clean (async pinned readback, §44.6's pattern);
  liveness heartbeats on both sides (§38.1's lesson); slots from a rolling ring
  so cross-graph overlap cannot alias them; the map cleared per compute_splits
  because the graph arena reuses addresses
* runs with CUDA graphs disabled -- marks execute per iteration outside any
  capture, so slots would go stale under replay

Two properties matter. It is **methodologically valid** -- in-stream, at
consumption time, no post-hoc reads. And it is an **amplifier**: it catches ANY
mutation of a copy, not just the ~3 % that read as f16 NaN, so if the fault is
a foreign write, mismatches should fire far more often than aborts ever did.

Verified live: 21 425 marks / 95 812 verifies after one 35k-token request
(~4.5 consumers per copy on average, mask and raw_k copies visibly tracked),
zero false positives, prefill 1676 t/s. Accepted blind spot: a writer already
queued on the compute stream between deposit and mark is summed into the
baseline -- out-of-stream writers, the H2 class, are what this catches.

The readout changes meaning now:

| observation | conclusion |
|---|---|
| MISMATCH fires (with or without abort) | the foreign-write class is real; the consumer and tensor are named |
| aborts continue, sums NEVER mismatch | no out-of-stream writer touches the copies -- H2 as formulated is dead, and the poison is computed into existence somewhere the probes have not looked |

Either way the next abort finally pays for itself.

### 46.1 The abort came with the detector live: ZERO mismatches in 49.8 M verifications

    IK_SUM_CHECK on, graphs off, abort after 1 260 953 tokens
    10.9 M marks / 49.8 M verifies -- every sum matched, including through the abort

**H2 as formulated is dead.** No out-of-stream writer touches the split-input
copies -- not once in fifty million deposit-to-consumption checks, not even
during the poisoning itself. An amplifier that catches ANY byte change stayed
silent while the logits went NaN.

What survives every valid measurement: the first NaN in stream order is always
the OUTPUT of a FLASH_ATTN_EXT node (in-stream output probes); the one direct
input observation said "inputs clean, output NaN" (45, cluster 1); and now the
inputs are proven untouched. Everything converges on **H1: the FLASH_ATTN_EXT
kernel intermittently computes NaN from clean inputs** -- on sm_120,
concurrency-sensitively, which is the signature of an intra-kernel race
(shared memory, barriers, warp divergence; #2317 already documented sm_120
taking different branches in FA).

The §44.6 dose-response reads naturally now: more host synchronisation means
lower occupancy pressure and different intra-kernel timing -- thinning a race
inside the kernel just as it thinned everything else.

### 46.2 The decisive test: run every FLASH_ATTN_EXT twice and compare

`IK_FA_TWICE`: after the normal execution, sum the output in-stream, execute
the SAME node again on the SAME inputs (attention writes every output element,
so re-execution is semantically safe), sum again, compare, latch on the first
divergence. No synchronisation. A deterministic kernel must produce identical
sums; §46.1 just eliminated the only other explanation for a divergence
(inputs changing between the two adjacent runs). If the kernel is
nondeterministic, this catches it every time it misfires -- with or without NaN
-- and names the node.

### 46.3 Second abort under the detectors: BOTH silent — and what that leaves

    abort after 111 478 tokens; 0 sum mismatches, 0 FA divergences (885 double-runs)

The inputs were not touched and the kernel, run twice on identical inputs,
agreed with itself even during the poisoning. Two explanations remain:

**(a) deterministic NaN from a rare input configuration.** The classic is a
fully-masked attention row (softmax 0/0). Testing this immediately produced a
finding that reshapes it: `IK_MASK_SCAN` shows **fully-masked REAL rows are
routine** -- ~12 per pass, mid-batch (rows ~2234-2238 of ~3432), every healthy
graph, no abort. In DSA the raw branch legitimately selects nothing for some
tokens, so the kernel must guard this case -- and does, almost always. The
surviving form of (a): the guard fails on a rare variant (row position within a
tile, tile fully masked, a kernel-variant boundary). Deterministic, data-shaped,
timing-of-batching-sensitive -- which would also finally explain §44.6's
dose-response: added synchronisation changes ubatch composition and padding,
i.e. the probability of the fatal shape, not any race.

**(b) the poison is already in the deposited bytes** -- host-side. Sum-check
cannot see it by design; §39's copy-check saw host-clean for layer 0 in the old
events, but layer-0 K passes through device-computed q_rope/k paths that no
probe currently covers at content level.

This run had no IK_NAN_CHECK, so the first-NaN node for this abort is unknown.
Now deployed, everything at once: NAN_CHECK (first NaN node + fattn input NaN
state) + SUM_CHECK + FA_TWICE + MASK_SCAN, graphs off, prefill 1388 t/s. The
next abort answers, in one event: which node, whether its inputs held NaN at
read time, whether they changed since deposit, whether the kernel diverged, and
what the mask's fully-masked-row picture looked like at that moment.

---

## 47. Dump-on-abort: the poisoned state becomes the reproducer (2026-08-26)

§46.4's four-way verdict makes the next abort worth ~300 MB of disk: fattn-0
computes NaN **deterministically** from clean inputs, and for layer 0 the whole
computation is reconstructible offline -- Q depends only on the ubatch tokens,
K on the layer-0 KV cache (host-resident with -nkvo), the mask on the batch
geometry. All of it survives the abort on the host side.

The sampler-catch in server-context.cpp now saves, BEFORE erasing the slot
(`IK_NAN_DUMP=0` to disable):

    logs/nan-dump-<us>/tokens.bin   full token history, int32
    logs/nan-dump-<us>/kv_l0.bin    layer-0 KV bytes, self-describing header
    logs/nan-dump-<us>/kv_l1.bin    layer-1 likewise
    logs/nan-dump-<us>/meta.txt     n_past, kv head/used/size, n_ubatch

Helpers `llama_ik_kv_info` / `llama_ik_dump_kv_layer` added to src/llama.cpp
(the KV tensors are not reachable through any public API). Full diagnostic
stack stays on; prefill 1398 t/s.

Plan once a dump lands: an offline replay harness re-runs the failing ubatch
from the dump -- same tokens, same KV, same shapes. Deterministic kernel means
the NaN must reproduce on demand, and from there the kernel can be dissected
in minutes per experiment instead of hours per soak. That reproducer, not a
statistic, is what upstream gets.

### 47.1 Two offline replays, two negatives — and the reason is structural

The replay harness (tools/replay-nan-dump.cpp) ran the dumped 125 196 tokens
with the final chunk aligned to the aborting shape (2416): all logits finite.
Injecting the dumped layer-0/1 KV bytes over the rebuilt cache before the fatal
chunk: still finite.

The structural reason: in DSA the mask is derived from the **indexer state**
(the csa/lid/hca state buffers) via top-k selection, and that state was neither
dumped nor reconstructible bit-exactly -- the server built it across many
requests with many batch shapes, and the trigger is a bit pattern. Replaying
tokens rebuilds a mathematically equivalent but bitwise different state.

### 47.2 In-place replay: the server is the replay device

Everything the failing decode read is still hot in the server process at catch
time -- KV, indexer state, all of it. The catch now (before dropping the slot):

1. `llama_kv_cache_seq_rm` of the failing batch's range
2. arms `ik_fattn_capture` in the CUDA layer
3. re-runs `llama_decode(batch_view)` -- same batch, same everything
4. counts NaN logits of the re-run and logs `in-place replay after NaN abort`
   with rc / nan_logits / captured / capture_dir

During the re-run the CUDA layer captures **fattn-0's exact device input bytes**
(all srcs, with types/shapes/strides) plus its output to
`logs/fattn0-capture-<us>/` (~340 MB, once). If the re-run reproduces --
§46.4's determinism says it should -- those files are the standalone
reproducer: the kernel can then be fired at them in isolation, no model, no
145 GB, no cache. `IK_NO_INPLACE_REPLAY=1` disables.

Readout: `nan_logits > 0` in the replay log line = determinism confirmed on the
real state + capture in hand. `nan_logits = 0` = the re-run diverged, which
would point at the indexer state mutating during the failed decode.

### 47.3 In-place replay REPRODUCED — and the walk-back finds the origin chunk

    abort 11:00:48; in-place replay: rc=0, nan_logits=129280 (ALL)

Deterministic reproduction on the real state, on demand — §46.4 confirmed in
production. But the capture photographed a healthy graph: the batch in scope at
the catch is the LAST batch, not the poisoning chunk; its fattn-0 was clean
(out.bin NaN=0, shape 8192 vs the origin's 6728) and the replay's NaN was pure
propagation from the poisoned cache in layers 3+. The armed capture then stayed
live and shot the next innocent prefill.

Replaced by the **walk-back replay**: rewind KV to the request's prefill base,
re-run THIS request's chunks exactly as the server chunked them (bit-exact —
what the offline replays could never be, §47.1), consume the in-stream NaN
latch after each chunk. The first chunk that latches is the origin; it is
re-run once more with the fattn-0 capture armed, and the capture disarms when
the window closes so it can never again photograph an unrelated graph. Each
chunk logs `walk-back replay chunk {p0, n, nan_node}`; the finale logs
`ORIGIN chunk re-run` with the capture directory.

### 47.4 The walk-back cannot work — the failing decode mutates its own trigger

    abort 17:02:58; request base 15864, single 1079-token chunk;
    bit-exact re-run of that chunk: CLEAN (nan_node=-1)

The original run of that chunk produced NaN; its bit-exact replay did not. The
only difference: the failing decode had already advanced the DSA indexer
persist state, and the replay ran from the mutated state -- different top-k,
different mask, no NaN. So the trigger involves indexer state that the failure
itself overwrites, and NO after-the-fact replay -- in-place or offline -- can
ever reproduce the origin. (The morning replay "reproduced" only propagation
from the poisoned cache, which survives; the origin does not.)

Replaced by the **origin dump**: ik_nan_report, which runs at the end of the
origin's own split a few nodes after the latch, now dumps the latched
FLASH_ATTN_EXT node's input and output bytes immediately -- during the original
event, before any state moves further. The inputs may already be
galloc-recycled by the origin's same-split successors, so the dump is validated
offline: a genuine mask holds only finite values and -inf, recycled f32 garbage
is unmistakable; q is reconstructible from tokens regardless. Combined with the
abort-time nan-dump (tokens + layer-0/1 KV), a validated origin dump is the
complete standalone reproducer.

### 47.5 Third clean-inputs verdict, first origin dump — and recycling-proof staging

    abort 21:12:53, origin fattn-0 again (n_tokens=1889, n_kv=2048 -- SMALL
    context; deep context is not required)
    in-stream: inputs CLEAN at read time; FA-twice: bit-identical double-run
    (including the NaN pattern); sum-check: silent; output ~3 % NaN

Third independent event with the same three-way verdict: the kernel produces a
~3 % NaN output from clean inputs, bit-deterministically. The origin dump fired
but end-of-split is already too late for the bytes: the mask region showed
1.29 % NaN (recycled by downstream-NaN successors), K likewise; q looked
intact. Exactly the predicted pollution.

Fixed by ORIGIN STAGING: pinned-host copies of fattn-0's inputs, refreshed
every prefill graph by a stream-ordered D2H issued immediately before the node
-- byte-for-byte what the kernel is about to read, immune to galloc recycling,
~50 ms per prefill graph (prefill 1357 t/s). On a latch, ik_nan_report writes
them as staged_src*.bin next to the (kept, polluted) device dumps.

The next abort therefore delivers guaranteed-valid kernel inputs. With those,
the standalone harness runs FLASH_ATTN_EXT on the exact bytes in isolation --
reproduce, then dissect the kernel variant and the value pattern that breaks
its fully-masked-row / softmax guard.

---

## 48. ROOT CAUSE: FLASH_ATTN_EXT emits NaN for a fully-masked leading row block (2026-08-27)

The staged capture from the 22:09 abort validated clean (q/k/sinks finite, mask
99.4 % -inf with ~8 allowed positions per row) and the standalone harness
(tools/fattn-repro.cpp) reproduced ON THE FIRST RUN:

    scale 1/sqrt(512): NaN 458752 / 34734080 -- exactly 14 tokens x 64 heads x 512

The mask has 27 fully-masked rows: 15 REAL leading rows (0-14) plus the 12
padding rows. The NaN tokens are 0-13 -- the leading fully-masked block (row 14
and the padding rows survive; mid-batch fully-masked rows never NaN, §46.3).

Then the two substitutions that close the case:

* **random q/k, real mask** -> identical NaN tokens 0-13. The values are
  irrelevant; the trigger is the mask.
* **random q/k, GENERATED mask** (rows 0-14 all -inf, an 8-wide causal window
  after, padding masked) -> identical NaN tokens 0-13. **The reproducer is
  100 % synthetic** -- a ~150-line program, no model, no capture, no private
  data, fails in under a second on sm_120.

**The bug:** the FLASH_ATTN_EXT path used by the DSV4 raw branch (head dim 512,
IQK_DISABLED, sinks attached, prec F32) mishandles query rows whose mask is
entirely -inf when they form the LEADING block of the batch. With sinks
attached those rows must output zeros (as they do mid-batch and in padding);
instead the leading block yields NaN.

**Why it presented as a once-per-250k-token heisenbug:** DSA legitimately
produces fully-masked raw-branch rows at a batch start -- tokens whose raw
window holds no cells, a boundary condition of batching x indexer state. Every
observed correlate now has its explanation: prefill-only (needs a multi-row
batch), layer 0-7 origins (uncompressed/raw layers), the synchronisation
dose-response (batch composition shifts the boundary-geometry probability, no
race involved), FA-twice bit-identical (deterministic value bug), inputs clean
in-stream (they ARE valid -- the kernel is wrong), ~3 % output NaN (14 of 1060
rows), and the walk-back non-reproduction (the indexer state moved, the mask
lost its leading block).

Eight days, eleven refuted hypotheses, four generations of in-stream
instrumentation -- and the answer is fourteen rows of -inf at the top of a
mask.

### 48.1 The kernel dive: localized to tile indexing, three guards added, handed to the author

The 1-second reproducer turned kernel debugging into printf archaeology.
Established inside `fattn-new-mma.cu` (the new-MMA path, taken for K=V=512):

* the affected rows reach finalize with `KQ_max = -FLT_MAX/2` (no finite score
  ever seen) and `KQ_rowsum = NaN` -- the NaN is born during the iteration
* `rowsum_add` is NaN in the FIRST iteration: the Q.K MMA output is **already
  NaN before the mask is added**
* the mask values read from the shared tile at that point are **garbage**
  (0.366, 0.226 -- in a row whose mask is entirely -inf), so the tile loads are
  reading the wrong memory for this configuration
* `flash_attn_mask_to_KV_max` correctly returns KV_max = 0 for the fully-masked
  leading q-tile; the failure is in how the kernel consumes that

Three unguarded 0/0 divisions were found and guarded on the way
(`flash_attn_combine_results_new`, `flash_attn_stream_k_fixup`, the main
kernel's rowsum normalisation) -- correct hardening regardless, but none of
them is the cause: the NaN enters through garbage tile reads upstream of all
three. A kb0_start clamp against KV_max (suspected inverted stream-k range)
did not cure it either -- the mis-indexing is deeper in the tile/prefetch
logic of this kernel.

That is the right moment to hand over: the kernel is ikawrakow's own, the
reproducer runs in under a second, and tools/fattn-nan-repro.cpp is fully
self-contained -- generated mask, random q/k, hardcoded shapes, zero private
bytes. `NaN 458752 / 34734080, tokens 0..13` on sm_120, exit code 42.

---

## 49. `--swa-compress` replaces `-nkvo` at no cost — and it is the fix for us (2026-08-29)

ikawrakow's reply pointed at `-nkvo` ("the flag is there for the extremely GPU
poor. I never use it, so it is possible something is broken") and recommended
`--swa-compress`, a flag this project had never tried. Measured on this box,
all at `-b 8192 -ub 8192 -c 131072`, unique prompts per run (a repeated prompt
hits the prompt cache and halves the reported prefill -- the first pass of this
comparison was wrong for exactly that reason):

| config | KV buffer | compute buffer | loads? |
|---|---|---|---|
| `-nkvo` (shipped, ncmoe 19) | 0 (host) | 7 576 | yes |
| KV on GPU, no swa-compress | 5 504 | **57 984** | **OOM** |
| KV on GPU + `--swa-compress`, ncmoe 19 | 355 | 15 236 | OOM (4 GB short) |
| **KV on GPU + `--swa-compress`, ncmoe 21** | **355** | **14 212** | **yes** |

Throughput, same build, same prompts:

| depth | `-nkvo` (ncmoe 19) | `--swa-compress` (ncmoe 21) |
|---|---|---|
| ~4.8k prefill | 1439.3 t/s | 1395.9 t/s (−3.0 %) |
| ~38k prefill | 1752.9 t/s | **1773.6 t/s (+1.2 %)** |
| generation @4.8k | 17.99 t/s | 17.68 t/s |
| generation @38k | 17.06 t/s | 17.17 t/s |

**Equivalent** -- and that is with TWO more expert layers pushed to the host
(ncmoe 21 vs 19), which §21 measured at −1.9 % prefill each. The KV cache
shrinks 5 504 → 355 MiB.

Two things worth reporting upstream. Our compute buffer with KV on the GPU is
**57 984 MiB where ikawrakow measures 8 768** at identical batch settings, and
**14 212 where he measures 4 550** with `--swa-compress` -- a 3.1-6.6x gap that
is not explained by his `-cmoe` vs our `-ncmoe 21`, and points at `-mla 3
-fidx` (the DSA raw branch) inflating the attention scratch on this build.
That gap is why §26 concluded "-nkvo cannot be dropped": true then, and true
without `--swa-compress` now.

The `--swa-compress` configuration is now soaking. §34 measured 22 aborts /
34.3 M tokens with `-nkvo` against 0 / 3.7 M without it, so if the fault is
`-nkvo`-specific this run should stay clean -- and the fully-masked-leading-row
reproducer stays valid regardless, since it never used `-nkvo` at all.

## 49.2 ROOT CAUSE, FOUND AND FIXED: the raw-branch window view overshoots kv_head

`-fidx` was exonerated first: masks dumped with it on and off, same conditions,
are **byte-identical** (md5 `7d1d53ee...`). It is a pure indexer fusion.

The fault is in `src/graphs/build_deepseek4.cpp:1130-1141`, the SWA window view:

```c
int nton  = k_fa_chunk*((ntokens + n_swa + k_fa_chunk - 1)/k_fa_chunk);
int first = raw_k->ne[2] - nton;          // <-- anchored on the PADDED kv length
if (first > 0) { /* view raw_k and raw_mask from column `first` */ }
```

`raw_k->ne[2]` is `kv_self.n` = `GGML_PAD(kv_head + n_tokens, 256)`, but the mask
rows are indexed from `kv_head`. So the view starts at `first` while row *j*
needs cells `[kv_head + j - n_swa + 1, kv_head + j]`. Define

    offset = first - kv_head = P - S
    P = GGML_PAD(kv_head+n_tokens,256) - (kv_head+n_tokens)   (0..255)
    S = nton - n_tokens                                        (>= n_swa)

When **P > S** the view starts past what the leading rows need and `offset` of
them come out entirely `-inf`. Confirmed against both captured masks, predicting
`n_kv` from `n_tokens` alone:

| capture | n_tokens | nton (predicted) | n_kv (measured) | offset |
|---|---|---|---|---|
| abort §48 | 1060 | 1280 | **1280** | +15 → 15 empty rows |
| healthy continuation | 8192 | 8448 | **8448** | −128 → none |

`S >= 256` whenever `n_tokens` is a multiple of 256, so **P > S is impossible for
aligned batches** — which is why 8192-token chunks never fail. It needs a
**non-256-aligned `kv_head`**, i.e. prompt-cache reuse, exactly what an agent
client does when it resends a growing conversation. Every recorded abort
geometry (1060, 1889, 2416, 6728) is in the vulnerable class; every synthetic
sweep that never reproduced used aligned chunks.

**Reproduced on demand** (seconds, not 250k tokens): a prefix request followed by
an extended one, so the continuation starts at an unaligned `kv_head` — 86
graphs with empty leading rows and an HTTP 500 NaN abort, first try.

**The fix**, one clause: the window must never start later than the oldest cell
row 0 still needs.

```c
if (int max_first = (int) llm.kv_head - n_swa + 1; first > max_first) {
    first = max_first;
}
```

Verified: same scenario, 6 063 window views, **zero** with `offset > 0` (max
−127, i.e. the clamp), zero empty rows, zero aborts, 45/45 requests OK.
Throughput unchanged (1422/1711 t/s against 1439/1753 before; generation 18.30/
16.95 against 17.99/17.06).

This also explains the two mitigations: `--swa-compress` takes a path where this
block never runs (0 shifts observed), and the author never sees the fault
because he does not use `-nkvo`, whose profile is what forces this branch here.
The kernel-side reproducer stays valid as hardening — `FLASH_ATTN_EXT` should
not turn a fully-masked row into NaN — but it is no longer the bug to fix.

**And ikawrakow was right**: fully-masked leading rows are not legitimate. They
were a malformed mask all along.

### 49.3 Depth comparison: `--swa-compress` wins at 120k, and the fix costs nothing

Both configurations on the fixed build, full prefills (unique prompts; a shared
prefix silently truncates `prompt_n` and makes the number meaningless -- two
runs had to be discarded for exactly that):

| depth | A: `-nkvo`, ncmoe 19 | B: `--swa-compress`, ncmoe 21 | B vs A |
|---|---|---|---|
| ~6.4k prefill | 1647.5 t/s | 1576.8 t/s | −4.3 % |
| ~52k prefill | 1684.9 t/s | 1693.7 t/s | +0.5 % |
| **~122k prefill** | 1359.5 t/s | **1518.9 t/s** | **+11.7 %** |
| TG @6.4k | 18.90 t/s | 18.01 t/s | −4.7 % |
| TG @52k | 18.02 t/s | 17.57 t/s | −2.5 % |
| **TG @122k** | 16.25 t/s | **16.79 t/s** | **+3.3 %** |

The crossover is real and in the predicted direction: with `-nkvo` the KV cache
sits in host RAM and every token pays for it over PCIe, a cost that grows with
depth; `--swa-compress` keeps it at 355 MiB on the GPU. Shallow work slightly
favours A (two fewer expert layers on the host), deep work clearly favours B --
and this box's actual workload is deep.

Also confirms the fix is free: A here (1647/1685/1360) matches the pre-fix
baseline (1439/1753 at the shallower shapes) within run-to-run spread, and the
`-nkvo` profile at 122k is close to §21's 1327.7.

**Both are now correct** -- §49.2 fixes the fault at its source, so this is a
performance choice, not a workaround. B is the better default for deep-context
serving.

### 49.4 The 256k profile: `--swa-compress` wins after all, at `-ub 4096`

The first pass rejected `--swa-compress` at 262144 because it needed `-ncmoe 25`
(four more expert layers on the host than `-nkvo`, against two at 131072) and
that lost more generation than the KV placement won. Matt pushed back, and the
rejected variable was the wrong one: keep `-ncmoe 21` and buy the room with
`-ub 4096` instead.

| depth | A: `-nkvo` n21 ub8192 | B: swa n25 ub8192 | **C: swa n21 ub4096** |
|---|---|---|---|
| ~6.4k | 1383.5 / 16.03 | 1332.4 / 14.58 | 1025.1 / **16.65** |
| ~52k | 1486.9 / 15.17 | 1520.2 / 14.31 | 1223.0 / **16.11** |
| ~122k | 1182.1 / 13.91 | 1342.4 / 14.37 | 1106.6 / **15.82** |

C has the best generation at every depth (+3.9 / +6.2 / +13.7 % over A) and
gives up only raw prefill throughput.

**Weighted by this box's real workload rather than by guesswork.** The server
logs give the depth distribution of 112 completed requests: median 5 148, p90
52 073, 84.8 % below 20k, 8.0 % above 100k. Modelling a turn as 1000 prefilled
plus 500 generated tokens:

| config | weighted turn |
|---|---|
| A `-nkvo` n21 ub8192 | 32.43 s |
| B swa n25 ub8192 | 35.12 s |
| **C swa n21 ub4096** | **31.19 s** (−3.8 %) |

C shipped. Note this contradicts the 128k sibling, where a large `-ub`
amortises expert streaming and is worth keeping — it still is; this profile
simply cannot afford 8192 without giving up an expert layer, and the layer is
worth more.

The lesson worth keeping: the first comparison varied `-ncmoe` because that was
the obvious knob, and concluded against `--swa-compress` on one data point. The
right question was which resource to spend, and only measuring the third
combination answered it.

### 49.5 All five configurations, and why 128k stays the default

Every number below is from 2026-08-29 on the fixed build, full prefills with
unique prompts:

| config | ~6.4k pp/tg | ~52k pp/tg | ~122k pp/tg |
|---|---|---|---|
| 128k `-nkvo` n19 ub8192 | **1647.5 / 18.90** | 1684.9 / 18.02 | 1359.5 / 16.25 |
| **128k swa n21 ub8192** | 1576.8 / 18.01 | **1693.7** / 17.57 | **1518.9 / 16.79** |
| 256k `-nkvo` n21 ub8192 | 1383.5 / 16.03 | 1486.9 / 15.17 | 1182.1 / 13.91 |
| 256k swa n25 ub8192 | 1332.4 / 14.58 | 1520.2 / 14.31 | 1342.4 / 14.37 |
| 256k swa n21 ub4096 | 1025.1 / 16.65 | 1223.0 / 16.11 | 1106.6 / 15.82 |

Weighted turn (1000 prefilled + 500 generated) against this box's measured
depth distribution — 112 requests, median 5 148, 84.8 % below 20k, 8.0 % above
100k:

| config | weighted turn | vs best |
|---|---|---|
| 128k `-nkvo` n19 | 27.51 s | — |
| **128k swa n21** | **28.61 s** | +4.0 % |
| 256k swa n21 ub4096 | 31.19 s | +13.4 % |
| 256k `-nkvo` n21 | 32.43 s | +17.9 % |
| 256k swa n25 | 35.12 s | +27.7 % |

**262144 costs ~13 % of every turn** — two more expert layers on the host plus
a halved `-ub` — to serve the 8 % of requests that run deep, and the deepest
request ever recorded here was 139 264 tokens. Not worth it while the client
can compact.

Shipped: **128k with `--swa-compress`**, which gives up 4 % against the old
`-nkvo` settings at shallow depth and is the fastest of all five at 122k
(1518.9 pp / 16.79 tg). Hermes is pointed at it with an explicit
`context_length: 131072` override, so it stops budgeting against the model's
advertised 262144 — a mismatch that had been there since long before any of
this.

### 49.6 Re-measured with the raw_k guard off — and why that mattered

Matt caught that `ik_rawk_guard` defaulted **ON** (`getenv(...) == 0 ? 0 : 1`),
so every measured run had it unless `IK_RAWK_GUARD=0` was passed explicitly.
Worse, it is asymmetric: the guard acts on host→device `raw_k` copies, which
only exist with `-nkvo`. So §49.5's table was biased **against** `-nkvo` by
exactly 536 MiB of compute buffer and its throughput cost — and `-nkvo` still
won. The default is now OFF (the hypothesis it tested died in §43.4).

Re-measured clean, median of 3 runs per point, spread 0.1–4.1 %:

| depth | A: `-nkvo` n19 | B: swa n21 | B vs A |
|---|---|---|---|
| ~6.4k | **1711.8 / 19.11** | 1603.8 / 18.12 | pp −6.3 %, tg −5.2 % |
| ~52k | 1671.1 / 18.13 | **1695.8** / 18.14 | pp +1.5 %, tg +0.1 % |
| ~122k | 1369.0 / 16.57 | **1516.8 / 17.05** | pp +10.8 %, tg +2.9 % |

Weighted turn: **A 27.18 s, B 28.35 s (+4.3 %)** — the same verdict as the noisy
pass (+4.0 %), so the guard cost accuracy, not the conclusion.

**Compute buffer scaling** (guard off, same `-ub 8192`):

| context | `-nkvo` | `--swa-compress` |
|---|---|---|
| 131072 | 7 040 | 14 212 |
| 262144 | 11 136 | 21 380 |
| growth | **+4 096** | +7 168 |

This is the durable argument for `-nkvo` on a card this size: its scratch is
half the size and grows at 57 % of the rate, and that headroom buys **two more
expert layers resident on the GPU** (`-ncmoe 19` against 21) — which is where
the 4.3 % comes from. It is not a fallback for the GPU-poor here; it is the
faster configuration for shallow-to-mid agent traffic, and the gap widens with
context length in VRAM terms even as it narrows in throughput.

Soak continues on A: **2 468 649 tokens, 0 aborts** before this benchmark
interruption (9.1 expected without the fix, P = 0.01 %).
