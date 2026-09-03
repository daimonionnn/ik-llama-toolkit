# ik-llama-toolkit

**A one-command inference server and benchmark harness for running large
language models that don't fit in your GPU's VRAM.** It keeps as much of the
model as possible on the GPU, spills the rest into system RAM, and tunes that
GPU/CPU split for the fastest inference the machine can give — so you can serve a
200B-parameter model from a single workstation instead of renting a multi-GPU
node.

Built on [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). **Where it
pays off is DeepSeek-class MLA models** — on DeepSeek-V4-Flash MXFP4 it beats
mainline llama.cpp by ~26% on prefill and ~19% on generation, and runs a q8_0
KV cache that mainline segfaults on. On plain GQA MoE models such as
Step-3.7-Flash, mainline is as fast or faster; see
[docs/RESULTS.md §5](docs/RESULTS.md).

> ### ⚙️ Tuned for one card. Runs on many.
>
> **Every number, default and profile in this repository was measured on one
> machine: an NVIDIA RTX PRO 6000 Blackwell (96 GiB) with 244 GiB of DDR5.**
> Treat the shipped `-ncmoe` / `-ub` values as *that machine's answers*, not as
> universal settings — they encode exactly how much VRAM this card has.
>
> **Nothing here is tied to that card.** It runs on any CUDA GPU, and through
> ik_llama.cpp on AMD/ROCm, Vulkan and Apple Metal as well. Moving to different
> hardware is a config change, not a code change:
>
> ```bash
> ./build.sh                 # detects and compiles for your GPU
> ./serve.sh                 # on a smaller card it auto-fits, and says so
> ./bench.sh ncmoe <profile> # then measure the best split for your card
> ```
>
> **You do not have to do anything for this.** If the GPU is not in the 96 GiB
> class, the toolkit drops the profile's pinned `-ncmoe` / `-ot`, falls back to
> `--fit`, and prints a warning telling you so. It decides on *total* VRAM, never
> on free VRAM — otherwise another process holding memory would silently change
> the split and make results irreproducible. `IK_ASSUME_REFERENCE_GPU=1` keeps
> the pinned values; `IK_REFERENCE_VRAM_MIN` moves the threshold.
>
> Auto-fit gets you *running*, not *fast*. `./bench.sh ncmoe` is still what finds
> your card's real floor.
>
> With less VRAM you push more expert layers to the host and lose throughput,
> not function — the whole point of the project is that the model does not have
> to fit. See [docs/FAQ.md](docs/FAQ.md) for a different card, less VRAM or
> multiple GPUs, and [docs/TUNING.md](docs/TUNING.md) for re-deriving the split.
> [docs/VS-DGX-SPARK.md](docs/VS-DGX-SPARK.md) shows how much the *model*
> changes the answer — the same tuning wins on one model and loses on another.

```bash
./build.sh        # compile ik_llama.cpp for this GPU (once)
./serve.sh        # start the server on the default model
./bench.sh quick  # measure it
```

The server exposes an OpenAI-compatible API and a web UI at
<http://127.0.0.1:8090>. (8090 rather than the usual 8080, which LM Studio and
Docker containers tend to occupy — override with `--port` if you prefer.)

---

## What tuning bought on this box

### DeepSeek-V4-Flash MXFP4

At three context sizes, all measured with
`tools/depthbench.sh` and `tools/sweep.sh` on the same machine, same build, same
quant — `max_tokens` 160, temperature 0, a salt unique per request so nothing is
served from cache:

| context | prefill @32k | generation @32k | checkpoints |
|---|---:|---:|---|
| **131072** *(default)* | **1830 tok/s** | 19.3 t/s | on, upstream defaults |
| 262144 | 1637 | 17.4 | on, `--cache-ram 32768` ceiling |
| 524288 | 1721 | 16.3 | off — twice the cost at this size |

The 131072 profile started this work at **486 tok/s**. It is now **3.8× faster**,
and the model, the quant and the answers are unchanged — every gain is placement,
batching or scheduling.

Three findings did nearly all of it:

* **`-rtr` was computing the experts on the wrong processor** (§21). It repacks
  host-resident experts into `MXFP4_R8`, a type with no CUDA kernel, which pins
  that work to the CPU. Dropping it hands the experts back to the GPU and is worth
  **~3×**. This one came from comparing against another toolkit, not from tuning.
* **The micro-batch has to be large enough to amortise weight streaming** (§22.4).
  `-ub 8192` over 2048 is +17 % at depth; `-b` must be raised with it or it is
  silently clamped.
* **An upstream f16 overflow in DSA attention** (§24) was causing an
  intermittent all-`nan` abort roughly every 100–330 k prefilled tokens. Fixed in
  `ff141691`; our checkout predated it by three and a half hours.

Two of those three came from looking outside this repository. The full
measurements, including the negative results and three conclusions that were
measured correctly and read wrongly, are in [docs/RESULTS.md](docs/RESULTS.md).

### Qwen3.8-Flash-Next

A hybrid SSM/attention MoE — 176.9 B parameters, 512 experts with 10 used, and
full attention on only every fourth of its 48 layers. Measured with
`llama-sweep-bench` (RESULTS §51), shallow figures, `-ctk/-ctv q8_0`:

| profile | prefill | generation | @32k | @~76–96k |
|---|---:|---:|---|---|
| **Q4_K_M 131072** *(fastest)* | **3486 tok/s** | **128.9 t/s** | — | 1440 / 62.2 |
| Q4_K_M 262144 | 3342 | 128.6 | — | 1346 / 59.7 |
| **Q8_0 131072** *(default)* | 2303 | 40.6 | 1854 / 35.3 | 1368 / 30.8 |
| Q8_0 262144 | 2163 | 36.9 | 1757 / 32.4 | — |

**The first draft of the Q4 profile, with the settings simply carried over from
DeepSeek (`-ncmoe 13 -ub 4096`), measured 2753 tok/s and 60.6 t/s.** Tuning it
was worth **+27 % prefill and +113 % generation** — and almost all of the second
number came from one knob:

* **Expert residency dominates here, far more than on DeepSeek.** Sweeping
  `-ncmoe` from 13 to 0 gains 12 % prefill but **113 % generation** (60.6 →
  128.9 t/s). Ten of 512 experts of width 640 per token is a wide, thin,
  scattered read — the access pattern PCIe handles worst — so every layer kept
  resident pays far more than the 2.8 % per layer DeepSeek showed in §21.
* **`-ub` behaves oppositely in the two quants.** On Q4 the weight split does not
  move with `-ub`, making it a free prefill knob (take the largest that fits:
  2048 at 128k, 1024 at 256k). On Q8 a smaller `-ub` buys one more resident
  expert layer worth +3.5 % generation and costs **51 % of prefill** — so the
  micro-batch wins.
* **`-ncmoe 0` is not "all on the GPU".** It works on Q4 by luck of where the
  tensors land; the same flag on Q8_0 asks for 126 971 MiB and dies in
  `cudaMalloc`. Q8 needs `-ncmoe 17`.

Note the quant trade this exposes: **Q4_K_M generates 3.2× faster than Q8_0** on
the same card, because Q8 leaves ~94 GiB of experts in host RAM against ~34.

### Against a pair of DGX Sparks

The popular way to run this model is 2× DGX Spark at TP=2. On **prefill this one
card is in their band** — above the poorly-tuned FP8 runs, level with the good
ones, below the NVFP4 record of 2639 tok/s. On **generation they win by 2–4×**,
and it is structural rather than a tuning gap: each node holds half the model in
unified memory at ~273 GB/s, against ~107 GB/s of DDR5 here for the experts that
do not fit in VRAM.

Two things are worth knowing before trusting any such comparison:

* **Published 2× Spark prefill ranges from 176 to 2639 tok/s** — fifteen-fold, on
  identical hardware. Quantisation, engine and tuning decide it, not the machine.
* **The published figure for an RTX PRO 6000 is 344 tok/s**, measured with
  `ds4.c` on a 2-bit quant. This repository gets **1830** out of the same card on
  effectively lossless MXFP4 — 5.3× — so a hardware comparison drawn from that
  number is wrong by more than the hardware difference it is trying to show.

**And the verdict flips by model.** Same card, same repository, opposite result:

| model + quant | this card | closest Spark reference | |
|---|---:|---:|---|
| DeepSeek-V4-Flash MXFP4 | 1830 pp / 19.3 tg | 2× Spark: ~1400–1900 pp / 40–53 tg | **they win tg 2–4×** |
| Qwen3.8 Q4_K_M | **3486 pp / 128.9 tg** | 1× Spark, gpt-oss-120b: 1956 / 60.6 | **we win ~1.8× / ~2.1×** |
| Qwen3.8 Q8_0 | 2303 pp / **40.6 tg** | 2× Spark FP8: ~1000–1500 / ~36–44 *(est.)* | wash on tg, ours on pp |

The Spark column is not like-for-like and cannot be — the quants differ, and for
Qwen the only mature-software data point is a different model of the same class
(gpt-oss-120b, which *fits* in one Spark's 128 GB). Nobody has published FP8
Qwen3.8 on a Spark pair at all, because FP8 measures *slower* than NVFP4 for MoE
on that hardware (52 against 66.9 tok/s on a Spark MoE). Read
[docs/VS-DGX-SPARK.md](docs/VS-DGX-SPARK.md) before quoting any of this.

What decides it is neither the machine nor how much spills into DDR5 — Qwen at
Q8_0 spills more than DeepSeek (95.9 GiB against 63) and still generates twice as
fast. It is **expert geometry**: DeepSeek's 6-of-256 experts on a 4096-wide
residual with a 2048 FFN move 6.49 B active parameters per token, against 2.36 B
for Qwen's 10-of-512 at 2560 × 640. Decode is bandwidth-bound, so the model that
reads fewer bytes per token wins — even at twice the bits per weight.

Tables and sources for both models in [docs/VS-DGX-SPARK.md](docs/VS-DGX-SPARK.md).

---

## The reference machine

These are the specs the shipped defaults are tuned for — **the whole repository
is one machine's tuned answer**, and the `-ncmoe` / `-ub` values in every profile
encode this card's 96 GiB rather than a general truth.

On different hardware everything still works. Fewer gigabytes of VRAM means more
expert layers on the host and less throughput, not a failure to run: the project
exists precisely because the model is bigger than the card. Re-run `./bench.sh
ncmoe` to re-pick the split, or start from a `--fit` profile that sizes itself
(see [docs/FAQ.md](docs/FAQ.md) and [docs/TUNING.md](docs/TUNING.md)).

| resource | reference configuration |
|----------|-----|
| GPU      | NVIDIA RTX PRO 6000 Blackwell Workstation — 96 GiB, `sm_120`, ~1.8 TB/s |
| CPU      | Intel Core Ultra 7 270K Plus — 8 P-cores + 16 E-cores = **24 cores**, no SMT |
| RAM      | 256 GB DDR5 (4×64 GiB) at 6667 MT/s, dual channel — ~107 GB/s; 244 GiB visible |
| Storage  | NVMe, 1.8 TB free |
| OS       | Ubuntu 26.04 LTS (`resolute`), kernel 7.0.0-29-generic, glibc 2.43 |
| Driver   | NVIDIA **595.84** (CUDA runtime 13.2); also verified on **610.43.02** / UMD 13.3 — no change |
| Toolkit  | CUDA 13.3 (`nvcc` V13.3.73), GCC 15.2.0 |

Every number in [docs/RESULTS.md](docs/RESULTS.md) was measured on that software
stack. A driver or CUDA change is enough to move them, so after one run
**`./check-driver-change.sh`** — it reports the new versions, re-probes the two
device attributes RESULTS depends on, and with `--bench` re-measures the shipped
default against its recorded baseline.

The constraint that drives everything is the ratio between the two memory pools:
VRAM is roughly **18× faster** than system RAM here, so every gigabyte of model
that does not fit on the GPU is disproportionately expensive. Almost all of the
tuning in this toolkit is about deciding *what* gets exiled to system RAM — which
is exactly the problem any "model bigger than VRAM" setup faces, on any GPU.

## The default model

**DeepSeek-V4-Flash**, MXFP4, served by the `deepseek-v4-flash-gpu-experts-128k`
profile — 131072 context, **~1850 tok/s prefill at 32k**, ~20 tok/s generation.
The experts that do not fit in VRAM are computed *on the GPU* rather than on the
CPU, which is worth roughly 3x prefill (RESULTS §21–§24). What that
profile does and why is in [docs/RESULTS.md §11](docs/RESULTS.md); the short
version is that the KV cache lives in system RAM so the VRAM it would occupy
goes to expert weights instead.

> **Known issue — do not raise `IK_UBATCH` above 1024.** The server intermittently
> aborts with all-`nan` logits (`Failed to sample token`) under sustained use.
> Six occurrences, every one at `-ub 2048`; none across 1.07 M prefilled tokens of
> the same interactive workload at 512 or 1024. The profile therefore ships
> `-ub 1024`. That cost 3–5 % of prefill when it was chosen; since the GPU moved to
> a full PCIe 5.0 x16 link it no longer does — `-ub 1024` now measures 486 tok/s at
> 32k, above what `-ub 2048` managed before (§20). The abort follows prefilled *volume*
> (roughly one per 100–330 k tokens), not uptime, so a short test proves nothing.
> Cause unknown; the investigation is [RESULTS §19](docs/RESULTS.md) and the open
> threads are TODO item 9.

Four sibling profiles cover the other trade-offs — 256k and 512k context, and
MTP variants that swap prefill for generation. See the wrapper list under
[Usage](#usage).

### The original default: Step-3.7-Flash

Still shipped and still the subject of §1–§5, now behind an explicit
`./serve.sh step-3.7-flash-q4`. Unsloth Dynamic `Q4_K_XL`, from
`$IK_MODELS_ROOT/unsloth/Step-3.7-Flash-GGUF/`.

| property              | value                                       |
|-----------------------|---------------------------------------------|
| architecture          | `step35`                                    |
| layers                | 45 — 3 dense, then 42 MoE                   |
| experts               | 288 total, **8 active** per token           |
| expert FFN width      | 1280 (plus one shared expert per layer)     |
| embedding width       | 4096                                        |
| attention             | 8 KV heads × 128, GQA                       |
| sliding window        | 512 tokens, pattern **1 full : 3 windowed** |
| trained context       | 262 144                                     |
| total / active params | ~220 B / **~7.4 B**                         |
| file size (Q4_K_XL)   | ~114 GiB across 4 shards                    |
| vision                | yes — `mmproj-F32.gguf` ships alongside     |

Two properties of this model drive everything:

1. **~91% of the weights are routed experts.** Attention, norms, embeddings,
   the shared experts and the 3 dense layers together are only ~4.4 GiB. So
   "how much fits on the GPU" is essentially "how many layers of experts fit".

2. **Only 8 of 288 experts run per token.** Generation touches ~7.4 B
   parameters, not 220 B. That is why this is fast at all — but it also means
   expert access is scattered, so the CPU side is bound by memory *latency and
   bandwidth*, not by arithmetic.

`Q8_K_XL` is also configured (`./serve.sh step-3.7-flash-q8`) but is not the
default: at ~230 GiB it pushes ~145 GiB onto the CPU instead of ~25 GiB, for a
quality gain that Unsloth's dynamic Q4 largely already captures.

### Measured performance (this box, GPU idle)

Generation speed is a straight trade against context length, because the KV
cache is large here (full-length on all 45 layers — see
[docs/TUNING.md §3](docs/TUNING.md)) and every gigabyte of cache is a gigabyte
not holding experts:

| context            | KV cache | `-ncmoe` | generation    |
|--------------------|----------|----------|---------------|
| 262 144 (profile default) | 24 GiB | 22    | **~25 tok/s** |
| 131 072            | 12 GiB   | 16       | ~33 tok/s     |
| 65 536             | 6 GiB    | 14       | ~38 tok/s     |
| 262 144, `q4_0` KV | 13 GiB   | 18       | ~30 tok/s     |

**Prefill** is ~208 tok/s at the default. If your work is prefill-heavy (long
prompts, RAG), the `step-3.7-flash-q4-r4` profile serves a copy whose
CPU-resident experts are pre-repacked, lifting prefill to **~246 tok/s (+18%)**
with generation unchanged and no startup penalty — but it is locked to the
262144 / `-ncmoe 22` config (see the profile header for why).

The `step-3.7-flash-q4` profile ships at the **full 262 144 context**. If you
want more speed and don't need the whole window, lower `IK_CTX` and `IK_NCMOE`
together (see the table in [docs/TUNING.md §1](docs/TUNING.md)). Prompt processing runs at a few
hundred tok/s regardless — inherent to a model larger than VRAM.

---

## Install

### 1. CUDA

**This is the one prerequisite that is not automatic.** The RTX PRO 6000
Blackwell is compute capability 12.0, and `nvcc` only gained `sm_120` in CUDA
12.8. Ubuntu's stock `nvidia-cuda-toolkit` is 12.4 and *cannot* produce code
that runs on this GPU.

```bash
sudo apt install -y cuda-toolkit-13-1
```

`build.sh` probes every toolkit it can find, compiles a test kernel for
`sm_120`, and refuses to build against one that fails — so you will get a clear
error rather than a binary that dies at load time.

**Any CUDA ≥ 12.8 is fine, 12 or 13.** Mainline llama.cpp built with CUDA 13.x
collapses on Blackwell past 8192 context, and `build-cuda12.sh` builds this
engine with CUDA 12.8 in a container as insurance — but the two measure
identically here, so it is a check that was run, not a step you need. Details
in [docs/TUNING.md §9](docs/TUNING.md).

### 2. Build

```bash
./build.sh
```

Takes 15–30 minutes cold, mostly CUDA kernels. It also picks a host compiler
`nvcc` accepts (GCC 15 is the system default and CUDA may reject it; GCC 13 is
the fallback).

```bash
./build.sh --clean     # rebuild from scratch
./build.sh --update    # git pull ik_llama.cpp, then rebuild
```

### 3. Free the GPU

LM Studio and Ollama both keep models resident long after their last request.
`serve.sh` checks for this and tells you who is holding VRAM:

```
warn other processes are holding GPU memory:
       1052139  55552 MiB  .../llama-server
       2276218  21922 MiB  /usr/local/lib/ollama/llama-server
```

Quit them, or let the toolkit do it:

```bash
IK_KILL_SQUATTERS=1 ./serve.sh
```

This matters more than it looks: `--fit` sizes the GPU/CPU split from *free*
VRAM at launch. Starting with 20 GiB free instead of 95 GiB silently pushes
~28 extra layers of experts onto the CPU and roughly halves your token rate.

---

## Usage

```bash
./serve.sh                              # default: DeepSeek fast-prefill 128k (484 pp / ~21 tg)
./serve.sh step-3.7-flash-q4            # the old default: Step-3.7-Flash Q4_K_XL, ~26 tok/s
./serve.sh mxfp4-tuned                  # DeepSeek-V4-Flash MXFP4, the tuned profile
./serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh   # fastest DeepSeek: all in VRAM + MTP, ~87 tok/s
./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh           # lossless DeepSeek, experts in DDR5, ~21 tok/s
./serve-deepseek-v4-flash-mxfp4-kvram-128k.sh             # experts on the CPU: 486 tok/s pp
./serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh       # experts on the GPU: 1794 tok/s pp (3.7x)
./serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh         # same, MTP: 24 tok/s tg, less prefill
./serve-deepseek-v4-flash-mxfp4-kvram-256k.sh             # same treatment at 256k: 406 pp
./serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh         # 256k + MTP: 20.5 tok/s tg (+26 %)
./serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh       # half-million ctx: 1721 tok/s pp at 32k
./serve-step-3.7-flash-q8.sh            # Step-3.7-Flash Q8_K_XL quality reference, ~13 tok/s
./serve.sh --list                       # what profiles exist
./serve.sh --ctx 131072                 # override context length
./serve.sh --port 9000 --host 0.0.0.0   # expose on the LAN
./serve.sh --dry-run                    # print the command, run nothing
./serve.sh -- --verbose                 # pass extra flags to llama-server
./stop.sh                               # stop whatever this toolkit is serving
./stop.sh --all                         # also free VRAM from LM Studio / Ollama
```

A running server is stopped with **Ctrl+C** in its terminal, or from anywhere
with **`./stop.sh`** — it finds this toolkit's server by any port/profile,
stops it cleanly, and reports the freed VRAM. `./stop.sh --all` additionally
evicts a GPU-resident LM Studio or Ollama model.

The `serve-*.sh` scripts are thin wrappers that free the GPU first and forward
any `serve.sh` flag. Their names spell out quant, placement and context:

- **`serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh`** →
  `serve.sh deepseek-v4-flash-mtp`. The ~81 GiB antirez 2-bit quant runs
  **entirely on the GPU** (no DDR5 spill) with a 5.5 GiB MTP draft on top:
  **~87 tok/s at 65536 ctx**, the fastest coherent DeepSeek config here. 131072
  does not fit alongside the draft.
- **`serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh`** →
  `serve.sh deepseek-v4-flash-gpu-experts-256k`. The 262144 sibling of the wrapper
  below, converted 2026-08-19: **1101 / 1637 / 1259 tok/s prefill at 4k / 32k /
  128k**, 2.4–3.5× the `-rtr` version it replaces, for ~10 % less generation
  (RESULTS §27.3). Needs `--n-cpu-moe 21` rather than 19, because the compute
  buffer scales with context as well as micro-batch — 11 136 MiB here against
  7040 at 131072 (§27.2). `deepseek-v4-flash-256k-kvram` is the same context on
  the `-rtr` path, kept as the fallback.
- **`serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh`** →
  `serve.sh deepseek-v4-flash-gpu-experts-128k`. **A different principle from
  every other wrapper here.** The others decide where weights *live*; this decides
  who *computes* them. The overflow experts still sit in host RAM, but they are
  shipped across PCIe each pass and the GEMM runs **on the GPU** — which is what
  happens as soon as `-rtr` is not passed, because `-rtr` repacks them into a
  CPU-only type. **1376 tok/s prefill at 4k and 1529 at 32k, roughly 3× the kvram
  profile** (RESULTS §21), for 1–4 % less generation and no quality trade. Needs a
  wide PCIe link and a GPU faster than the CPU at quantised GEMM; on a narrow link
  the `-rtr` profiles win instead. Now swept (RESULTS §22): `--n-cpu-moe 19`,
  `-ub 8192`, `-b 8192`, which is **1329 / 1794 / 1328 tok/s at 4k / 32k / 128k**
  against the kvram profile's 478 / 486 / 437 — 2.8x, 3.7x and 3.0x — for 3-7 %
  less generation. Against `multi-gpu-llm-toolkit` on the same card and the same
  file it wins at depth and loses shallow — **+12.5 % at 128k, -31 % at 4k**
  (RESULTS §23), because `-ub 8192` needs depth before it pays. Not the default yet: the §19 abort has not been re-tested in
  this path, and that soak is what is still owed.
- **`serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh`** → `serve.sh deepseek-v4-flash`.
  The ~146 GiB MXFP4 quant (effectively lossless QAT) at 131072 ctx; `--fit`
  fills VRAM to ~93.8 GiB (margin tuned down to 4096 for this context, worth
  +7 % — see RESULTS §8) and the remaining ~52 GiB of experts run **on the CPU
  out of DDR5**, which pins generation to **~21 tok/s**.
- **`serve-deepseek-v4-flash-mxfp4-kvram-128k.sh`** → `serve.sh deepseek-v4-flash-128k-kvram`.
  Same quant, same context, different placement: the KV goes to RAM (`-nkvo`),
  the freed VRAM goes to experts (`--n-cpu-moe 17`), the CPU experts are
  repacked for AVX2/VNNI (`-rtr`), and the micro-batch that this unlocks does the
  rest — **484 tok/s prefill (+69 %) and ~21 tok/s generation** vs the `--fit`
  wrapper, prompt cache intact (RESULTS §11). The trade is robustness: manual
  placement, ~1.6 GiB VRAM headroom, and `-rtr` re-reads the model each start.
  Prefer the `--fit` wrapper for anything that must come up unattended.
- **`serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh`** → the same with MTP
  speculative decoding: **24.1 tok/s generation against 434 prefill**, versus
  21.3 / 499 without (RESULTS §14). A straight prefill-for-generation swap on
  identical weights — long prompts want the wrapper above, long answers want
  this one.
- **`serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh`** → `serve.sh deepseek-v4-flash-512k`.
  The same quant at **524288 context**, with the placement inverted: the KV goes
  to RAM (`-nkvo`) and the VRAM it frees goes to experts (`--n-cpu-moe 19`,
  no `--fit`). Worth **+22 % generation and +35 % prefill** over `--fit` at that
  context — see RESULTS §10. A full 500k prefill still takes ~52 minutes, and
  the prompt cache is off, so this is for long single-shot contexts, not chat.

  All three default to **port 8090** (8080 is usually taken — LM Studio's API
  server, or a Docker container mapping it)
  and enable MLA + the fused DSA indexer.
- **`serve-step-3.7-flash-q8.sh`** → `serve.sh step-3.7-flash-q8`, the ~195 GiB
  quality reference. e.g. `./serve-step-3.7-flash-q8.sh --ctx 65536`.

Query it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"step-3.7-flash","messages":[{"role":"user","content":"Hi"}]}'
```

### Benchmarks

```bash
./bench.sh quick      # pp512/2048/8192 + tg128            ~5 min
./bench.sh sweep      # throughput vs context depth        ~10 min
./bench.sh threads    # find the best -t for this CPU      ~10 min
./bench.sh ncmoe      # find the best expert split         ~25 min
./bench.sh batch      # find the best -b / -ub             ~15 min
./bench.sh rtr        # is run-time repack worth it?       ~15 min
./bench.sh full       # all of the above                   ~80 min
```

Reports are written to `results/<profile>-<mode>-<timestamp>.md` with the full
hardware and git context recorded in the header. See
[docs/BENCHMARKING.md](docs/BENCHMARKING.md) for how to read them.

---

## Configuration

Settings resolve lowest-to-highest:

```
config/default.env  <  config/models/<profile>.env  <  environment  <  CLI flag
```

So a one-off experiment never needs a file edit:

```bash
IK_CTX=131072 IK_THREADS=8 ./serve.sh
```

When a benchmark finds something better, write it into the profile to make it
permanent.

### The settings that matter most

| variable                 | default         | what it does |
|--------------------------|-----------------|-----|
| `IK_CTX`                 | `262144`        | Context length. Drives KV-cache size, which drives everything |
| `IK_NCMOE`               | `22`            | Keep experts of the first N layers on CPU (the tuned split for 262 144) |
| `IK_FIT`                 | `1`             | Fallback if `IK_NCMOE` is unset: let ik_llama size the split itself |
| `IK_FIT_MARGIN`          | `2048`          | MiB of VRAM to leave free (only used by `--fit`) |
| `IK_OT`                  | *(unset)*       | Full manual control via `-ot` regexes |
| `IK_CTK` / `IK_CTV`      | `q8_0`          | KV cache precision (`q4_0` frees ~11 GiB at 262k) |
| `IK_THREADS`             | `24`            | CPU threads for generation (all cores; generation is flat past 12 — see TUNING §2) |
| `IK_THREADS_BATCH`       | `24`            | CPU threads for prompt processing — **worth +32% prefill over 12** |
| `IK_MODELS_ROOT`         | `$HOME/.lmstudio/models` | Where the GGUFs live. Profiles are written against this, so the toolkit runs on any machine — `IK_MODELS_ROOT=/mnt/models ./serve.sh` |
| `IK_BATCH` / `IK_UBATCH` | `4096` / `1024` | Prefill batch sizes. **Do not raise `IK_UBATCH`** — see the known issue above |
| `IK_RTR`                 | `0`             | Repack CPU experts — faster prefill, but disables mmap |
| `IK_SER`                 | *(unset)*       | Use fewer than 8 experts. Faster, changes output |

The three placement modes (`IK_OT`, `IK_NCMOE`, `IK_FIT`) are mutually
exclusive and checked in that order — so the default profile's `IK_NCMOE=22`
wins over `IK_FIT=1`. **`IK_NCMOE` is context-specific**: if you lower `IK_CTX`
you can lower it too (more experts fit on the GPU → faster); if you raise
`IK_CTX` past the default you must raise it or switch to `IK_FIT=1`, or the
server OOMs. The measured splits are tabulated in
[docs/TUNING.md §1](docs/TUNING.md). Full annotated list in
[`config/default.env`](config/default.env).

---

## Layout

```
ik-llama-toolkit/
├── build.sh              compile ik_llama.cpp (CUDA + host compiler probing)
├── build-cuda12.sh       same, with CUDA 12.8 in a container (optional; see TUNING §9)
├── serve.sh              one-command server launch
├── stop.sh               stop any running server (any profile/port)
│
│  DeepSeek MXFP4, experts computed ON THE GPU -- the tuned line (RESULTS §21-§29)
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh
│                         the default: 1830 tok/s pp / 19.3 tg at 32k
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh
│                         1637 / 17.4 at 32k, checkpoints capped at 32 GiB
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh
│                         1721 / 16.3 at 32k, checkpoints off (they cost 28 % here)
│
│  Same model on the CPU path (-rtr). Kept as fallbacks: a different code path
├── serve-deepseek-v4-flash-mxfp4-kvram-128k.sh      486 / 21.7 at 32k
├── serve-deepseek-v4-flash-mxfp4-kvram-256k.sh
├── serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh  27 tg at 4k -- MTP, the fastest
├── serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh  generation here; will not
│                                                    convert (RESULTS §27.4)
├── serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh    --fit placement, unattended
│
│  Other models
├── serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh
│                         DeepSeek IQ2XXS all-in-VRAM + MTP, ~87 tok/s
├── serve-step-3.7-flash-q8.sh    Step-3.7-Flash Q8, the quality reference
│
├── tools/                the measurement harness -- RESULTS §17 onwards is its output
│   ├── depthbench.sh     prefill + generation vs prompt depth, one profile
│   ├── sweep.sh          the same across configurations, one server per arm
│   ├── stress.sh         drive short prompts hard, to reproduce the §19 abort fast
│   └── ckpt-value.sh     do context checkpoints pay off on YOUR traffic? (§30)
├── bench.sh              older llama-bench harness (§1-§16)
│
├── config/
│   ├── default.env       global defaults, fully annotated
│   └── models/*.env      per-model profiles
├── lib/common.sh         shared helpers: config, preflight, arg assembly
├── docs/
│   ├── RESULTS.md        every measurement taken, including the negative ones
│   ├── VS-DGX-SPARK.md   this card vs 2x DGX Spark, DeepSeek and Qwen
│   ├── TUNING.md         why every parameter is set the way it is
│   ├── BENCHMARKING.md   how to measure and how to read the numbers
│   ├── FAQ.md            platform, portability, hardware bottlenecks
│   ├── TROUBLESHOOTING.md
│   └── external/         what was sent upstream, and every crash record
├── compat/               generated <math.h> shim (CUDA 13 + new glibc)
├── logs/                 server logs, timestamped
├── results/              benchmark reports (the 16 the docs cite are tracked)
└── ik_llama.cpp/         upstream source (git clone, not vendored here)
```

---

## Further reading

- [docs/VS-DGX-SPARK.md](docs/VS-DGX-SPARK.md) — how this box compares to 2× DGX
  Spark on DeepSeek-V4-Flash and Qwen3.8-Flash-Next (the verdict flips between
  them), and why published numbers vary fifteen-fold on identical hardware
- [docs/RESULTS.md](docs/RESULTS.md) — every measurement taken for Q4 and Q8:
  the split sweeps, thread scaling, rtr/R4, PCIe, and the Q4-vs-Q8 comparison
- [docs/FAQ.md](docs/FAQ.md) — platform & portability: other NVIDIA cards, AMD
  /ROCm, multi-GPU, and why this box's setup was bumpy
- [docs/TUNING.md](docs/TUNING.md) — the arithmetic behind the defaults, and
  what to change when the model or context changes
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) — methodology, interpreting
  `pp`/`tg`, and locking in what you find
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — OOM, gibberish output,
  slow loads, `sm_120` errors
- [ik_llama.cpp parameter reference](ik_llama.cpp/docs/parameters.md) — upstream
  documentation for every flag

## License

[MIT](LICENSE).

This repository contains only its own scripts, profiles and documentation.
`ik_llama.cpp/` is **not** vendored here — `build.sh` clones it, and it carries
its own MIT licence from the ggml authors. The one piece of derived material is
[`docs/external/ds4-blackwell-discrete-fixes.patch`](docs/external/ds4-blackwell-discrete-fixes.patch),
a diff against [antirez/ds4](https://github.com/antirez/ds4), which is MIT as
well. So everything here is MIT-compatible in both directions.
