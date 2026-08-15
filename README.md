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
[docs/RESULTS.md §5](docs/RESULTS.md). It ships
**tuned for one specific machine** (an RTX PRO 6000 Blackwell, 96 GiB), but
nothing here is tied to that card: it runs on any CUDA GPU — and, through
ik_llama.cpp, on AMD/ROCm, Vulkan and Apple Metal too — with the split re-tuned
per machine by `./bench.sh`. Adapting it to other hardware (a different NVIDIA
card, less VRAM, multiple GPUs) is a one-line change; see [docs/FAQ.md](docs/FAQ.md).

```bash
./build.sh        # compile ik_llama.cpp for this GPU (once)
./serve.sh        # start the server on the default model
./bench.sh quick  # measure it
```

The server exposes an OpenAI-compatible API and a web UI at
<http://127.0.0.1:8090>. (8090 rather than the usual 8080, which LM Studio and
Docker containers tend to occupy — override with `--port` if you prefer.)

---

## The reference machine

These are the specs the shipped defaults are tuned for. On different hardware
everything still works — you just re-run `./bench.sh` to re-pick the split
(see [docs/FAQ.md](docs/FAQ.md) and [docs/TUNING.md](docs/TUNING.md)).

| resource | reference configuration |
|----------|-----|
| GPU      | NVIDIA RTX PRO 6000 Blackwell Workstation — 96 GiB, `sm_120`, ~1.8 TB/s |
| CPU      | Intel Core Ultra 7 270K Plus — 8 P-cores + 16 E-cores = **24 cores**, no SMT |
| RAM      | 224 GiB DDR5 (2×48 + 2×64 GiB) at 6267 MT/s, dual channel — ~100 GB/s |
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

**DeepSeek-V4-Flash**, MXFP4, served by the `deepseek-v4-flash-128k-kvram`
profile — 131072 context, ~463 tok/s prefill, ~21 tok/s generation. What that
profile does and why is in [docs/RESULTS.md §11](docs/RESULTS.md); the short
version is that the KV cache lives in system RAM so the VRAM it would occupy
goes to expert weights instead.

> **Known issue — do not raise `IK_UBATCH` above 1024.** The server intermittently
> aborts with all-`nan` logits (`Failed to sample token`) under sustained use.
> Six occurrences, every one at `-ub 2048`; none across 1.07 M prefilled tokens of
> the same interactive workload at 512 or 1024. The profile therefore ships
> `-ub 1024`, which costs 3–5 % of prefill — that is where the ~463 above comes
> from rather than the ~484 quoted in §11. The abort follows prefilled *volume*
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
./serve-deepseek-v4-flash-mxfp4-kvram-128k.sh             # fast-prefill variant: 484 tok/s pp (+69 %)
./serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh         # same, MTP: 24 tok/s tg, less prefill
./serve-deepseek-v4-flash-mxfp4-kvram-256k.sh             # same treatment at 256k: 406 pp
./serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh         # 256k + MTP: 20.5 tok/s tg (+26 %)
./serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh           # half-million ctx, KV in RAM, ~16 tok/s
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
- **`serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh`** → `serve.sh deepseek-v4-flash-512k`.
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
├── serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh
│                         wrapper: DeepSeek IQ2XXS all-in-VRAM + MTP, ~87 tok/s
├── serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh
│                         wrapper: DeepSeek MXFP4, experts in DDR5, ~21 tok/s
├── serve-deepseek-v4-flash-mxfp4-kvram-128k.sh
│                         wrapper: fast-prefill 128k variant, 484 tok/s pp
├── serve-deepseek-v4-flash-mxfp4-gpu-cpu-512k.sh
│                         wrapper: DeepSeek MXFP4 at 512k, KV in RAM, ~16 tok/s
├── serve-step-3.7-flash-q8.sh  wrapper: Step-3.7-Flash Q8 quality reference
├── stop.sh               stop any running server (any profile/port)
├── TODO.md               open measurement threads (see RESULTS §8-§10)
├── bench.sh              benchmark harness
├── config/
│   ├── default.env       global defaults, fully annotated
│   └── models/*.env      per-model profiles (q4, q4-r4, q8, deepseek-v4, mxfp4-tuned)
├── lib/common.sh         shared helpers: config, preflight, arg assembly
├── docs/
│   ├── RESULTS.md        every measurement taken (Q4 + Q8)
│   ├── FAQ.md            platform, portability, hardware bottlenecks
│   ├── TUNING.md         why every parameter is set the way it is
│   ├── BENCHMARKING.md   how to measure and how to read the numbers
│   └── TROUBLESHOOTING.md
├── compat/               generated <math.h> shim (CUDA 13 + new glibc)
├── logs/                 server logs, timestamped
├── results/              benchmark reports
└── ik_llama.cpp/         upstream source (git clone)
```

---

## Further reading

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
