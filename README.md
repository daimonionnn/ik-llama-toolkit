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

**To keep it running unattended, use systemd rather than a shell.** This box
ships a user unit that survives logout and reboot:

```bash
systemctl --user start   ik-llama-server.service     # or stop / restart
systemctl --user enable  ik-llama-server.service     # start at boot
systemctl --user is-active ik-llama-server.service
journalctl --user -u ik-llama-server.service -f
```

It deliberately pins no profile: `ExecStart` runs `serve-host-0.0.0.0.sh`, which
honours `IK_PROFILE` from [`config/default.env`](config/default.env), so changing
the repository default changes what the unit serves. **Note that wrapper binds
every interface with no authentication** — set `Environment=IK_API_KEY=...` in the
unit, or point `ExecStart` at `./serve.sh` to stay on loopback. A copy of the
unit is in [`config/systemd/ik-llama-server.service`](config/systemd/ik-llama-server.service).

Check `systemctl --user is-active` before starting a server by hand: two of them
collide over port 8090 and over VRAM.

---

## The default model: DeepSeek-V4-Flash

**DeepSeek-V4-Flash-0731 in MXFP4** — lmstudio-community's repack of the QAT
weights, so effectively lossless — served by the
`deepseek-v4-flash-gpu-experts-128k` profile at 131072 context:

| depth | prefill | generation |
|---|---:|---:|
| ~20k | **1 850 tok/s** | **19.9 t/s** |
| ~122k | 1 720 | 19.1 |

(RESULTS §49.10–§49.11, 2026-09-04, `tools/depthbench.sh`, temperature 0, a
salt per request so nothing is served from cache, the card at its 600 W
maximum. At the 400 W cap this machine normally runs, prefill is 4–5 % lower
— 1 777 / 1 627 — and generation the same, §49.12.) `./serve.sh` with no
arguments starts it; the profile is `IK_PROFILE` in
[`config/default.env`](config/default.env).

The model is ~146 GiB against 96 GiB of VRAM, so the routed experts of 19 of
its 43 layers live in host RAM. What makes the profile fast is *who computes
them*: they are streamed across PCIe every micro-batch and the GEMM runs on the
GPU — which is what happens as soon as `-rtr` is not passed, because `-rtr`
repacks them into a CPU-only type (§21). The KV cache goes to host RAM too
(`-nkvo`): MLA's latent is small and DeepSeek Sparse Attention reads only ~512
positions of it, and the VRAM this frees holds two more expert layers (§49.6).
`-ub 8192` amortises the streaming (§22.4). The VRAM is full to within about
2 GiB — and that is headroom, not room for another layer: `-ncmoe 18` loads and
dies in the first u-batch, because ≈ 3.5 GiB of runtime allocations sit outside
the buffers the load log reports (§49.11).

**It runs on a locally patched `ik_llama.cpp`** — the SWA-mask clamp that ended
the NaN aborts and the mask sharing that shrank the compute buffer, the last two
findings below. The diffs and their apply order are in
[`docs/external/patches/`](docs/external/patches/README.md); `build.sh` does
*not* apply them, so re-apply after `./build.sh --update`. The load log's
`CUDA0 compute buffer size = 4744.03 MiB` is the proof they are active (the
unpatched graph reports 7040.03). Soak: **3 046 843 prefilled tokens with zero
aborts** after the clamp, where 12.3 were expected at the old rate; the mask
patch has served since 2026-09-04 and real traffic is its soak.

### DeepSeek profiles

| profile | context | placement | measured pp / tg | when |
|---|---:|---|---|---|
| **`deepseek-v4-flash-gpu-experts-128k`** *(default)* | 131072 | `-nkvo -ncmoe 19 -ub 8192` | **1 850 / 19.9** at 20k, 1 720 / 19.1 at 122k | everything — the deepest request ever seen here was 139k |
| `deepseek-v4-flash-gpu-experts-256k` | 262144 | `--swa-compress -ncmoe 21 -ub 4096` | 1 223 / 16.1 at 52k, 1 107 / 15.8 at 122k (§49.4) | when the window is really needed; costs ~13 % of every turn |
| `deepseek-v4-flash-512k` | 524288 | `-nkvo -ncmoe 25 -ub 8192`, checkpoints off | 1 721 / 16.3 at 32k, 1 335 / 14.9 at 128k (§29) | long single shots, not chat: checkpoints and prompt cache are off (54 GiB of RAM and 26 % of generation at this size, §9.3), so a re-sent conversation re-prefills |
| `deepseek-v4-flash-mtp` | 65536 | antirez IQ2XXS, entirely in VRAM, + MTP draft | **~87 t/s** generation (§7) | when 2-bit quality is acceptable; the fastest coherent DeepSeek here |

The 256k and 512k figures predate the mask patch, which cut their compute
buffers to 3.1 and 7.3 GiB; both are due a re-placement (TODO #17).

Eight further profiles are **legacy**, kept so RESULTS §5–§16 can be re-run
rather than reconstructed: the `-rtr` family (`deepseek-v4-flash-128k-kvram`,
`-128k-kvram-mtp`, `-256k-kvram`, `-256k-kvram-mtp`), which computes the
experts on the CPU at ~480 tok/s prefill against ~1 850 (§21); the `--fit`
profiles (`deepseek-v4-flash`, `mxfp4-tuned`); the `-ub 1024` experiment; and
`step-3.7-flash-q4-r4`, whose model file was deleted. `./serve.sh --list` groups
them separately, and the `# LEGACY:` line in each header says why. Their
measurements are in the headers and in RESULTS; nothing here recommends them.

### How it got here

The profile started at **486 tok/s** prefill. It is now **3.8× faster** with the
same model, the same quant and the same answers — every gain is placement,
batching, scheduling or a graph fix. Four findings did nearly all of it:

* **`-rtr` was computing the experts on the wrong processor** (§21). It repacks
  host-resident experts into `MXFP4_R8`, a type with no CUDA kernel, which pins
  that work to the CPU. Dropping it hands the experts back to the GPU and is worth
  **~3×**. This one came from comparing against another toolkit, not from tuning.
* **The micro-batch has to be large enough to amortise weight streaming** (§22.4).
  `-ub 8192` over 2048 is +17 % at depth; `-b` must be raised with it or it is
  silently clamped.
* **An intermittent all-`nan` abort every 100–330k prefilled tokens** kept the
  micro-batch at 1024 for weeks. The cause was a malformed SWA mask — the window
  view was anchored on the padded KV length rather than on `kv_head`, so leading
  rows could arrive entirely `-inf` and flash-attention turned them into NaN. A
  one-line clamp, zero aborts since (§49.2, upstream
  [#2344](https://github.com/ikawrakow/ik_llama.cpp/issues/2344), closed).
* **The compute buffer was 105 copies of three masks** (§49.9–§49.11). Every
  layer re-viewed the same host-side masks and the scheduler copied each view
  separately, by its span — 22 GiB at one placement. Sharing them is 3 copies,
  7 040 → 4 744 MiB on the shipped profile, and the bandwidth it frees is
  **+24 % prefill / +10 % generation at 122k**. Found because upstream said our
  buffer numbers did not pass the smell test; offered back in the same issue.

The full measurements, including the negative results and the conclusions that
were measured correctly and read wrongly, are in
[docs/RESULTS.md](docs/RESULTS.md).

---

## The other models

### Qwen3.8-Flash-Next

A hybrid SSM/attention MoE — 176.9 B parameters, 512 experts with 10 used, and
full attention on only every fourth of its 48 layers. The Q8_0 profile was the
default for one day (2026-09-03, for Hermes); DeepSeek took the slot back with
the mask patch. Measured with `llama-sweep-bench` (RESULTS §51), shallow figures,
`-ctk/-ctv q8_0`:

| profile | prefill | generation | @32k | @~76–96k |
|---|---:|---:|---|---|
| **`qwen38-flash-next-q4km-128k`** *(fastest)* | **3486 tok/s** | **128.9 t/s** | — | 1440 / 62.2 |
| `qwen38-flash-next-q4km-256k` | 3342 | 128.6 | — | 1346 / 59.7 |
| `qwen38-flash-next-q8-128k` | 2303 | 40.6 | 1854 / 35.3 | 1368 / 30.8 |
| `qwen38-flash-next-q8-256k` | 2163 | 36.9 | 1757 / 32.4 | — |

The first draft of the Q4 profile, with the settings carried over from DeepSeek
(`-ncmoe 13 -ub 4096`), measured 2753 tok/s and 60.6 t/s; tuning was worth
+27 % prefill and **+113 % generation**, almost all of it from one knob.
**Expert residency dominates here, far more than on DeepSeek**: sweeping
`-ncmoe` from 13 to 0 gains 12 % prefill but 113 % generation, because ten of
512 experts of width 640 per token is a wide, thin, scattered read — the access
pattern PCIe handles worst. (`-ncmoe 0` is not "all on the GPU": it works on Q4
by luck of where the tensors land, and on Q8_0 dies in `cudaMalloc`; Q8 needs
17.) **`-ub` behaves oppositely in the two quants** — a free prefill knob on Q4,
while on Q8 a smaller one buys a resident expert layer worth +3.5 % generation
for 51 % of prefill. And **Q4_K_M generates 3.2× faster than Q8_0** on the same
card, because Q8 leaves ~94 GiB of experts in host RAM against ~34.

### Step-3.7-Flash — the original default

Where the toolkit started (RESULTS §1–§5), still shipped as
`./serve.sh step-3.7-flash-q4`: Unsloth Dynamic `Q4_K_XL`, ~220 B parameters
with ~7.4 B active, 288 experts with 8 used, GQA rather than MLA. It trains to
262144 and the profile serves that, at `-ncmoe 22` for **~25 tok/s** generation
and ~208 tok/s prefill; every halving of the context buys a few tokens per
second (the table is in [docs/TUNING.md §1](docs/TUNING.md)). Its KV cache is
full-length on all 45 layers, which is why context is expensive here and cheap
on the MLA models. `step-3.7-flash-q8` is the ~195 GiB quality reference at
~13 tok/s. This is also the model on which mainline llama.cpp is as fast or
faster — see the intro.

### Against a pair of DGX Sparks

The popular way to run these models is 2× DGX Spark at TP=2, and **the verdict
flips by model**:

| model + quant | this card | closest Spark reference | |
|---|---:|---:|---|
| DeepSeek-V4-Flash MXFP4 | 1850 pp / 19.9 tg | 2× Spark: ~1400–1900 pp / 40–53 tg | **they win tg 2–4×** |
| Qwen3.8 Q4_K_M | **3486 pp / 128.9 tg** | 1× Spark, gpt-oss-120b: 1956 / 60.6 | **we win ~1.8× / ~2.1×** |
| Qwen3.8 Q8_0 | 2303 pp / **40.6 tg** | 2× Spark FP8: ~1000–1500 / ~36–44 *(est.)* | wash on tg, ours on pp |

What decides it is expert geometry, not the machine: DeepSeek moves 6.49 B
active parameters per token, Qwen 2.36 B, and decode is bandwidth-bound.
Published 2× Spark prefill for DeepSeek ranges from 176 to 2639 tok/s on
identical hardware, so read [docs/VS-DGX-SPARK.md](docs/VS-DGX-SPARK.md) before
quoting any of this.

---

## The reference machine

The specs the shipped defaults are tuned for. On different hardware everything
still works — fewer gigabytes of VRAM means more expert layers on the host and
less throughput, not a failure to run.

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

**Any CUDA ≥ 12.8 is fine, 12 or 13.** `build-cuda12.sh` builds the engine with
CUDA 12.8 in a container as insurance against the CUDA-13 collapse mainline
llama.cpp shows on Blackwell — the two measure identically here, so it is a
check that was run, not a step you need ([docs/TUNING.md §9](docs/TUNING.md)).

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

The clone is gitignored, so the local patches live only in its working tree:
`--update` warns when it sees them, and they are re-applied by hand from
[`docs/external/patches/`](docs/external/patches/README.md) afterwards.

### 3. Free the GPU

LM Studio and Ollama both keep models resident long after their last request.
`serve.sh` checks for this and prints who is holding VRAM, with PIDs and sizes;
quit them, or let the toolkit do it with `IK_KILL_SQUATTERS=1 ./serve.sh` (the
wrappers set it). It matters more than it looks: a pinned profile simply fails
to load, and `--fit` sizes the split from *free* VRAM at launch — starting with
20 GiB free instead of 95 GiB silently pushes ~28 extra layers of experts onto
the CPU and roughly halves your token rate.

---

## Usage

```bash
./serve.sh                                      # the default: DeepSeek-V4-Flash MXFP4 at 131072, ~1850 pp / 19.9 tg
./serve.sh deepseek-v4-flash-gpu-experts-256k   # any profile by name; --list shows them all
./serve.sh qwen38-flash-next-q4km-128k          # the fastest thing here: 3486 pp / 128.9 tg, 4-bit
./serve.sh step-3.7-flash-q4                    # the original default, ~25 tg
./serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh   # wrapper: frees the GPU, then the same as line 1
./serve.sh --list                       # what profiles exist, legacy ones grouped apart
./serve.sh --ctx 65536                  # override context length
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

The `serve-*.sh` scripts are thin wrappers: each frees the GPU
(`IK_KILL_SQUATTERS=1`), takes port 8090, and `exec`s `serve.sh <profile>`,
forwarding any flag. Their names spell out model, quant, placement and context;
[Layout](#layout) maps every wrapper to its profile, and the profile's header in
[`config/models/`](config/models/) carries the measurements that justify each
setting.

Query it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-gpu-experts-128k","messages":[{"role":"user","content":"Hi"}]}'
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
[docs/BENCHMARKING.md](docs/BENCHMARKING.md) for how to read them. The DeepSeek
numbers above come from `tools/depthbench.sh` and `tools/sweep.sh`, which
measure a real server at real prompt depths rather than `llama-bench`.

---

## Configuration

Settings resolve lowest-to-highest:

```
config/default.env  <  config/models/<profile>.env  <  environment  <  CLI flag
```

So a one-off experiment never needs a file edit:

```bash
IK_CTX=65536 IK_THREADS=8 ./serve.sh
```

When a benchmark finds something better, write it into the profile to make it
permanent.

### The settings that matter most

| variable                 | default         | what it does |
|--------------------------|-----------------|-----|
| `IK_CTX`                 | `65536`         | Context length. Drives KV-cache size, which drives everything. Every profile overrides it — the default profile uses 131072 |
| `IK_NCMOE`               | *(unset)*       | Keep experts of the first N layers on CPU. Set per profile, not globally — the default profile pins 19, and 18 does not fit (§49.11) |
| `IK_FIT`                 | `1`             | Fallback if `IK_NCMOE` is unset: let ik_llama size the split itself |
| `IK_FIT_MARGIN`          | `2048`          | MiB of VRAM to leave free (only used by `--fit`) |
| `IK_OT`                  | *(unset)*       | Full manual control via `-ot` regexes |
| `IK_CTK` / `IK_CTV`      | `q8_0`          | KV cache precision. The DeepSeek profiles override to `f16` — the MLA latent is already compressed; on Step-3.7 `q4_0` frees ~11 GiB at 262k |
| `IK_THREADS`             | `24`            | CPU threads for generation (all cores; generation is flat past 12 — see TUNING §2) |
| `IK_THREADS_BATCH`       | `24`            | CPU threads for prompt processing — **worth +32% prefill over 12** |
| `IK_MODELS_ROOT`         | `$HOME/.lmstudio/models` | Where the GGUFs live. Profiles are written against this, so the toolkit runs on any machine — `IK_MODELS_ROOT=/mnt/models ./serve.sh` |
| `IK_BATCH` / `IK_UBATCH` | per profile | Prefill batch sizes. Bounded by the compute buffer, which scales with `-ub` × `n_kv` — raising it can stop the profile loading |
| `IK_RTR`                 | `0`             | Repack CPU experts. On DeepSeek it pins them to the CPU — 3× slower prefill (§21); on Step-3.7 it was +16 %. Disables mmap either way |
| `IK_SER`                 | *(unset)*       | Use fewer than 8 experts. Faster, changes output |

The three placement modes (`IK_OT`, `IK_NCMOE`, `IK_FIT`) are mutually
exclusive and checked in that order, so a profile's pinned `IK_NCMOE` wins over a
bare `IK_FIT=1` in the profile. An **explicit** `IK_FIT=1` in the environment
does override it (`load_config` records what you asked for before the profile is
sourced), and so does the automatic fallback on a GPU outside the 96 GiB class.
**`IK_NCMOE` is context-specific**: if you lower `IK_CTX` you can lower it too
(more experts fit on the GPU → faster); if you raise `IK_CTX` past the default
you must raise it or switch to `IK_FIT=1`, or the server OOMs. The measured
splits are tabulated in [docs/TUNING.md §1](docs/TUNING.md). Full annotated list
in [`config/default.env`](config/default.env).

---

## Layout

```
ik-llama-toolkit/
├── build.sh              compile ik_llama.cpp (CUDA + host compiler probing)
├── build-cuda12.sh       same, with CUDA 12.8 in a container (optional; see TUNING §9)
├── check-driver-change.sh  did a driver/CUDA update move the numbers? (--bench re-measures)
├── serve.sh              one-command server launch; --list shows the profiles
├── stop.sh               stop any running server (any profile/port)
│
│  Wrappers: free the GPU, then exec serve.sh <profile>.
│  DeepSeek MXFP4, experts computed ON THE GPU -- the tuned line (RESULTS §21-§29, §49)
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-128k.sh      THE DEFAULT: 1850 pp / 19.9 tg at 20k
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-256k.sh      1223 / 16.1 at 52k, --swa-compress
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-512k.sh      1721 / 16.3 at 32k, caches off
├── serve-deepseek-v4-flash-antirez-IQ2XXS-gpu-mtp-65k.sh  2-bit, all in VRAM + MTP, ~87 tg
│
│  Legacy -- the CPU path (-rtr), --fit, and the -ub 1024 experiment; ~480 pp
├── serve-deepseek-v4-flash-mxfp4-kvram-128k.sh            -> deepseek-v4-flash-128k-kvram
├── serve-deepseek-v4-flash-mxfp4-kvram-mtp-128k.sh        -> deepseek-v4-flash-128k-kvram-mtp
├── serve-deepseek-v4-flash-mxfp4-kvram-256k.sh            -> deepseek-v4-flash-256k-kvram
├── serve-deepseek-v4-flash-mxfp4-kvram-mtp-256k.sh        -> deepseek-v4-flash-256k-kvram-mtp
├── serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh          -> deepseek-v4-flash (--fit)
├── serve-deepseek-v4-flash-mxfp4-gpu-experts-128k-ub1024.sh
│
│  Qwen3.8-Flash-Next (RESULTS §51) and Step-3.7-Flash
├── serve-qwen38-flash-next-q4km-128k.sh   3486 pp / 128.9 tg, 4-bit -- the fastest of all
├── serve-qwen38-flash-next-q4km-256k.sh   3342 / 128.6
├── serve-qwen38-flash-next-q8-128k.sh     2303 / 40.6, the quality reference
├── serve-qwen38-flash-next-q8-256k.sh     2163 / 36.9
├── serve-step-3.7-flash-q8.sh             Step-3.7-Flash Q8_K_XL, ~13 tg
│
├── tools/                the measurement harness -- RESULTS §17 onwards is its output
│   ├── depthbench.sh     prefill + generation vs prompt depth, one profile
│   ├── sweep.sh          the same across configurations, one server per arm
│   ├── stress.sh         drive short prompts hard, on a server of its own
│   ├── soak.sh           hours of mixed traffic against the running unit
│   └── ckpt-value.sh     do context checkpoints pay off on YOUR traffic? (§30)
├── bench.sh              older llama-bench harness (§1-§16)
│
├── config/
│   ├── default.env       global defaults, fully annotated; IK_PROFILE lives here
│   ├── models/*.env      per-model profiles, each with its measurements in the header
│   └── systemd/          the user unit that serves the default
├── lib/common.sh         shared helpers: config, preflight, arg assembly
├── docs/
│   ├── RESULTS.md        every measurement taken, including the negative ones
│   ├── VS-DGX-SPARK.md   this card vs 2x DGX Spark, DeepSeek and Qwen
│   ├── TUNING.md         why every parameter is set the way it is
│   ├── BENCHMARKING.md   how to measure and how to read the numbers
│   ├── FAQ.md            platform, portability, hardware bottlenecks
│   ├── TROUBLESHOOTING.md
│   └── external/         what was sent upstream and every crash record;
│       └── patches/      the local diffs against ik_llama.cpp, with apply order
├── compat/               generated <math.h> shim (CUDA 13 + new glibc)
├── logs/                 server logs, timestamped
├── results/              benchmark reports (the ones the docs cite are tracked)
└── ik_llama.cpp/         upstream source (git clone, not vendored here)
```

---

## Further reading

- [docs/RESULTS.md](docs/RESULTS.md) — every measurement taken, in order: the
  Step-3.7 sweeps (§1–§5), DeepSeek placement, batching and the NaN hunt
  (§7–§49), Qwen (§51)
- [docs/VS-DGX-SPARK.md](docs/VS-DGX-SPARK.md) — how this box compares to 2× DGX
  Spark on DeepSeek-V4-Flash and Qwen3.8-Flash-Next (the verdict flips between
  them), and why published numbers vary fifteen-fold on identical hardware
- [docs/TUNING.md](docs/TUNING.md) — the arithmetic behind the defaults, and
  what to change when the model or context changes
- [docs/FAQ.md](docs/FAQ.md) — platform & portability: other NVIDIA cards, AMD
  /ROCm, multi-GPU, and why this box's setup was bumpy
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md) — methodology, interpreting
  `pp`/`tg`, and locking in what you find
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — OOM, gibberish output,
  slow loads, `sm_120` errors
- [docs/external/patches/README.md](docs/external/patches/README.md) — what is
  patched in the clone and why, and what was sent upstream
- [ik_llama.cpp parameter reference](ik_llama.cpp/docs/parameters.md) — upstream
  documentation for every flag

## License

[MIT](LICENSE).

This repository contains only its own scripts, profiles and documentation.
`ik_llama.cpp/` is **not** vendored here — `build.sh` clones it, and it carries
its own MIT licence from the ggml authors. The patches under `docs/external/`
are diffs against it and against [antirez/ds4](https://github.com/antirez/ds4),
which is MIT as well. So everything here is MIT-compatible in both directions.
