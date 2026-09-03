# Benchmarking

```bash
./bench.sh <mode> [profile]
```

| mode      | what it answers                            | ~time  |
|-----------|--------------------------------------------|--------|
| `quick`   | How fast is it right now?                  | 5 min  |
| `sweep`   | Does it hold up as the context fills?      | 10 min |
| `threads` | What is the best `-t` for this hybrid CPU? | 10 min |
| `ncmoe`   | Can I beat `--fit`'s expert split by hand? | 25 min |
| `batch`   | What are the best `-b` / `-ub`?            | 15 min |
| `rtr`     | Is run-time repack worth losing mmap?      | 15 min |
| `full`    | All of the above                           | 80 min |

Loading this model costs real time (~114 GiB), so modes sweep inside a single
`llama-bench` invocation wherever the parameter allows it. `threads` and `batch`
load the model **once**. `ncmoe` and `rtr` change how tensors are placed at load
time, so those need one load per data point — which is why they are the slow
ones.

Every run writes `results/<profile>-<mode>-<timestamp>.md` (tables plus a
header recording GPU, CPU, RAM, ik_llama commit and the config used) and
`.raw.log` with the untouched output.

---

## The two tools

**`llama-bench`** runs a fixed matrix: prefill `N` tokens, generate `M` tokens,
repeat, report the mean. Good for A/B comparisons because every run is
identical. This is what `quick`, `threads`, `ncmoe`, `batch` and `rtr` use.

**`llama-sweep-bench`** repeats the same measurement at growing context depths:
process a chunk, generate from that depth, extend, repeat. This is what `sweep`
uses, and it is much closer to how a chat server actually behaves — a number
measured at an empty cache tells you very little about turn 30 of a
conversation.

Both accept the same placement flags as `llama-server`, so results transfer
directly to the running server.

---

## Reading the numbers

Two metrics, and they respond to completely different things:

**`pp` — prompt processing / prefill, tokens/s.** How fast your input is
ingested. Compute-bound, batched, runs mostly on the GPU. Improved by larger
`-ub` and more `IK_THREADS_BATCH`. Expect it in the thousands on the current default; the "hundreds, not thousands" note further down predates the gpu-experts placement.

**`tg` — token generation, tokens/s.** How fast the reply streams. Bound by
reading the 8 active experts per layer, and specifically by the layers sitting
in system RAM. This is the number you feel while using the model, and the one
that expert placement moves.

A change that raises `pp` while lowering `tg` almost always means it consumed
VRAM that was holding experts. On this machine that trade is usually a loss —
you prefill once and generate hundreds of times.

### Measured numbers

The full measured grid of context × expert-split lives in
[TUNING.md §1](TUNING.md#measured-splits-2026-07-25-gpu-idle-q8_0-kv-618-threads).
The short version, all on this box with the GPU idle:

- **Untuned starting point** (`--fit`, 65 536 context): tg ~27 t/s.
- **What tuning bought**: hand-picking `--n-cpu-moe` instead of `--fit` lifted
  tg to **~46 t/s** at 65 536 — a +70% win, entirely from putting more experts
  on the GPU. `--fit` was leaving ~6 expert layers on the CPU unnecessarily.
- **Step-3.7-Flash at 262 144 context** (`-ncmoe 22`, q8_0 KV): tg **~25 t/s**. (This was the shipped default when §1-§5 were written; today's default is `qwen38-flash-next-q8-128k` at 2303 pp / 40.6 tg.)
  Lower than the 65 536 number *by design* — the full 262k window costs 24 GiB
  of KV cache, which is ~9 expert layers of VRAM handed back to the CPU.

Two things worth understanding:

- **Generation speed is set by how many expert layers sit in system RAM**, which
  is set by context length (KV cache size) and the `-ncmoe` split. This is the
  one dial that matters; see [TUNING.md §1](TUNING.md#what-it-costs).
- **Prefill is much lower than a GPU-resident model** would give (hundreds, not
  thousands of t/s), because at large batches all 288 experts activate, so the
  CPU-resident layers bottleneck prefill just as they do generation. Inherent to
  running a model larger than VRAM.

The prediction worth holding onto is the *shape*: `tg` falls roughly linearly
with the number of expert layers on the CPU. If a change makes `tg` fall off a
cliff instead, something else is wrong — start with
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Getting comparable results

Benchmarking a model of this size has more traps than a small one. (The figures below were taken on Step-3.7-Flash at ~114 GiB; the current default is 175 GiB, so allow proportionally more.)

**Free the GPU first.** `bench.sh` warns about other processes holding VRAM but
does not stop. With `--fit` in play, a benchmark run against 20 GiB of free VRAM
measures a completely different expert split than one against 95 GiB. This is
the single most common cause of results that make no sense.

**Warm the page cache.** The first run after boot reads ~114 GiB from NVMe;
later runs hit RAM. Discard the first run, or run `quick` once as a warmup.

**Pin the split when comparing.** `--fit` re-decides on every launch, so two
runs may not be measuring the same configuration. For A/B work set an explicit
split:

```bash
IK_NCMOE=17 IK_FIT=0 ./bench.sh quick
```

`bench.sh ncmoe` does this for you.

**Watch for thermal drift.** The P-cores boost to 5.3 GHz and will not hold it
across an 80-minute `full` run. If `threads` results look noisy, re-run the
interesting points on their own.

**Change one thing at a time.** Obvious, but easy to violate when a profile
edit changes `-ub` and `-t` together.

---

## Turning results into config

Once a sweep finds something better, make it permanent by editing the profile:

```bash
# config/models/step-3.7-flash-q4.env
: "${IK_THREADS:=8}"      # bench.sh threads, 2026-07-24: +6% tg over 6
: "${IK_UBATCH:=2048}"    # bench.sh batch,   2026-07-24: +18% pp, tg unchanged
```

Keep the note about which run justified it. In six months, when a new
ik_llama.cpp version changes the picture, the comment is what lets you tell a
deliberate choice from an accident.

Then confirm the combination end to end — sweeps optimise one variable at a
time and the winners do not always compose:

```bash
./bench.sh quick && ./bench.sh sweep
```

---

## Beyond throughput

`bench.sh` measures speed, not quality. If you want to check that a setting —
especially `IK_SER`, or a coarser KV cache — has not damaged the model, the
upstream tools are already built:

```bash
# Perplexity over a text corpus (lower is better; only compare like with like)
ik_llama.cpp/build/bin/llama-perplexity -m <model> -f <text-file> -c 4096

# Interactive smoke test
ik_llama.cpp/build/bin/llama-cli -m <model> -ngl 99 --fit -p "..."
```

Perplexity numbers are only meaningful against the *same* corpus and context
length, so record a baseline before you start changing quality-affecting flags.
