# TODO

Open threads left by the 2026-08-12 measurement session. Each one is a question
the data raised but did not answer, with enough context to pick it up cold.
Numbers and method live in [docs/RESULTS.md](docs/RESULTS.md) §8–§10.

---

## 1. Apply the 512k findings to 262144 — the forgotten middle

**Why.** 256k is the one context where none of today's work landed. The
128k wrapper switches `--fit-margin` to 4096 only at `ctx <= 131072`; at 262144
it deliberately falls back to the profile's 8192, because a smaller margin was
never tested there. And the `-nkvo` + manual-placement trick was measured at 512k
and at 131072, but never in between.

**Expectation.** The win scales with how much VRAM the KV occupies — 5.4 GiB at
128k gave +2.4 %, 21.5 GiB at 512k gave +22 %. At 262144 the KV is **11.4 GiB**,
squarely in between, so ~+10 % is the interpolation. That would justify a
`deepseek-v4-flash-256k` profile alongside the 512k one.

**What to run** (~40 min, same method as §10.2 — HTTP against a live server,
lean flags, cgroup cap, ~130k depth so each point is ~10 min):

1. baseline: `IK_CTX=262144 IK_FIT=1 IK_FIT_MARGIN=8192` (reference)
2. does a smaller margin even load? `IK_FIT_MARGIN=4096`, same ctx
3. the real test: `IK_FIT=0` + `-nkvo --n-cpu-moe N`, walking N down from ~24
   until the compute buffer OOMs

Read the actual split out of the load log (`CUDA0 buffer size` / `CUDA_Host
buffer size` / `KV self size`) rather than trusting the flags — that is how the
512k run caught `--fit` silently ignoring `-nkvo`.

**If it pays:** ship `config/models/deepseek-v4-flash-256k.env` +
`serve-deepseek-v4-flash-mxfp4-gpu-cpu-256k.sh`, modelled on the 512k pair.

---

## 2. ~~Is `-b` the lever on the compute buffer?~~ — RESOLVED 2026-08-12, no

Answered by the §11 sweeps at 131072, no separate 512k run needed:

* **`-b` does not touch the compute buffer.** `-b 8192` vs `-b 4096` at the same
  `-ub`: identical 3 624 MiB buffer, identical prefill (364.2 vs 366.8).
* **The buffer tracks `-ub`, linearly** (512→3 624, 1024→7 248 under `--fit`),
  and the real lever is `-nkvo`: the attention scratch follows the KV to host
  memory and the buffer collapses to 440 MiB.
* The original goal — freeing `--n-cpu-moe 17` at 512k — is dead anyway: the
  §10.2 walk already ran **with** `-nkvo` and n17 still CUDA-OOMs there on
  weights + DSA caches. 19 stays the 512k ceiling.

---

## 3. ~~Checkpoints or prompt cache — which one is the 26 %?~~ — RESOLVED 2026-08-12

Measured in §11.3, and the §10.4 guess was backwards: the **context checkpoints**
are what make a re-send free, not `--cache-ram`. `-ctx-ckpt 0` alone loses the
reuse entirely while saving ~1.5 % prefill. Both stay ON in the shipped
profiles; the caches cost only 3.5 % at 32k depth (the 26 % figure was at 256k).

<details><summary>original item</summary>

## 3. Checkpoints or prompt cache — which one is the 26 %?

**Why.** `-ctx-ckpt 0` and `--cache-ram 0` have only ever been measured
**together** (§9.3: 54 GiB of RAM and +26 % generation at 256k; §10.4: 17.43 vs
13.1–14.9 tok/s at 128k). The attribution matters because the two have very
different costs: context checkpoints are pure overhead as far as anything here
shows, while the prompt cache is what makes a re-send free — §9.1's repeat pass
hit it and skipped an 8-minute prefill.

If the speed belongs to the checkpoints, `-ctx-ckpt 0` alone could go into the
shipped wrappers and be a clean win. If it belongs to the cache, it stays a
trade and the wrappers should keep it on.

**What to run.** Three servers at `-c 131072`, ~120k depth: stock,
`-ctx-ckpt 0` only, `--cache-ram 0` only. ~30 min.

</details>

---

## 4. The ~4k generation artefact

**Why.** A first, uncached request at ~4k depth generates far slower than the
same request re-sent (14.2 / 10.5 / 15.0 across three runs, recovering to 21.5),
and it is *not* a "generation after a big prefill" effect — 32k has a larger
prefill and is stable to within 2 % (§9.1). The only suspicious coincidence is
that 4096 is exactly `-b`: that prompt fits in a single full batch, while every
deeper one ends on a partial batch.

**What to run.** Measure the first uncached request at ~4k with `-b 2048` and
`-b 8192`. If the dip follows the batch size, the hypothesis is confirmed and
the §9.1 row can be explained instead of merely flagged.

**Low priority** — it is a measurement artefact, not something that costs real
work, unless it turns out to affect ordinary short-prompt turns too.

---

## 5. Benchmark `antirez/ds4` against ik_llama

**Why.** [ds4](https://github.com/antirez/ds4) ("DwarfStar") is a purpose-built
engine for exactly the model this box runs, written by the author of **the quant
already in use here** — `antirez/deepseek-v4-gguf`. It started Metal-only but now
lists Metal, CUDA and ROCm. Two features have no equivalent in ik_llama and both
target pain this repo measured:

* a **KV cache persisted to disk across restarts**, where §10 measured a 500k
  prefill costing 52 minutes and §9.3/§10.4 showed the in-RAM prompt cache is
  the only thing making a second turn bearable;
* **1M context**, against the 524288 that fits here today.

It also ships **DSpark**, its own speculative decoding — worth noting because a
widely-copied AI-written summary misattributes DSpark to SGLang. It is ds4's.

**The bar is high.** ik_llama does **87–94 tok/s** on this box with the antirez
IQ2XXS quant plus MTP (§7). ds4 publishes 39.35 tok/s on an M5 Max and 18.05 on
a DGX Spark GB10; no RTX PRO 6000 numbers exist. DGX Spark is memory-bandwidth
poor, so it is a weak predictor for this card — but nothing suggests a walkover.

**What to run.** Clone to `~/development/ds4` (its own upstream checkout, the way
`ik_llama.cpp/` is here), build the CUDA backend, serve the **same** IQ2XXS
chat-v2 file. Then measure with the §9/§10 method — HTTP against the live server,
the same depths, unique prompt prefixes — so the numbers drop straight into the
same tables rather than being "roughly comparable". Compare against both
baselines: 87–94 tok/s (IQ2XXS + MTP) and ~21 tok/s (MXFP4 GPU+CPU).

**Decision rule.** Numbers go into RESULTS §11 either way. ds4 gets its own
`ds4-toolkit` repo **only if** it wins on speed, or if the persistent disk KV
proves valuable enough on its own — `serve.sh`'s `IK_*` abstraction (`--fit`,
`-ncmoe`, margins) has nothing in common with ds4 and should not be bent to hold
a second engine.

**Caveats:** beta quality by the author's own statement, heavily AI-assisted
code, deliberately narrow, and it loads only the author's own GGUF files.

---

## Evaluated and rejected

**KTransformers** — wrong hardware profile for this box, decided 2026-08-12.

Its validated DeepSeek-V4-Flash configuration is a single RTX 5090 (32 GiB) with
≥200 GiB of RAM, reporting 20+ tok/s decode; the famous ~27× over llama.cpp comes
from **AMX** MoE kernels on dual 32-core Xeons. Two facts about this machine kill
it:

* the CPU is a **Core Ultra 7 270K Plus — AVX2 and AVX_VNNI only, no AVX512, no
  AMX**, so it would land on the slowest fallback kernels, and
* its entire premise is offloading experts to RAM, which this box does not need:
  96 GiB of VRAM holds the whole 81 GiB antirez quant, and that is exactly where
  the 87–94 tok/s comes from.

Its single-GPU figure (20+ tok/s) also matches what ik_llama already does on the
MXFP4 path (~21 tok/s), and §10 showed generation here tracks one variable —
GiB of weights left in DDR5. A faster CPU kernel does not add memory bandwidth.

**SGLang / vLLM with `flashinfer_mxfp4`** — same root problem. They target
fully-resident GPU serving; MXFP4 is 145.6 GiB against 96 GiB of VRAM, so it
would need the offload path that is not their strength.
