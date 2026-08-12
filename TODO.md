# TODO

Open threads left by the 2026-08-12 measurement session. Each one is a question
the data raised but did not answer, with enough context to pick it up cold.
Numbers and method live in [docs/RESULTS.md](docs/RESULTS.md) §8–§10.

---

## 1. ~~Apply the 512k findings to 262144~~ — RESOLVED & SHIPPED 2026-08-12

Measured in §12.1 and the interpolation undersold it: the kvram treatment at
262144 is worth **+73 % prefill / +8 % generation** over stock (402.5 / 13.38 vs
232.6 / 12.36 at 130k depth, caches on) — bigger than at 128k, because `--fit`
at margin 8192 leaves only 73 GiB of weights on the GPU there. Margin 4096 does
not load at this context (7.1 GiB compute buffer), so the §8 wrapper gating was
right. Shipped as `deepseek-v4-flash-256k-kvram` +
`serve-deepseek-v4-flash-mxfp4-kvram-256k.sh`. One cost worth knowing: the
checkpoints cost 24 % generation at this KV size (§12.1) — kept on regardless.

<details><summary>original item</summary>

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

</details>

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

## 4. The uncached-request dip — band mapped, mechanism still open

**Status 2026-08-12 (§12.2).** Both hypotheses died under measurement: the dip
is indifferent to `-b` (2048 / 4096 / 8192 all dip identically at 4k), and
"last logical batch nearly full" fails too (b 8192 at 2k depth = 24 % full,
dips hard). What the data says instead: **any uncached request at ~1k–16k depth
generates 15–35 % slower than its re-send**, under `--fit` and kvram alike —
including the shipped default. Clean at ≥32k. Upper edge between 16 327 and
33 079; lower edge below 1k.

**Still open: the mechanism.** The re-send restores the same depth from a
context checkpoint and is fast, so it is not the attention cost of the depth —
a fresh prefill leaves different state behind than a restore. Suspects worth
chasing in engine code: the DSA compressed-cache tiers (CSA/HCA/LID), or graph
scheduling right after prefill. Next step would be an upstream ik_llama.cpp
issue with the §12.2 table; the impact is bounded (first response of a session
in that band runs at ~80 % generation, once).

---

## 5. ~~Benchmark `antirez/ds4` against ik_llama~~ — DONE 2026-08-12

Measured head to head on the same IQ2XXS file (RESULTS §13): **ds4 prefills
1.5–1.8× faster** (≈2 100 tok/s, flat from 4k to 65k, vs ik's 1 423→1 166) and
**generates at 0.7–0.9×** (44.8–72.7 vs 55.9–79.5). Neither wins outright; with
MTP ik reaches 87–94 tok/s and ds4 has no equivalent here.

Getting there took four local patches (committed as
`docs/external/ds4-blackwell-discrete-fixes.patch`, and on branch
`local/blackwell-discrete-fixes` in `~/development/ds4` with `run-cuda-local.sh`),
each hiding the next: no `HostRegisterReadOnly` on driver 595.84; a `r--s`
Metal-branch mapping that cannot be pinned at all; a successful registration
short-circuiting the device weight cache (0.54 tok/s of PCIe zero-copy); and an
arena allocator whose packing overhead (~1.5×) makes full residency impossible
until the whole model goes in one arena. Partial residency is a cliff — 99.1 %
resident is 20.5 tok/s, 100 % is 72.6.

**Open follow-ups**, in order of value:

* **Report upstream.** Patches 1–2 (driver capability fallback, Linux mapping
  flags) are straightforwardly upstreamable and would make ds4 work on any
  discrete NVIDIA card with this driver class. Deliberately *not* filed yet —
  Matt's call, 2026-08-12.
* **DSpark.** ds4's speculative decoding needs its own support GGUF
  (`./download_model.sh ds4f-dspark`); it is the closest analogue to ik's MTP
  and would decide the generation column fairly.
* **Persistent disk KV.** Untested and the reason ds4 was interesting in the
  first place (§10 measured a 52-minute 500k prefill; ds4 claims it survives
  restarts).
* **MXFP4.** ds4 has its own ~156 GB MXFP4 file. It would not fit in VRAM, and
  the residency cliff above suggests the spilling case is exactly where ds4 is
  weakest — worth knowing, but a large download for a likely-negative result.

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
