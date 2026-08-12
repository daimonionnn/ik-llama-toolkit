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

## 2. Is `-b` the lever on the compute buffer?

**Why.** At 512k the CUDA compute buffer is 13.2 GiB, and it is what stops
`--n-cpu-moe 17` from loading — i.e. it directly costs ~3 GiB of experts that
would otherwise leave DDR5. §10.3 ruled out `-ub`: halving it to 256 freed only
668 MiB and cost 21 % prefill. `-b` (4096) is the remaining candidate, untested.

**What to run.** At `-c 524288`, `-nkvo --n-cpu-moe 19`, compare `-b 4096` (the
default) against `-b 2048` and `-b 1024`, watching VRAM at load. If the buffer
shrinks by ~3 GiB, retry `--n-cpu-moe 17` and 15.

**Caveat.** `-b` is expected to cost prefill like `-ub` did. At 512k prefill is
already 52 minutes for a full context, so a further loss there is not free.

---

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
