# TODO

Open threads from the 2026-08-12/13 measurement sessions. Each is a question the
data raised but did not answer, with enough context to pick it up cold. Numbers
and method live in [docs/RESULTS.md](docs/RESULTS.md) §8–§16.

Items 1–3 and 5 are resolved (kept, folded, for the reasoning); 4 and 6–8 are
open. Nothing here blocks anything shipped.

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

Getting there took four local patches (published as
[`daimonionnn/ds4@local/blackwell-discrete-fixes`](https://github.com/daimonionnn/ds4/tree/local/blackwell-discrete-fixes),
mirrored here as `docs/external/ds4-blackwell-discrete-fixes.patch`),
each hiding the next: no `HostRegisterReadOnly` on driver 595.84; a `r--s`
Metal-branch mapping that cannot be pinned at all; a successful registration
short-circuiting the device weight cache (0.54 tok/s of PCIe zero-copy); and an
arena allocator whose packing overhead (~1.5×) makes full residency impossible
until the whole model goes in one arena. Partial residency is a cliff — 99.1 %
resident is 20.5 tok/s, 100 % is 72.6.

**Open follow-ups**, in order of value:

* ~~**Report upstream.**~~ Filed 2026-08-13:
  [antirez/ds4#791](https://github.com/antirez/ds4/issues/791) — all four causes,
  with patches offered for (1) driver capability fallback and (2) the Linux
  mapping flags, and (3)/(4) left as discrete-vs-unified policy questions.
  Awaiting a reply; open a PR for 1–2 if the shape suits them.
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

## 6. The 4-P-core generation anomaly

**Why.** §16.2's reading — generation is hurt by heterogeneous cores, because
ggml waits for the slowest thread at every barrier — explains seven of eight
measurements. It does not explain this one: **4 P-cores with 4 threads gives
23.18 tok/s, beating 8 P-cores with 8 threads at 21.56**, on the same 1:1 ratio,
the same homogeneity, and half the compute. That is 7.5 % against a 2.1 % noise
floor, so it is real.

Whatever it is, per-barrier overhead apparently grows with thread count as well.
Worth a walk of 2/4/6/8 threads pinned to matching core counts, and if it holds,
an upstream question about ggml's barrier cost per thread.

**Low priority** — the configurations involved all cost most of the prefill,
which is the metric this box optimises for.

---

## 7. ~~`-muge` aborts on mixed-type quants~~ — FIXED UPSTREAM 2026-08-13

Not a mixed-type quant at all, and not really a `-muge` bug: it is a `-muge` +
`-rtr` interaction caused by one wrong entry in `interleaved_properties()`,
where `GGML_TYPE_MXFP4_R8` maps to itself instead of `GGML_TYPE_MXFP4` while
every other interleaved type maps to its base. Verified both ways — `-muge`
without `-rtr` loads fine, and the one-line fix makes `-muge -rtr` load cleanly
(43 layers merged, 34 tensors repacked). Full account in RESULTS §14.2.

Filed with the patch: [ikawrakow/ik_llama.cpp#2305](https://github.com/ikawrakow/ik_llama.cpp/issues/2305)
— closed the same day ("Thanks! Fixed now."), landed as
[`ee77f7ff` *Fix MXFP4 non-interlevaed type* (#2306)](https://github.com/ikawrakow/ik_llama.cpp/commit/ee77f7ff),
byte-identical to the reported patch. The local edit has been discarded and the
checkout moved to upstream.

---

## 8. Offline `_R4` repack for MXFP4

Every kvram profile carries `-rtr`, which forces `--no-mmap` and re-reads the
model at every start (~30 s warm here, minutes cold). §1.7 shows the gain can be
baked into the file with `llama-quantize --repack`, keeping mmap on.

**Not obviously worth it:** ~146 GB more on disk, and §1.7's lesson is that only
the CPU-resident layers may be `_R4` — an `_R4` expert on the GPU collapses to
0.35 tok/s — so the file would be **locked to one `--n-cpu-moe`**. With three
shipped kvram profiles at n17/n18/n20, that is three files or one that only
suits one profile. Revisit if startup time ever starts to hurt.

---

## 9. NaN logits abort under sustained agent load — cause unknown

**2026-08-13, ~4.3 h into a run of the shipped default profile**, the 99th
request aborted:

```
=============================== Failed to sample token
src/llama-sampling.cpp:745: Fatal error   (core dumped)
```

**Every logit was `nan`** (artifacts in `docs/external/crashes/`), so the
forward pass produced NaN and the sampler is only where it surfaced. Config was
`deepseek-v4-flash-128k-kvram` exactly as shipped: 92 630.93 MiB GPU /
56 498.00 CPU, KV on host, `-rtr` (51 tensors repacked), `-ub 2048`, at ~18.7k
depth just after context checkpoint 27 of 32.

**What was ruled out:**

* **Hardware** — no Xid, NVRM, MCE or ECC entries in the kernel log either side
  of the crash; GPU at 36 °C, no throttling.
* **The driver upgrade** — an untracked `probabilities.txt` already existed
  before the 610.43.02 upgrade, so this has happened at least once on 595.84.
* **Fuzzy prompt-cache reuse**, the obvious first suspect: the log shows 99
  reuses, *all* below 1.0 similarity and 73 with an `n_past` / `n_past_prompt`
  mismatch, including values as low as 0.0 and 0.541. The crash happened at
  0.953, which is unremarkable among them. Common conditions cannot explain a
  rare event.

**Suspects, in order:**

1. **`-rtr` / the `MXFP4_R8` path.** One bug in `MXFP4_R8` handling already
   turned up today (§14.2, `interleaved_properties()` mapping it to itself), so
   that path is demonstrably less exercised than the rest. It is in the shipped
   default. **The clean test is to run without `-rtr` for a comparable stretch**
   — it costs 25 % prefill, which is why it is a decision rather than an
   obvious move.
2. **The prompt cache exceeding its own limit.** The log reports
   `cache state: 1 prompts, 16116.439 MiB (limits: 8192.000 MiB, ...)` — twice
   the configured `--cache-ram`. That is a bug on its own terms regardless of
   whether it relates to the NaN, and worth reporting upstream.
3. **ECC is disabled on this card.** A single silent bit flip in ~90 GiB of
   resident weights would produce exactly this signature: rare, unreproducible,
   NaN. Enabling it costs ~6 % of VRAM capacity, which **would break the n17
   placement** — so it is a real trade, not a free safety net.

**Operationally:** the server dies rather than degrading, which leaves the
Hermes fallback with no backend until restarted. An auto-restart wrapper would
mask this; it would not fix it.

**FOURTH OCCURRENCE, 2026-08-13 18:47 — the whole caching layer is cleared, and
the crash tracks PREFILL VOLUME.** It ran with `--cache-ram 0` *and*
`-ctx-ckpt 0` — zero of either in the log — and aborted after 11 minutes.

The four runs together give the real signal:

| run | time to crash | prefill tokens/min | prefill tokens before crash |
|---|---:|---:|---:|
| 1 | 259 min | 1 138 | 294 709 |
| 2 | 31 min | 3 363 | 103 135 |
| 3 | 17 min | 5 674 | 94 372 |
| 4 | **11 min** | **12 083** | 134 124 |

Time-to-crash varies **24×** and tracks the prefill rate almost exactly, while
total prefilled tokens before each abort stays within one order (94k–295k). So
this is roughly **one abort per 100–300k prefilled tokens** — it is not a
function of uptime or request count.

The rising rate is an artefact of the bisect itself: with reuse disabled every
turn re-prefills from scratch (crash 4's log contains a 57 509-token
"reprocessing from scratch"). That does not invalidate steps 1–2, it just means
they raised exposure rather than making anything worse — which is useful.

**Step 3 needed a placement change first.** Turning `-rtr` off made the server
segfault on the second logical batch of the very first request:

```
kv cache rm [p0, end) p0=4096
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2560.08 MiB: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 2684440832
Segmentation fault (core dumped)
```

Comparing eight runs, `-rtr` turns out to shrink the **CUDA compute buffer**, not
just repack CPU experts. Weights are byte-identical either way (92 630.93 MiB
CUDA0 / 56 498.00 MiB host); the compute buffer is **1760.01 MiB with `-rtr` and
2496.01 MiB without it**. Since `--n-cpu-moe 17` was tuned to sit ~1.6 GiB from
the ceiling, the extra 736 MiB consumes the margin, and when `ggml_gallocr`
re-reserves at 2560 MiB it must hold the old buffer and the new one at once.

That is a genuine finding worth keeping — the profile's comment credited `-rtr`
with +25 % prefill from repacking alone, but part of its value is that it also
buys back VRAM headroom, which is what makes `-ub 2048` affordable at
`--n-cpu-moe 17`. Mechanism not yet established (§14.2's `interleaved_properties`
work is the obvious place to look). **Worth its own TODO.**

So step 3 runs at `--n-cpu-moe 18`. This does vary a second knob, but a placement
one: it moves one more layer's experts to host RAM, which changes no arithmetic.

**FIFTH OCCURRENCE, 2026-08-13 22:10 — step 3 is NEGATIVE, and with it the whole
of this toolkit's tuning is cleared.** Running `-rtr` off at `--n-cpu-moe 18`,
with cache and checkpoints still off, it aborted after 12 minutes having
prefilled **188 265 tokens** — squarely inside the band every other run landed in.

| step | removed | crash | prefill before abort | verdict |
|---|---|---|---:|---|
| — | baseline | 1 | 294 709 | — |
| — | baseline | 2 | 103 135 | — |
| 1 | prompt cache | 3 | 94 372 | negative |
| 2 | context checkpoints | 4 | 134 124 | negative |
| 3 | run-time repack | 5 | 188 265 | negative |

Five aborts, 94k–295k prefilled tokens each, time-to-crash spanning 259 → 11 min
purely with the prefill rate. **One abort per ~100–300k prefilled tokens, and
none of the prompt cache, the context checkpoints or `-rtr` is responsible.**

The kvram profile has therefore been **restored to its tuned state** — `-rtr` on,
`--n-cpu-moe 17`, cache and checkpoints back to defaults. Keeping them off bought
nothing and cost §11.3's free re-send.

**Step 4: run the `--fit` profile.** What is left of this toolkit's own tuning is
`-nkvo` and the manual `--n-cpu-moe` placement, and they cannot be removed
piecemeal — without `-nkvo` the KV (5504 MiB) returns to VRAM and the compute
buffer grows, so the placement has to change with it. The clean test is the
profile that shares none of it:

```
./serve-deepseek-v4-flash-mxfp4-gpu-cpu-128k.sh
```

`--fit` placement, KV on the GPU, `-ub 512`, no `-rtr`. Same weights, same quant.
Cost: prefill drops to ~287 tok/s (§11), so reaching 100–300k prefilled tokens
takes proportionally longer — expect to need roughly an hour of steady use rather
than 12 minutes.

If that crashes too, nothing configured here is the cause and the report goes
upstream on the strength of the volume correlation alone, which is by now the
single most reproducible fact about this bug.

**STEPS 4-5 ARE POSITIVE AND THEY AGREE: `-ub` SEPARATES THE TWO GROUPS
CLEANLY.** Two profiles have now survived far past the band, and the only thing
they share is a small micro-batch.

| run | `-ub` | `-rtr` | `-nkvo` | placement | prefilled tokens | outcome |
|---|---:|---|---|---|---:|---|
| 1 | 2048 | on | yes | ncmoe 17 | 294 709 | abort |
| 2 | 2048 | on | yes | ncmoe 17 | 103 135 | abort |
| 3 | 2048 | on | yes | ncmoe 17 | 94 372 | abort |
| 4 | 2048 | on | yes | ncmoe 17 | 134 124 | abort |
| 5 | 2048 | off | yes | ncmoe 18 | 188 265 | abort |
| step 4 | **512** | off | no | `--fit` | 541 369 | **clean** |
| step 5 | **512** | on | yes | ncmoe 17 | 528 577 | **clean** |

**Step 5 is the decisive one.** It carries `-rtr`, `-nkvo` and `--n-cpu-moe 17` —
byte-identical to the configuration of aborts 1-4 — and differs from them in
exactly one flag. 528 577 prefilled tokens over 20 evaluations and 201 minutes,
deepest prompt 94 973 tokens, zero aborts. That is 1.79x the worst abort and 5.6x
the earliest.

Every abort has `-ub 2048`. Neither clean run does. **Micro-batch size is the
variable**, and step 4's survival — which I initially read as evidence for
placement — is better explained the same way: that profile also runs `-ub 512`.

Not yet proof of a mechanism, and n=2 on the clean side. But it is the first
variable in the whole bisect that partitions the runs without exception, and it
is the one that changes the *shape* of the computation: a 2048-row micro-batch
means different GEMM dimensions and different kernel selection from a 512-row one.
This build already produced one micro-batch-shaped bug today (§14.2,
`interleaved_properties` mapping `MXFP4_R8` to itself), which is at least a hint
that the wide-batch repacked path is the least-travelled one.

**`-ub 1024` ALSO SURVIVES, so the threshold sits between 1024 and 2048.**
532 148 prefilled tokens over 18 evaluations and 30 minutes, zero aborts. Three
clean runs now, and the partition is unbroken:

| `-ub` | runs | prefilled tokens | outcome |
|---:|---:|---|---|
| 2048 | 5 | 94k-295k | **all abort** |
| 1024 | 1 | 532 148 | clean |
| 512 | 2 | 528 577 / 541 369 | clean |

**And it is nearly free.** Measured at 32k depth with a fixed methodology, two
independent runs 10 minutes apart:

| `-ub` | prefill at 32k | source |
|---:|---:|---|
| 2048 | 484.1 tok/s | §11 |
| 1024 | 465.9 / 459.9 tok/s | depthbench, 1.2-1.5 % spread |

**~4 % of prefill** to move off the configuration that aborts every 100-300k
prefilled tokens. Full depth curve at `-ub 1024` (`max_tokens` 160, temp 0,
unique salt per request, 2 repeats):

| depth | prefill tok/s | generation t/s |
|---:|---:|---:|
| 515 | 277.6 | 21.55 |
| 1 027 | 356.2 | 21.41 |
| 4 101 | 472.2 | 21.47 |
| 32 701 | 462.9 | 20.07 |
| 127 981 | 409.6 | 17.48 |

Prefill peaks around 4k and shallow prompts pay a fixed overhead (515 tokens
reaches only 60 % of peak). Generation is flat to 32k, then falls.

A cliff between 1024 and 2048 rather than a gradient is itself a clue: it points
at a branch selected by micro-batch size rather than at anything that degrades
progressively. `-b` is 4096 throughout, so `-ub` is the only thing moving.

**SIXTH ABORT, 2026-08-14 10:23 — the reversion test DID reproduce it, under
interactive traffic, at 331 269 prefilled tokens.** The band is wider than the
94k-295k seen before, which is why the run briefly looked like a refutation: it
passed the old upper edge and aborted a minute later.

| `-ub` | traffic | runs | prefilled tokens | outcome |
|---:|---|---:|---|---|
| 2048 | interactive | 6 | 94k-331k | **all abort** |
| 2048 | synthetic | 1 | 397 077 | clean, but stopped rather than survived |
| 1024 | mixed | 1 | 532 148 | clean |
| 512 | interactive | 2 | 528 577 / 541 369 | clean |

**Six of six at `-ub 2048`; none at anything smaller.** The two clean 512 runs
were real interactive use, so 1.07 M tokens of the workload that aborts six times
at 2048 pass without one. `-ub 2048` is implicated; the profile is back to 1024.

What the synthetic run shows is weaker than it looks — it was stopped at 397 077,
not survived, so it cannot rule the micro-batch guilty *only* in combination with
interactive traffic. It is suggestive, not evidence.

**Where to look next: the partial micro-batch path.** Interactive traffic is
mostly short prompts — median 1 800 tokens in the crash-6 run, 12 of 23 below
2048, meaning a single partial micro-batch. The benchmark's eight huge uniform
prompts fill nearly every micro-batch and did not reproduce it. That asymmetry is
the most specific lead the investigation has produced.

*(Earlier reading of this run, before it aborted: "397 077 prefilled tokens with NO
abort")* — 1.35x the worst previous crash — before it was stopped. The clean
partition is therefore NOT a controlled result, and `-ub` alone does not explain
the aborts.

One difference remains untested: the **shape** of the load. All five aborts came
from interactive Hermes traffic (many turns, varied and often short prompts); the
reversion test was eight huge uniform prompts, where nearly every micro-batch is
full. A fault in a partial micro-batch path at large `-ub` would barely be
touched by that. The next data point is real use at `-ub 2048`, which is why the
profile is back to 2048 as a test setting.

*(Superseded: `-ub 1024` was made the default earlier the same day)* in
`config/models/deepseek-v4-flash-128k-kvram.env`. It costs ~4 % of prefill
(462.9 vs 484.1 tok/s at 32k) and removes the only variable that has ever
separated an abort from a clean run. A profile that aborts every 100-300k
prefilled tokens is not usable; 4 % is cheap for that.

Next, in order:
  1. **Confirm by reversion (still owed).** Put `-ub 2048` back on an otherwise identical run.
     A sixth abort inside 94k-295k turns a clean partition into a controlled
     result, and it is the single cheapest experiment left (~15 min at this rate).
  2. **Narrow the threshold.** If 2048 aborts and 512 does not, 1024 says whether
     this is a cliff or a gradient. `-b` is fixed at 4096 throughout, so `-ub` is
     the only thing moving.
  3. **Then file upstream** — with the partition table, the volume correlation and
     `probabilities.txt` for all five aborts, which is a much stronger report than
     the volume correlation alone.

---

## 10. Why does `-rtr` shrink the CUDA compute buffer?

Discovered while setting up bisect step 3 for item 9, and not something the
profile's own documentation predicted.

With everything else identical -- same model, `-c 131072`, `-nkvo`, `-ub 2048`,
same `--n-cpu-moe`, byte-identical weight buffers -- the CUDA0 compute buffer is:

| `-rtr` | compute buffer | runs |
|---|---:|---:|
| on | 1760.01 MiB | 7 |
| off | 2496.01 MiB | 1 |

**+736 MiB, or +42 %, from a flag documented as repacking CPU-side experts into
`MXFP4_R8` for the AVX2/VNNI kernels.** Those tensors are on the host; why the
*CUDA* graph's scratch space should depend on them is unclear.

Two guesses worth testing:
  * the repacked type changes which ops the scheduler assigns to CUDA vs CPU, so
    the CUDA graph genuinely holds fewer/smaller intermediates;
  * or an un-repacked MXFP4 expert forces a dequant/staging buffer on the GPU
    side for the hybrid path.

Why it matters: §11 attributes the kvram profile's win to repacking (+25 %) and
`-ub 2048` (+20 %) as if independent, but the second is only affordable *because*
of the first. If the buffer growth can be avoided without `-rtr`, `--n-cpu-moe 17`
becomes reachable in more configurations; if it cannot, the profile comment
should say so plainly.

