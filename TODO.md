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

## 1 (historical). Apply the 512k findings to 262144 — the forgotten middle

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

## 3 (historical). Checkpoints or prompt cache — which one is the 26 %?

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

## 4. ~~The uncached-request dip~~ — GONE 2026-08-19, it was `-rtr`

> Re-measured in the `gpu-experts` regime across seven depths: generation is flat
> at 20.06 / 20.52 / 20.14 / 20.45 / 20.37 / 20.17 / 19.38 from 523 to 32 737
> tokens — 2.3 % spread where §12 recorded a collapse to 14-15 t/s. The dip
> belonged to the CPU path and disappeared with it (RESULTS §27.5). Closed without
> the mechanism ever being found, which is an acceptable ending for a symptom of a
> configuration nobody runs any more.

## 4 (historical). The uncached-request dip — band mapped, mechanism still open

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

## 6. ~~The 4-P-core generation anomaly~~ — DOES NOT REPRODUCE 2026-08-19

> `-t` 24 / 16 / 8 / 4 in the `gpu-experts` regime gives 20.39 / 20.65 / 19.65 /
> 16.36 tg at 4k — monotonic, no anomaly. §16's unexplained 23.18 t/s at four
> threads was, like item 4, a property of the `-rtr` path (RESULTS §27.6).

## 6 (historical). The 4-P-core generation anomaly

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

## 8. ~~Offline `_R4` repack for MXFP4~~ — OBSOLETE 2026-08-17, wrong direction

> §21 settled this by accident. `_R4` types have **no CUDA kernel at all** (0 files
> under `ggml/src/ggml-cuda/`, 7 under `ggml/src/iqk/`), so baking a repack into
> the file does not merely trade disk for load time — it **pins those experts to
> the CPU permanently**, which is the strategy that lost 3x on this machine.
>
> The existing `step-3.7-flash-q4-r4` profile is the same trap already sprung: it
> runs `-rtr 0` and still computes on the CPU, because the repack is in the file.
> Its 114 GB GGUF is a candidate for deletion (see item 12).
>
> Original reasoning below; it was sound about disk and load time, and simply did
> not know that the choice of layout is also the choice of processor.

## 8 (historical). Offline `_R4` repack for MXFP4

Every kvram profile carries `-rtr`, which forces `--no-mmap` and re-reads the
model at every start (~30 s warm here, minutes cold). §1.7 shows the gain can be
baked into the file with `llama-quantize --repack`, keeping mmap on.

**Not obviously worth it:** ~146 GB more on disk, and §1.7's lesson is that only
the CPU-resident layers may be `_R4` — an `_R4` expert on the GPU collapses to
0.35 tok/s — so the file would be **locked to one `--n-cpu-moe`**. With three
shipped kvram profiles at n17/n18/n20, that is three files or one that only
suits one profile. Revisit if startup time ever starts to hurt.

---

## 9. NaN logits abort — OPEN, cause unknown

> **2026-08-19: not solved. See RESULTS §32.** The fix below is in the running
> build and the abort came back anyway — twice, at a rate real traffic cannot
> tell apart from before it (1 per 3.7 h against 1 per 3.3 h). Two hypotheses
> drawn from the crash logs were tested and both failed: checkpoint-adjacency is
> vacuous at 293 checkpoints per 422 tasks, and the `n_past` mismatch occurs in
> 12 % of ordinary restores and is absent from the newest crash. Nine aborts now
> span two builds, both expert strategies and `-ub` 1024/2048/8192. Next step is
> upstream, with `docs/external/crashes/crash8`/`crash9` attached.
>
> **2026-08-17, kept because it is where the mistake is legible: "probably
> solved, see RESULTS §24."** Our checkout predated
> `ff141691` "Use f32 accumulation in CUDA DSA implementation" by three and a half
> hours — merged the afternoon of the first abort. f16 overflow in the attention
> accumulator produces inf, then NaN, then a poisoned output tensor, which matches
> all-NaN logits, the prefilled-volume correlation, and aborts in both §21
> regimes. On the updated build `tools/stress.sh` ran 610 103 prefilled tokens of
> exactly the shape that used to abort, clean, at no measurable performance cost.
> Still owed: a day of real Hermes traffic. Everything below is the investigation
> as it happened and is kept for the negative results.

## 9 (historical). NaN logits abort under sustained agent load — cause unknown

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

## 10. ~~Why does `-rtr` shrink the CUDA compute buffer?~~ — ANSWERED 2026-08-19

> **It does not.** Nine measurements already in the logs fit
> `buffer = max(rate(context) x ub, floor(-rtr))`, with rate 0.859 MiB/unit at
> 131072 and 1.359 at 262144, and a ~2496 MiB floor present only when `-rtr` is
> off. That floor is staging space for the expert tensors the GPU computes once
> the repack is not pinning them to the CPU — §21's mechanism seen from the
> allocator (RESULTS §28).
>
> It also explains why the effect kept appearing and vanishing: at `-ub 2048` the
> floor exceeds the proportional term and is visible; above `-ub` ~2900 it is not.
> Every shipped profile runs well above that crossover, so the floor never decides
> anything, and the question is closed without modelling its exact size.

> **2026-08-19: the effect is real and my doubt was wrong.** Controlled at fixed
> `--n-cpu-moe 18` and `-ub 2048`, byte-identical weights: **1760.01 MiB with
> `-rtr`, 2496.01 without**, +42 %. The "DOUBTFUL" note added on 2026-08-17 was
> based on an uncontrolled comparison and has been removed.
>
> What that investigation did turn up is that the buffer scales with **context**
> as well as `-ub`: 7040 MiB at 131072 against 11 136 at 262144, same `-ub 8192`.
> So §18.3's 0.859 MiB-per-unit is a 131072-only law (RESULTS §27.2).
>
> Still open: *why* the repack changes the CUDA graph's scratch at all, when the
> tensors it rewrites live on the host.


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

---

## 11. ~~Tune the `gpu-experts` regime~~ — DONE 2026-08-17, one thing owed

> All four steps are complete and shipped: `-nkvo` cannot be dropped (§22.3),
> `--n-cpu-moe` wants the floor at 19 (§22.2), `-ub 8192` / `-b 8192` is worth
> +17 % at depth (§22.4), threads were already right (§22.5). The 262144 profile
> was converted the same way on 2026-08-19 (§27.3).
>
> What is still owed is not tuning but the stability soak, which is item 9.

## 11 (historical). Tune the `gpu-experts` regime — it starts 3x ahead and untuned

New profile `deepseek-v4-flash-gpu-experts-128k` (RESULTS §21). Experts stay in
host RAM but are computed **on the GPU**, which is what happens as soon as `-rtr`
is not passed. Measured 1375.7 / 1562.8 / 1529.0 / 1145.4 pp at 4k / 16k / 32k /
128k, against 478 / — / 486 / 437 for the shipped kvram profile.

Every number in it was borrowed from `multi-gpu-llm-toolkit`, which swept them
for a different engine. Nothing here has been swept. In rough order of expected
value:

1. ~~**`-nkvo` off.**~~ **DONE 2026-08-17, and the answer is no** — at least not
   at `-ub 4096`. Swept `--n-cpu-moe` 19, 20 and 22 with the KV back on the GPU;
   **all three fail to load with CUDA OOM**, even n22, which frees four expert
   layers (~13 GiB). Baseline reproduced to 0.1 % in the same sweep (1375.0 /
   1526.8 pp), so the harness is sound.

   What blocks it is not the KV — 5504 MiB would fit — but the attention scratch,
   which returns to VRAM with it. The profile comment already recorded that going
   `-nkvo` collapses the compute buffer from 3624 to 440 MiB **at `-ub 512`**; at
   `-ub 4096` the same ratio would be tens of GiB.

   **The number, from the OOM message itself:**

   | | CUDA0 compute buffer at `-ub 4096` |
   |---|---:|
   | `-nkvo` (KV in RAM) | 3 520 MiB |
   | KV on the GPU | **28 992 MiB** |

   8.24x, and the same ratio the profile recorded at `-ub 512` (3624 vs 440), so
   the attention scratch is linear in `-ub` too — at **7.08 MiB per unit** against
   0.859 for the rest. Fitting it in the free VRAM would need `-ub` below ~1150,
   which throws away the amortisation that makes streaming pay in the first place.

   **`-amb 512` does nothing here, on either axis.** All three OOM arms requested
   the identical 28 992.18 MiB, and the arm that did load matched the baseline in
   both allocation (3520.02 MiB) and speed (1372.3 / 1532.9 pp against 1375.0 /
   1526.8 — 0.2 % and 0.4 %). ik_llama's attention-batch cap does not reach this
   path; presumably `-mla 3` takes a different route. Worth knowing before
   reaching for it again.

   So `-nkvo` is not a choice in this regime, it is a requirement, and the
   remaining 10-12 % against `multi-gpu-llm-toolkit` is explained rather than
   closed: upstream fits `--n-cpu-moe 18` + KV on GPU + `-ub 4096` in the same
   96 GiB, so its attention scratch is far smaller, and ours forces the KV out to
   host RAM and attention onto the CPU. That is an ik_llama implementation
   difference, not something a flag here can reach.
2. ~~**`--n-cpu-moe`.**~~ **DONE 2026-08-17 — 18 is right, and it is right because
   it is the floor.** Lower is better on both axes, monotonically, until it stops
   fitting:

   | `--n-cpu-moe` | host weights | pp 4k | pp 32k | tg 32k |
   |---:|---:|---:|---:|---:|
   | 14, 16 | — | **OOM** | | |
   | **18** | 59 762 MiB | **1366.0** | **1528.4** | **19.83** |
   | 20 | 66 290 | 1307.9 | 1457.4 | 18.52 |
   | 24 | 79 346 | 1218.0 | 1353.6 | 16.44 |

   Each layer pushed to host RAM costs **-1.9 % prefill and -2.8 % generation**,
   linearly across the range. The compute buffer stays 3520.02 MiB throughout, so
   `--n-cpu-moe` moves weights only — it does not buy scratch.

   This is a different shape from the `-rtr` path, where §17 measured ~102 us per
   layer per token and the curve flattened above 19. There the constraint was CPU
   GEMM; here it is bytes over the link every batch, so nothing saturates and the
   penalty stays linear. Generation suffers more than prefill (-2.8 % vs -1.9 %)
   because a single decoded token has nothing to amortise the transfer over.

   16 does not load: fewer layers in RAM means more weights in VRAM and no room
   left for the 3520 MiB compute buffer. **17 does not load either**, so 18 is
   exactly the floor at `-ub 4096` — the borrowed value happened to be the right
   one, and there is nothing below it to win.

   Consequence for `-ub`: the two are in direct conflict. A larger micro-batch
   needs a bigger compute buffer, which has to be paid for by pushing another
   layer to host RAM at -1.9 %. So raising `-ub` only wins if it beats that.
3. ~~**`-ub` / `-b`.**~~ **DONE 2026-08-17 — `-ub 8192` at `--n-cpu-moe 19` is
   worth +17.9 % at 32k**, easily repaying the extra layer.

   | arm | pp 4k | pp 32k | tg 32k | compute buffer |
   |---|---:|---:|---:|---:|
   | ub 2048 / b 8192 / n18 | 1029.3 | 1122.6 | 19.73 | 2496 MiB |
   | ub 4096 / b 8192 / n18 | **1370.8** | 1520.7 | 19.62 | 3520 |
   | ub 4096 / b 4096 / n18 | 1371.9 | 1435.9 | 19.77 | 3520 |
   | ub 6144 / b 8192 / n18 | — | **OOM at depth** | | 5280 |
   | **ub 8192 / b 8192 / n19** | 1336.7 | **1792.9** | 19.15 | 7040 |

   Three things worth keeping:

   * **The optimum depends on depth.** `-ub 4096` wins at 4k by 2.5 %; `-ub 8192`
     wins at 32k by 17.9 %. At 4k a 4101-token prompt is a single micro-batch
     either way, so the larger setting only shows its cost, not its benefit.
   * **`-b` matters, but only at depth.** Identical at 4k (1371.9 vs 1370.8, 0.08 %)
     and +5.9 % for 8192 at 32k. A 32k prompt splits into eight logical batches at
     `-b 4096` and four at 8192, and each boundary costs something; at 4k there is
     one batch either way. RESULTS §2 called `-b` inert, but measured it in the
     `-rtr` path where CPU GEMM buried the difference.
   * **Loading is not fitting.** `ub 6144` at n18 started cleanly, reported its
     5280 MiB buffer, served a 4k prompt — and then died at 32k with a CUDA OOM in
     `cuMemCreate` (`ggml-cuda.cu:469`), because the allocator grows on demand. Any
     fit check has to run at depth. Note this is an OOM, **not** the NaN abort of
     item 9.

   The compute buffer follows 0.859 MiB per unit of `-ub` exactly, so what fits
   can be computed: at n19 the free VRAM is 11 784 MiB, which is `-ub 12288` with
   1229 MiB to spare — too thin given what happened to 6144. Hence n20 for 12288
   and n21 for 16384, being swept now.
4. ~~**Threads.**~~ **DONE 2026-08-17 — the two split cleanly, and the current
   values are already right.**

   | arm | pp 4k / 32k | tg 4k / 32k |
   |---|---:|---:|
   | `-t 24 -tb 24` (shipped) | 1334.3 / 1795.3 | 20.05 / **19.20** |
   | `-t 24 -tb 8` | 1331.2 / 1794.7 | 20.05 / 19.09 |
   | `-t 24 -tb 4` | 1315.9 / 1791.9 | 20.18 / 19.16 |
   | `-t 8 -tb 24` | 1353.4 / 1801.5 | 19.23 / **18.04** |

   **`-tb` is inert across 4 to 24** — 0.03 % at 32k between 24 and 8. That is a
   third, independent confirmation of the mechanism: prefill genuinely does not
   run on the CPU any more. §16 measured +32 % for more prefill threads in the
   `-rtr` path; here the knob does nothing.

   **`-t` still matters**: 8 threads costs 4.1 % of generation at 4k and 6.0 % at
   32k. Decode ships little across the link, so host-side work still shows.

   Practical note worth keeping: `-tb` can be dropped to 8 for free if the CPU is
   wanted elsewhere while the server runs. Nothing is gained by it, but nothing is
   lost either.

**Before this can become the default:** the NaN-logits abort (item 9) was
characterised entirely in the `-rtr` CPU path. `-ub 4096` here is far above the
`-ub 2048` that aborts there, but it is a different code path and proves nothing
either way. It needs its own soak — one abort per 100-330k prefilled tokens was
the rate in the other path, and at ~1500 tok/s that band is reached in minutes.

---

## 12. Unfinished measurements — one answered since, two need re-planning

Each of these got exactly one data point before the session running it went away.
**The machine has since changed twice**, so this is not simply "resume": the RAM
went 122 -> 91 -> 244 GiB, which moves every headroom figure the plans below were
built on. Re-derive before running.

All three need the GPU for roughly an hour in total, and each got exactly one
data point before the session that started it went away. The single points are
recorded so nobody re-derives them, but none of them decides anything.

### 12.1 ~~Does RAM latency matter?~~ — ANSWERED, and the kit is gone

**Answered by §26 rather than by finishing this run.** Three bandwidth points on
the same machine now say generation tracks MT/s and nothing else:

| MT/s | bandwidth | tg at 32k |
|---:|---:|---:|
| 6267 | 100.3 GB/s | 19.18 |
| 6400 | 102.4 | **19.18** |
| 7400 (dual DIMM) | 118.4 | 20.22 |

+2 % of bandwidth bought 0 %; +18 % bought 5.4 %. Capacity, rank count and the
move from four DIMMs to two changed nothing on their own.

The two partial readings taken on the tight-timing kit agree: 4k came out at
1356.6 pp / 21.08 tg on one attempt and 1384.4 / 21.39 on the next, against
1382.1 / 21.23 for the looser kit at the same 7400 — i.e. **the two runs of the
same kit differ by more than the two kits differ**, which is what "no effect"
looks like.

Cannot be finished as designed regardless: that kit is no longer in the machine.
If it ever matters, the honest version needs both kits present and several
repeats, because the effect being looked for is smaller than the noise.

**Practical conclusion for buying memory for this workload: MT/s is the only
number that matters.** CAS can be ignored; capacity buys context and
`--n-cpu-moe` headroom, not speed.

### 12.2 ~~Convert the 256k profile~~ — DONE 2026-08-19, shipped at n21 / ub 8192

`deepseek-v4-flash-256k-kvram` still runs `-rtr 1` at `--n-cpu-moe 18`,
`-ub 2048` — the CPU path. Same model and quant as the 128k profile, which gained
**3.7x** from the conversion (§21), so this is the largest unclaimed win in the
repository.

Baseline measured before the interruption: **4k -> 455.8 pp / 21.17 tg**.

**The VRAM headroom below still holds** — it is a function of the card, not the
system RAM — but the *host* side has changed completely. These arms were planned
when the machine had 91 GiB and `--n-cpu-moe` was capped by system memory as much
as by VRAM. At 244 GiB that constraint is gone: even `--n-cpu-moe 24` (76.5 GiB of
host weights) leaves ~165 GiB free, so the sweep can go higher than planned if the
VRAM side allows it.

Host weights follow `1010 + n * 3264` MiB, so at `-ub 8192` (compute buffer 7040):

| `--n-cpu-moe` | free VRAM | spare |
|---:|---:|---:|
| 18 | 8 520 | 1 480 — too thin |
| 19 | 11 784 | 4 744 — exactly what survived at 128k |
| 20 | 15 048 | 8 008 |
| 21 | 18 312 | 11 272 |

    ./tools/sweep.sh deepseek-v4-flash-256k-kvram 4096,32768,128000 \
        "baseline rtr n18 ub2048:" \
        "gpu-experts n19 ub8192:IK_RTR=0,IK_NCMOE=19,IK_UBATCH=8192,IK_BATCH=8192,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo" \
        "gpu-experts n20 ub8192:…" "gpu-experts n21 ub8192:…"

### 12.3 ~~The MTP variants~~ — DONE 2026-08-19, they do not fit (RESULTS §27.4)

`deepseek-v4-flash-128k-kvram-mtp` and `-256k-kvram-mtp` also carry `-rtr 1`.
Converting them is worth testing but **the hypothesis is different**: they exist
for speculative decoding, i.e. for *generation*, and `gpu-experts` costs 3-7 % of
generation while buying prefill (§21). So the conversion may improve the axis
these profiles do not care about and damage the one they do.

Judge them on **tg, not pp**. If tg drops, leave them on `-rtr` and say so in the
profile, which is a useful result rather than a failed one.

Two things to recompute first, both learned the hard way:

* the draft model is 5.5 GB and sits in VRAM (`-ngld 99`), so it comes off the
  headroom before the compute buffer does. At `-ub 8192` that pushes the floor to
  `--n-cpu-moe 21`, not 19 — plan an arm at `-ub 4096` too, in case 8192 will not
  fit at all.
* do **not** override `IK_EXTRA_ARGS` for these arms. It carries
  `--spec-type mtp:n_max=1,p_min=0.0`, which contains a comma, and `sweep.sh`
  splits overrides on commas — the result would quietly be a run with no
  speculative decoding at all, measured as if it were MTP.

### 12.4 Also open: delete the 114 GB `_R4` GGUF?

`step-3.7-flash-q4-r4` needs `Step-3.7-Flash-UD-Q4_K_XL-R4-*.gguf`, 114 GB, beside
the 114 GB non-R4 file that `step-3.7-flash-q4` uses — the same model twice,
differing only in weight layout. On this hardware the R4 copy is dominated (item
8), so the disk is arguably wasted. Not deleted: it is irreversible and a repack
costs hours. Matt's call.

---

## 13. ~~Do context checkpoints cost prefill throughput?~~ — YES, 11-27 % (RESULTS §30)

> Measured on the 262144 profile: `-ctx-ckpt 0` is worth +26.8 % prefill at 4k,
> +15.0 % at 32k, +11.3 % at 128k, and ~8 % of generation at every depth. A third
> arm showed the **prompt cache costs nothing** — the whole toll is checkpoint
> creation.
>
> Not acted on, because `depthbench` salts every prompt and therefore measures the
> cost with none of the benefit (§11.3: checkpoints are what make a re-send free).
> Whether to keep them depends on prefix-reuse rate — see item 14.

## 13 (historical). Do context checkpoints cost prefill throughput?

§29.5 noticed that the 524288 profile out-prefills the 262144 one at both 32k and
128k (1721 vs 1637, 1335 vs 1259) **despite four more expert layers in host RAM**,
which §22.2 prices at ~7.6 %. Something is paying that back and then some.

The one configuration difference is that the 512k profile ships `-ctx-ckpt 0
--cache-ram 0` while the 256k profile leaves both on. Building a checkpoint is
~872 MiB of copying at 131072 and ~1744 at 262144, and it happens during prefill.

**One run settles it**: the 256k profile with `-ctx-ckpt 0` added, against its own
current numbers. If prefill jumps, checkpoints have a throughput cost that §11.3
never measured — it only measured what they buy on a re-send — and every profile's
`-ctx-ckpt` becomes a real trade rather than a default.

---

## 14. ~~Prefix-reuse rate of Hermes traffic~~ — ANSWERED 2026-08-19, keep checkpoints

> `tools/ckpt-value.sh` (new) prices the trade in seconds off the server's own log.
> Two real Hermes runs:
>
>     created  372  costing  117.7 s
>     restored  17  costing    1.1 s
>     prefill avoided  609 870 tokens = 1105.6 s
>     net  +987 s
>
> Only **4.6 % of checkpoints are ever used** — but creation is cheap and constant
> (~320 ms) while a restore returns work proportional to depth (one recovered
> 52 655 tokens, over 90 s of prefill). The ratio is 8.4 : 1 in favour of keeping
> them, so §30's cost figure must not be acted on alone: `depthbench` salts every
> prompt and measures the cost with none of the benefit.
>
> **Nothing changed.** A trap worth recording: summing `n_past_prompt` gives 75.3 %
> "reuse", which is the wrong measure — ordinary conversation continuation reuses
> the slot's KV for free and needs no checkpoint. The source only consults
> checkpoints on prompt *divergence*, so only `restored context checkpoint` lines
> count.

§30 turned `-ctx-ckpt` into a real trade rather than a default: checkpoints cost
11-27 % of prefill throughput and ~8 % of generation, and buy up to 100 % of
prefill back on any prompt whose prefix repeats. Which side wins is a property of
the workload, and this repository has never measured it.

It does not need a benchmark. A normal working day's `logs/server-*.log` already
contains it — every request logs `n_past_prompt` (how much of the prompt was
already resident) against `prompt_n` (how much had to be prefilled). The ratio
over a day is the answer.

    reuse rate > ~20 %  ->  keep checkpoints, they pay for themselves
    reuse rate < ~20 %  ->  -ctx-ckpt 0 on every profile, worth 11-27 % prefill

Worth doing on real Hermes traffic rather than a synthetic run, since the whole
question is about how much agent conversations actually share.

---

## 15. ~~Give 131072 the same `--cache-ram` ceiling as 262144?~~ — NO, 2026-08-19

> Matt's call, and a reasonable one: at 131072 a checkpoint is 872 MiB, so even
> the full default 32 is ~27 GiB on a 244 GiB machine, and the profile has never
> come close to trouble. The ceiling on 262144 exists because a checkpoint there
> is twice the size and the 524288 arm proved what happens without one; that
> argument does not carry down to 131072.

## 15 (historical). Give 131072 the same `--cache-ram` ceiling as 262144?

§31.3 added `--cache-ram 32768` to the 262144 profile after a 524288 arm with no
ceiling reached 231 GiB RSS and was OOM-killed along with the editor. The 131072
profile still runs the upstream default of 8192 MiB, which upstream does not
enforce when the cache holds one prompt (that is the #2320 bug our local patch
fixes).

At 131072 a checkpoint is 872 MiB, so the default 32 authorises ~27 GiB. That is
survivable on 244 GiB and has never caused trouble, so this is tidiness rather
than urgency — but the profile currently relies on the workload never filling it,
which is the same assumption that failed at 524288.

Cheap to settle: pick a ceiling that keeps ~19 checkpoints (~16 GiB) and confirm
with `tools/ckpt-value.sh` that the restore rate does not fall.

