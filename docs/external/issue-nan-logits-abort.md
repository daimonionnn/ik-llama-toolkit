# Issue prepared for ikawrakow/ik_llama.cpp

*(Not yet filed. Title: "DeepSeek-V4-Flash: all-NaN logits abort under sustained
server load, survives the f32 DSA fix". Attach `docs/external/crashes/`.)*

---

`llama-sampling.cpp:745` aborts because every candidate logit is NaN. Nine times
now, across two builds, over 39 hours of ordinary interactive serving. The f32
DSA accumulation fix (#2311, `ff141691`) was expected to be the cause and is not:
the abort continues on a build that contains it, at a rate this data cannot
distinguish from the rate before it.

No minimal reproducer. What follows is everything that has been established, with
the negative results included, so that nobody repeats them.

## The signature is identical every time

    =============================== Failed to sample token
    Data has been stored in probabilities.txt
    llama-sampling.cpp:745: Fatal error

Nine of the ten `probabilities.txt` dumps survive; #8's was lost because the file
is written to the working directory under a fixed name, so the next abort
destroys it. All nine are the same:

    candidates->size: 40
    max  = nan
    sump = nan
    probabilities:
    0  38  nan  nan
    1  22  nan  nan
    2  10  nan  nan
    ...

Same 40 token ids, in the same order, in all nine — `38 22 10 34 26 18 20 4 24
12 …`, which is what a partial sort leaves behind when every comparison is false.

**The logit column is already NaN.** The poison arrives at the sampler; it is not
produced by it. So this is a decode-side problem and top-k/temperature are only
where it becomes fatal.

## The ten aborts

| # | started | build | `-rtr` | `-ub` | `n_ctx` | depth at abort | uptime | tasks |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | 08-13 07:45 | `7ebbb906` | on | 2048 | 131072 | 18 695 | 4.3 h | 99 |
| 2 | 08-13 15:24 | `2cda8d2d` | on | 2048 | 131072 | 27 916 | 0.5 h | 24 |
| 3 | 08-13 17:37 | `2cda8d2d` | on | 2048 | 131072 | 12 805 | 0.3 h | 6 |
| 4 | 08-13 18:36 | `2cda8d2d` | on | 2048 | 131072 | 11 161 | 0.2 h | 11 |
| 5 | 08-13 21:57 | `2cda8d2d` | off | 2048 | 131072 | 30 959 | 0.2 h | 25 |
| 6 | 08-14 06:23 | `2cda8d2d` | on | 2048 | 131072 | 76 076 | 6.0 h | 25 |
| 7 | 08-17 18:06 | `2cda8d2d` | off | 8192 | 131072 | 39 518 | 0.5 h | 71 |
| 8 | 08-18 12:53 | `8337e4cd` | off | 8192 | 131072 | 92 008 | 0.1 h | 12 |
| 9 | 08-19 18:50 | `8337e4cd` | off | 8192 | 131072 | 16 934 | 3.3 h | 105 |
| 10 | 08-20 08:33 | `8337e4cd` | off | 8192 | 131072 | 29 228 | 0.1 h | 2 |

Depth at abort spans 11 k to 92 k and correlates with nothing. It is not a
context-limit effect: `n_ctx` is 131072 throughout.

## What #2311 did and did not do

`ff141691` is in the running build — verified, not assumed:

    git merge-base --is-ancestor ff141691 HEAD    -> yes
    binary is 56 s newer than the last source change

| build | serving time | tasks | aborts | rate |
|---|---:|---:|---:|---|
| `2cda8d2d` / `7ebbb906` | 23.0 h | 518 | 7 | 1 per 3.3 h |
| `8337e4cd` | 7.6 h | 636 | 3 | 1 per 2.5 h |

**Three events cannot measure a rate.** The interval around 1-per-2.5 h is wide
enough that a large real improvement is entirely possible, and this table does
not exclude one. It excludes only the strong claim: the abort is not gone.

If it helps: the reasoning that made #2311 look like the answer was that f16
saturates at 65 504, an overflowing accumulator gives inf, inf−inf gives NaN, and
a poisoned tensor propagates to every logit — which matches all-NaN rather than
some-NaN, and matches the abort appearing in configurations that differ in
everything except attention. That reasoning still looks right in shape. It just
does not appear to be this code path, or not only this one.

## A backtrace, at last -- and what it shows

Abort #10 was the first caught with the server running under gdb, so there is
finally a stack. (`GGML_ABORT` does try: it forks gdb to attach to its own
parent, which Yama's ptrace_scope=1 refuses. The fallback to
`backtrace_symbols_fd` is gated on gdb exiting `EXIT_FAILURE`, and gdb exits 0
after a refused attach -- so on any ptrace_scope=1 host an abort prints nothing.
That is a small separate wart, and it is why reports 1-9 have no stack.)

```
#5  ggml_abort (file="...llama-sampling.cpp", line=745, fmt="Fatal error") at ggml.c:266
#6  llama_sample_token_with_rng_impl (...) at llama-sampling.cpp:745
#7  llama_sample_token_with_rng (...) at llama.cpp:12758
#8  llama_sampling_sample_impl (..., idx=4, ...) at common/sampling.cpp:556
#9  common_sampler_sample (...) at common/sampling.cpp:704
#10 server_context::process_batch_tokens (..., n_batch=8192) at server-context.cpp:4791
#11 server_context::update_slots (...) at server-context.cpp:4998
```

It confirms the logits arrive poisoned rather than being spoiled by the sampler,
which was already the reading from the dumps. It does not locate the producer --
`llama_decode` has returned by then.

**But it shows something worth acting on independently of the root cause.** The
call site at `server-context.cpp:4790` is already wrapped:

```cpp
try {
    id = common_sampler_sample(slot.ctx_sampling, ctx, tok_idx);
    common_sampler_accept(slot.ctx_sampling, ctx, id, true);
} catch (const std::exception & e) {
    LOG_ERROR("sampling failed, releasing slot", {...});
    send_error(slot, std::string("sampling error: ") + e.what(), ERROR_TYPE_SERVER);
    slot.release();
    ...
}
```

The server is *designed* to survive a failed sample: log it, fail that one
request, release the slot, carry on. That handler can never run, because the
failure path calls `GGML_ABORT` -> `ggml_abort` -> `abort()`, which is not a C++
exception. Frame #4 in the stack is `__GI_abort`.

So a condition the server already knows how to handle takes the whole process
down, and every other in-flight request with it. Throwing from
`llama_sample_token_with_rng_impl` instead of aborting -- or returning a sentinel
the caller turns into a throw -- would make the existing handler reachable and
turn "the server dies every few hours" into "one request returned a 500". The
dump to `probabilities.txt` can stay exactly as it is.

That is worth doing whether or not the NaN is ever explained, and it does not
prejudge the cause.

## A lead: #2311 may simply not reach this GPU

#2317 established that on Blackwell the DSA code lands on a *different branch*
than on Ampere — same executable, same request, `_c` on an RTX 5090 (sm_120) and
`_call` on an RTX 3090 (sm_86) in the same host. That issue was about an argmax
near-tie and its reporter withdrew the correctness interpretation, so this is not
a claim that #2317 is unresolved. It is one narrow fact taken from it: **sm_120
does not execute the same DSA path as sm_86.**

This build is compiled `-DCMAKE_CUDA_ARCHITECTURES=120-real`, sm_120 only, on an
RTX PRO 6000 Blackwell. So the obvious question is whether the f32 accumulation
of #2311 is actually in effect on this architecture, or whether sm_120 takes a
branch that still accumulates in f16. That would explain the whole shape of this
report: the fix is present in the source, verifiably compiled in, and changes
nothing about the abort rate.

We cannot test this here — there is only one GPU in this box and it is Blackwell.
If someone with both architectures can run the same load on sm_86, that is
probably the cheapest discriminating experiment available.

## Ruled out

Each of these was bisected with the abort still occurring:

* **`-rtr`** — on for five, off for four. (On this model `-rtr` decides whether
  the host experts are computed on the CPU as `MXFP4_R8` or streamed to the GPU;
  the two are entirely different code paths for the expert GEMM.)
* **`-ub`** — 1024, 2048 and 8192.
* **Prompt cache and context checkpoints** — see below, tested and not exonerated
  so much as untestable from this data.
* **Driver and hardware** — unchanged across #1–#7, and the machine had a PCIe
  link change and four RAM configurations between #7 and #9 with no effect on
  whether it happens.

## Two hypotheses that looked right and failed

Recorded because both are convincing from a crash excerpt alone:

1. **"A checkpoint operation immediately precedes every abort."** True in the
   logs, and nearly vacuous: run #9 created 293 checkpoints across 105 tasks.
   Almost every event in that log is immediately preceded by a checkpoint
   operation.

2. **"The prompt cache reports `n_past != n_past_prompt` just before the
   abort."** Present in four of the older excerpts, and looks like an off-by-N in
   cache reuse. But that mismatch occurs in 12 % of *ordinary* cache restores in
   run #9, and run #9's own abort does not have it (15 803 = 15 803).

## What has not been tested

* **Other context sizes.** All nine are at `n_ctx 131072`. 262144 has 7.1 h and
  524288 has 3.5 h of traffic with no abort, but at 1 per ~4 h that is far too
  little to mean anything.
* **A synthetic reproducer.** `tools/stress.sh` in our repo drives short prompts
  against a growing conversation at ~25x the token rate of real traffic; 610 k
  prefilled tokens came back clean. That was originally read as clearing the
  build. It is more honestly read as: this tool does not reproduce the abort, so
  it cannot clear anything.
* **Disabling the prompt cache entirely** for a long run. That is the next thing
  we will run here unless you would rather we try something else first.

## Related upstream

* **#2311** (merged 2026-08-13) — the f32 DSA accumulation fix this report is
  about. https://github.com/ikawrakow/ik_llama.cpp/pull/2311
* **#2317** (closed 2026-08-14) — sm_120 vs sm_86 DSA branch divergence, the lead
  above. https://github.com/ikawrakow/ik_llama.cpp/issues/2317
* **#1952, #1693, #1520** — earlier aborts at the same sampler line, on Qwen3.5,
  Nex-N2-Pro and Gemma-4 MoE. All closed, all different root causes. The line is
  simply where any all-NaN logit vector becomes fatal, so a match there means
  little on its own.

No open issue covers this. Searched `Failed to sample token`, `NaN`, and
`V4-Flash` across the repository on 2026-08-20.

## Environment

ik_llama.cpp `8337e4cd` (build 102), DeepSeek-V4-Flash-0731 MXFP4 (145.6 GiB,
4-file split, lmstudio-community), `-c 131072 -mla 3 -fidx -nkvo --n-cpu-moe 19
-ub 8192 -b 8192 -ctk f16 -ctv f16 --reasoning-format deepseek`, RTX PRO 6000
Blackwell 96 GiB (sm_120), Core Ultra 7 270K Plus (AVX2, no AVX-512), 244 GiB
DDR5, driver 610.43.02, CUDA 13.3, Ubuntu 26.04.

Traffic is a coding agent: many requests, conversations that grow to tens of
thousands of tokens, heavy prompt-cache reuse.

## Attached

`crash-20260813-1204` through `crash10` context logs and nine
`probabilities.txt` dumps, plus `crash10-backtrace.log` -- the full stack, all 54
threads. Full server logs for any of the ten are available on request; they are
large, so they are not attached by default.

Happy to run patches, instrumented builds, or a bisect here; the machine
reproduces this every few hours without any effort, which makes it a reasonable
place to test a fix even though it is a poor place to find one.
