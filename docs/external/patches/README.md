# Local patches against ik_llama.cpp

Everything this project changes in the `ik_llama.cpp` clone, split into what is
carried forward and what was diagnostic scaffolding for the NaN-logits hunt
(RESULTS §19, §32–§49).

Base after the 2026-09-05 update: upstream `fe215a8c`
(previously `15dddc60`, and `8337e4cd` where the whole investigation ran). All
seven `keep-*` patches re-applied to `fe215a8c` in table order without a
conflict; `keep-server-context.patch` was regenerated first, because the copy
committed on 2026-09-01 carried a hunk header claiming eleven added lines where
two were present and did not apply even to its own base.

## Carried forward — ≈ 280 lines

Apply in table order; `keep-mask-share.patch` is generated on top of
`keep-build_deepseek4.patch` (they touch the same window-view lines) and
`keep-window-cpu.patch` on top of both.

| patch | what it does |
|---|---|
| `keep-build_deepseek4.patch` | **the NaN fix.** The DSV4 raw-branch SWA window view derives its start from the 256-padded KV length while the mask rows are indexed from `kv_head`; when the padding slack exceeds the window slack the view starts past what the leading rows need and they arrive at `FLASH_ATTN_EXT` entirely `-inf`, which becomes NaN. Clamps the start to the oldest cell row 0 still needs. RESULTS §49.2 |
| `keep-mask-share.patch` | **the compute-buffer / bandwidth fix.** Three host-side masks (CSA ×2 per CSA layer, HCA, and the raw/SWA window of every layer) were re-viewed per layer; `ggml_backend_sched` copies each view separately, at the start of its split, through the blocking host→CUDA path, span of the view = the whole mask. Returns the mask itself where the view is the whole mask and cuts the SWA window once per graph (`lctx.dsv4.inputs.raw_mask_window`, `src/llama-context.h`). 105 → 3 mask copies per u-batch and per token; `-ncmoe 28` at 131072 / `-ub 4096` 22 336 → 3 520 MiB, the shipped `-nkvo -ncmoe 19` 7 040 → 4 992, prefill at 122k +24 % (+10 % generation). `IK_MASK_SHARE=0` restores the old graph for A/B. RESULTS §49.9–§49.10 |
| `keep-window-cpu.patch` | **the last mask copy, gated.** After `keep-mask-share` the scheduler still copies one strided span of the host mask per graph — 2 GiB at 131072 / `-ub 8192`, 8 GiB at 524288 — because the window's `ggml_cont` lands on the GPU. Names the cont `dsv4_raw_mask_window_cpu` and pins it to the CPU backend from `llama_build_graph`'s callback (`src/llama-build-context.cpp`), so only the 132 MiB window crosses. Worth 248 MiB and no throughput at 131072 (the span was never the buffer's peak), 3.8 GiB at 524288; on the token graph it costs −1 % generation, so it applies to u-batches of ≥ `IK_MASK_WINDOW_CPU` tokens (default 256; 0 = off). RESULTS §49.11 |
| `keep-llama-sampling.patch` | throw instead of `GGML_ABORT` when every candidate logit is NaN, so the server survives; also writes a uniquely-named probability dump. §33 |
| `keep-server-context.patch` | on that throw, drop the slot's KV, cache tokens and checkpoints instead of letting the next request build on state that demonstrably went wrong. §33 |
| `keep-server-task.patch` | `--cache-ram` limit. §31 |
| `keep-fattn-new-mma.patch` | guards three unprotected `0/0` divisions in the new-MMA FA kernel (combine, stream-k fixup, main-kernel rowsum). Hardening only — none of them was the cause. §48.1 |

`mask-share-upstream.patch` is the same change rebased onto the upstream
tree without the NaN clamp — the version handed to ikawrakow in the #2344
comment (applies to `15dddc60` with `git apply`).
`mask-share-window-upstream.patch` is that plus `keep-window-cpu` (+119 −8,
three files), generated 2026-09-04 by applying the two on a clean `15dddc60`
worktree; it re-applies clean there, applies clean to `fe215a8c` as well
(checked 2026-09-05), and carries no clamp.

## Archived — `diagnostics-full-8337e4cd.patch`, 1982 lines

The instrumentation that found the bug, kept because it is reusable and because
the upstream author may want to see how the evidence was produced:

* **in-stream NaN probes** (`ggml-cuda.cu`) — first NaN by node in stream order, plus input-side probes on `FLASH_ATTN_EXT` launched on the compute stream immediately before the node, so a race stays live. §44.5
* **deposit checksums** (`IK_SUM_CHECK`) — every split-input copy summed right after its H2D deposit and re-summed before each consumer, compared on device, no host synchronisation. 49.8 M verifications, zero mismatches — this is what killed the "something overwrites the copy" hypothesis. §46
* **FA double-run** (`IK_FA_TWICE`) — runs each prefill `FLASH_ATTN_EXT` twice on identical inputs and compares output sums, proving the kernel was bit-deterministic. §46.2
* **mask scanner** (`IK_MASK_SCAN`) — latches fully-masked mask rows in-stream. §46.3
* **origin staging** — pinned-host copies of `fattn-0`'s inputs refreshed every prefill graph, so a NaN event yields the exact bytes the kernel read, immune to allocator recycling. This is what produced the reproducer. §47.5
* **dump-on-abort / walk-back replay** (`server-context.cpp`, `llama.cpp`) — saves tokens and layer-0/1 KV at abort time and replays the request's chunks to find the origin. §47–47.3
* **`IK_SWA_DBG`** — logs `kv_head / n_tokens / nton / first / offset` per window view; this is the probe that made the fault arithmetic visible and is the one worth keeping if anyone revisits this area. §49.2
* lifetime/overlap probes (`ggml-alloc.c`, `ggml-backend.cpp`) and the `raw_k` allocator guard — all refuted hypotheses, kept for the record. §41, §43.4

Standalone reproducers live outside this directory:
`tools/fattn-nan-repro.cpp` (self-contained, no model, ~1 s) and
`tools/fattn-repro.cpp` (replays a captured origin dump).

Applying the archive to a current tree will conflict; it is a record, not a
maintained patch. Recover it against `8337e4cd`.
