# Local patches against ik_llama.cpp

Everything this project changes in the `ik_llama.cpp` clone, split into what is
carried forward and what was diagnostic scaffolding for the NaN-logits hunt
(RESULTS §19, §32–§49).

Base after the 2026-08-31 update: upstream `15dddc60`
(previously `8337e4cd`, where the whole investigation ran).

## Carried forward — 236 lines

Apply in table order; `keep-mask-share.patch` is generated on top of
`keep-build_deepseek4.patch` (they touch the same window-view lines).

| patch | what it does |
|---|---|
| `keep-build_deepseek4.patch` | **the NaN fix.** The DSV4 raw-branch SWA window view derives its start from the 256-padded KV length while the mask rows are indexed from `kv_head`; when the padding slack exceeds the window slack the view starts past what the leading rows need and they arrive at `FLASH_ATTN_EXT` entirely `-inf`, which becomes NaN. Clamps the start to the oldest cell row 0 still needs. RESULTS §49.2 |
| `keep-mask-share.patch` | **the compute-buffer / bandwidth fix.** Three host-side masks (CSA ×2 per CSA layer, HCA, and the raw/SWA window of every layer) were re-viewed per layer; `ggml_backend_sched` copies each view separately, at the start of its split, through the blocking host→CUDA path, span of the view = the whole mask. Returns the mask itself where the view is the whole mask and cuts the SWA window once per graph (`lctx.dsv4.inputs.raw_mask_window`, `src/llama-context.h`). 105 → 3 mask copies per u-batch and per token; `-ncmoe 28` at 131072 / `-ub 4096` 22 336 → 3 520 MiB, the shipped `-nkvo -ncmoe 19` 7 040 → 4 992, prefill at 122k +19 %. `IK_MASK_SHARE=0` restores the old graph for A/B. RESULTS §49.9–§49.10 |
| `keep-llama-sampling.patch` | throw instead of `GGML_ABORT` when every candidate logit is NaN, so the server survives; also writes a uniquely-named probability dump. §33 |
| `keep-server-context.patch` | on that throw, drop the slot's KV, cache tokens and checkpoints instead of letting the next request build on state that demonstrably went wrong. §33 |
| `keep-server-task.patch` | `--cache-ram` limit. §31 |
| `keep-fattn-new-mma.patch` | guards three unprotected `0/0` divisions in the new-MMA FA kernel (combine, stream-k fixup, main-kernel rowsum). Hardening only — none of them was the cause. §48.1 |

`mask-share-upstream.patch` is the same change rebased onto the upstream
tree without the NaN clamp — the version to hand to ikawrakow (applies to
`15dddc60` with `git apply`).

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
