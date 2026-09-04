# Second follow-up for ikawrakow/ik_llama.cpp#2344

*Drafted 2026-09-04 after the 512k series (RESULTS §49.11–§49.13) and the
overnight soak of the mask-sharing build. NOT POSTED — waiting for Matt's go.
The patch it links is `patches/mask-share-window-upstream.patch`; the link
points at `blob/main`, so the file has to be pushed first. The body below the
rule is what would be posted.*

---

Three things since the comment above, in case they bear on whether you want the PR.

**1. The window cut on the CPU — done.** The one span copy the patch left (the `ggml_cont` of the strided SWA-window view lands on the GPU, so the scheduler copies the whole span) is now pinned to the CPU backend from `llama_build_graph`'s callback, the way `kqv_merged_cont` is for `-nkvo`; only the `n_swa × n_tokens` window crosses. At 131072 / `-ub 8192` that is worth 248 MiB and nothing in throughput — the span was never the buffer's peak there — at 524288 it is 11 135 → 7 304 MiB. On the single-token graph it costs 0.7–1.0 % generation (a CPU split and its two synchronisations to save a 16 MiB copy), so it applies to u-batches of ≥ 256 tokens (`IK_MASK_WINDOW_CPU`; the reserve graph and every prefill u-batch take it, the token graph does not). Both changes together against `15dddc60`, no other local code: [mask-share-window-upstream.patch](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/docs/external/patches/mask-share-window-upstream.patch) (+119 −8, three files).

**2. 524288, where the old graph hurt most.** `-c 524288 -nkvo -ub 8192 -b 8192 -ctk/-ctv f16 -mla 3 -fidx`, prompt cache off, same binary, patched / `IK_MASK_SHARE=0`, both at `-ncmoe 25` — the placement the old 21 376 MiB buffer forced:

| depth | prefill t/s | generation t/s |
|---|---|---|
| ~32k | 1 853.9 / 1 712.9 (+8.2 %) | 16.39 / 15.82 (+3.6 %) |
| ~128k | 1 654.6 / 1 330.1 (+24.4 %) | 15.80 / 14.50 (+9.0 %) |
| ~256k | **1 304.1 / 953.5 (+36.8 %)** | **15.08 / 13.19 (+14.3 %)** |

The gap grows with depth because the per-layer span copies were sized by the context, not the prompt. The buffer is 7 304 MiB patched, and the 14 GiB that freed put four more expert layers on the GPU: at `-ncmoe 21` the same profile does 1 965 / 18.39 at 32k, 1 741 / 17.61 at 128k, 1 359 / 16.80 at 256k, and prefills 499 951 tokens at 805 t/s (10.3 min). Against what it shipped with, +43 % prefill and +27 % generation at 256k.

**3. Stability.** The mask-sharing build (without the window cut) took a night of agent-shaped traffic on the production profile (`-nkvo -ncmoe 19 -ub 8192`, 131072): 7.05 M prefilled tokens over 646 requests in one server process — conversations grown turn by turn to 124k and branched back to non-256-aligned cache positions, the shape that produced every abort in this issue — no failed request, and a fixed ~3k-token probe re-sent at temperature 0 throughout returned the identical answer every time.

The PR offer stands. Numbers and method: [RESULTS.md](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/docs/RESULTS.md) §49.11–§49.13.

---
*Driven together with Claude (via Claude Code), as before.*
