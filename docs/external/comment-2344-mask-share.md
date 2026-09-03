# Follow-up comment prepared for ikawrakow/ik_llama.cpp#2344

*Drafted 2026-09-03 after the issue was closed. NOT POSTED — waits for an
explicit go. Numbers from RESULTS §49.9–§49.10; the patch it refers to is
`patches/mask-share-upstream.patch`; links point at `blob/main`, so they are
live from the first push.*

---

One follow-up, because the closing note deserved a measurement at your test point rather than an argument — and the measurement led to a patch that I think you will want to look at.

**Your observation reproduces here.** Same probe as before, `-c 131072 -fa on -ctk/-ctv f16 -b 4096 -ub 4096`, KV on GPU, `--swa-compress`, sm_120:

| `-ncmoe` | GPU-resident expert layers | CUDA0 compute buffer |
|---|---|---|
| `-cmoe` | 0 | 2 978.89 MiB |
| 42 | 1 | 2 978.89 |
| **38** | 5 | **2 978.18 — constant, as you saw** |
| 35 | 8 | 3 490.18 (+512) |
| 28 | 15 | 5 026.18 (+2 048) |
| 21 | 22 | 7 074.18 (+4 096) |

(Without `--swa-compress` the `-cmoe → -ncmoe 38` step alone is 4 224 → 9 536, so I assume you ran with it.) The buffer is flat up to the deepest placement your VRAM allows and starts growing one step below it, in exact multiples of 512 MiB = n_ctx × n_ubatch bytes.

**What it is.** Three host-side mask inputs get a fresh view per layer in `build_deepseek4.cpp`: the CSA mask (twice per CSA layer — `dsv4_build_raw_mask_view` for `MASK_TOPK`, the view in `dsv4_build_lid_top_k_shared` for `INDEXER_TOPK`), the HCA mask (same function), and the raw/SWA mask — the `mask_base1` branch makes a full `n_ctx × n_tokens` view *plus* a `ggml_cont` of it per layer whenever the raw plan is empty (the reserve graph, and every prefill without `--swa-compress`), and the `first > 0` window cut makes a strided view per layer at runtime. `ggml_backend_sched` keys split inputs by tensor, so each view becomes its own device copy, and `ggml_backend_sched_split_graph` places all of a split's input copies at the start of the split (`ggml-backend.cpp:1916`). With `-cmoe` each layer is its own split and the copies die one at a time; with N contiguous GPU-resident expert layers — one big split — they stack. `GGML_SCHED_DEBUG=1` on a real 5k-token prompt at `-ncmoe 28`, no `--swa-compress`, `-ub 4096`, counted over all 58 splits: **43 raw + 42 CSA + 20 HCA = 105 mask copies per u-batch and per generated token** (85 at reserve; the 16 layers inside the big split alone stack 14 × 256 MiB + 16 × 1 024 MiB = 19 968 MiB at its start, which is most of the 22 336 MiB buffer).

The copies are also not free at runtime: `ggml_backend_cuda_cpy_tensor_async` declines a CPU source, so each one goes through `ggml_backend_synchronize` + blocking `ggml_backend_tensor_copy`, and for a strided view `ggml_nbytes` is the span — the whole mask. At 122k depth and `-ub 8192` that is 43 × ~1 GB (mean over the prompt) for the raw window alone, ~59 GB per u-batch including CSA/HCA, at the 50 GB/s a pinned H2D copy does on this box.

**Patch** ([mask-share-upstream.patch](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/docs/external/patches/mask-share-upstream.patch), +73 −8 on two files, applies to `15dddc60` with `git apply`): return `mask` itself in the three places where the view would be the whole mask (`dsv4_build_raw_mask_view` both branches, `dsv4_build_lid_top_k_shared`), and cut the SWA window once per graph instead of once per layer (a 4-field cache in `dsv4_runtime::input_state`, reset at the top of `build_deepseek4()`; `first`, `nton` and the mask are the same for all layers of a graph). `IK_MASK_SHARE=0` restores the old graph from the same binary, which is how the A/B below was done. Mask copies per u-batch / per token: 105 → 3.

Reserve buffers, old → patched, 131072:

| placement | `-ub` | `--swa-compress` | old | patched |
|---|---|---|---|---|
| `-cmoe` | 8192 | no | **8 768.06** (your number) | **5 507.83** |
| `-cmoe` | 4096 | no | 4 224.03 | 3 328.02 |
| `-ncmoe 28` | 4096 | no | 22 336.18 | **3 519.94** |
| `-ncmoe 19` | 8192 | no | 57 984 (OOM) | 5 380.13 |
| `-ncmoe 21` | 8192 | yes | 14 212.36 | **4 616.36** |
| `-ncmoe 21` | 4096 | yes | 7 074.18 | 3 106.18 |
| `-cmoe` | 4096 | yes | 2 978.89 | 3 298.03 (+319) |
| `-cmoe` | 8192 | yes | 4 549.77 | 5 188.06 (+638) |
| `-nkvo -ncmoe 19` | 8192 | no | 7 040.03 | 4 991.80 |

The two `+` rows are the trade-off: with `-cmoe --swa-compress` the shared copy now lives until its last consumer instead of dying with each layer's split, so one CSA mask + one HCA mask stay resident. Everything else drops, and the buffer no longer depends on expert placement.

Throughput, same binary, `-ub 8192`, 131072, 2 repeats per point, temperature 0 (`-nkvo -ncmoe 19`, my production profile; patched / old):

| depth | prefill t/s | generation t/s |
|---|---|---|
| ~20k | 1 773.0 / 1 699.5 (+4.3 %) | 19.96 / 19.41 (+2.8 %) |
| ~122k | **1 618.9 / 1 360.9 (+19.0 %)** | **19.12 / 17.27 (+10.7 %)** |

Depth-proportional, as an H2D volume that scales with n_kv has to be. Completions are byte-identical patched vs old at temperature 0 (13.6k-token prompt, two u-batches, both this profile and `--swa-compress -ncmoe 21`).

One thing the patch leaves: the window cut is still a strided view of the host mask, so one span copy per graph remains (~1–2 GiB at reserve; the residual gap between the 3 519.94 and 3 106.18 rows). Doing the `ggml_cont` on the CPU side (34 / 68 MiB) would remove it, but that needs the build callback to pin one named tensor the way `kqv_merged_cont` is pinned for `-nkvo`, so I left it to you. Happy to open a PR if you prefer that form.

Write-up with the dumps and the arithmetic: [RESULTS.md](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/docs/RESULTS.md) §49.9–§49.10.

---
*Driven together with Claude (via Claude Code), as before.*
