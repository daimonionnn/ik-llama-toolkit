# Follow-up comment for ikawrakow/ik_llama.cpp#2344

*Drafted 2026-09-03 after the issue was closed, shortened and re-based on the
600 W numbers 2026-09-04 (RESULTS §49.9–§49.10). The patch it refers to is
`patches/mask-share-upstream.patch`; links point at `blob/main`. Posted
2026-09-04 as
<https://github.com/ikawrakow/ik_llama.cpp/issues/2344#issuecomment-5532719298>;
the body below the rule is what was posted, the long first draft is in git
history.*

---

One follow-up with a measurement at your test point, because it led to a patch.

**Your observation reproduces, and it is the top of a staircase.** Same probe as before (`-c 131072 -fa on -ctk/-ctv f16 -b 4096 -ub 4096`, KV on GPU, `--swa-compress`, sm_120), CUDA0 compute buffer by `-ncmoe`: `-cmoe` 2 978.89 MiB, `42` 2 978.89, **`38` 2 978.18 — constant, as you saw**, `35` 3 490.18 (+512), `28` 5 026.18 (+2 048), `21` 7 074.18 (+4 096). Flat up to the deepest placement your VRAM allows, then +512 MiB = n_ctx × n_ubatch bytes per step.

**Cause.** Three host-side masks get a fresh view per layer in `build_deepseek4.cpp`: the CSA mask (twice per CSA layer — `dsv4_build_raw_mask_view` for `MASK_TOPK`, the view in `dsv4_build_lid_top_k_shared` for `INDEXER_TOPK`), the HCA mask, and the raw/SWA mask (the `mask_base1` branch makes a full `n_ctx × n_tokens` view plus a `ggml_cont` of it per layer whenever the raw plan is empty; the `first > 0` window cut makes a strided view per layer at runtime). `ggml_backend_sched` keys split inputs by tensor, so every view becomes its own device copy, allocated at the start of its split (`ggml-backend.cpp:1916`). With `-cmoe` each layer is a split and the copies die one at a time; N contiguous GPU-resident expert layers are one split, and they stack. `GGML_SCHED_DEBUG=1` on a 5k-token prompt at `-ncmoe 28`, no `--swa-compress`: **43 raw + 42 CSA + 20 HCA = 105 mask copies per u-batch and per generated token**.

They cost time as well as space: `ggml_backend_cuda_cpy_tensor_async` declines a CPU source, so each copy is a `ggml_backend_synchronize` + blocking `ggml_backend_tensor_copy` of the view's *span* — the whole mask. At 122k depth and `-ub 8192` that is ~59 GB of host→device traffic per u-batch, with the GPU idle.

**Patch** ([mask-share-upstream.patch](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/docs/external/patches/mask-share-upstream.patch), +73 −8, applies to `15dddc60` with `git apply`): return `mask` itself in the three places where the view would be the whole mask, and cut the SWA window once per graph instead of once per layer (a 4-field cache in `dsv4_runtime::input_state`, reset at the top of `build_deepseek4()`; `first`, `nton` and the mask are the same for every layer of a graph). `IK_MASK_SHARE=0` restores the old graph in the same binary. Mask copies per u-batch / per token: 105 → 3.

Reserve buffers at 131072, old → patched:

| placement | `-ub` | `--swa-compress` | old | patched |
|---|---|---|---|---|
| `-cmoe` | 8192 | no | 8 768.06 (your number) | 5 507.83 |
| `-ncmoe 28` | 4096 | no | 22 336.18 | 3 519.94 |
| `-ncmoe 21` | 8192 | yes | 14 212.36 | 4 616.36 |
| `-ncmoe 21` | 4096 | yes | 7 074.18 | 3 106.18 |
| `-nkvo -ncmoe 19` | 8192 | no | 7 040.03 | 4 991.80 |
| `-cmoe` | 4096 | yes | 2 978.89 | 3 298.03 (+319) |

The `+319` (and +638 at `-ub 8192`) with `-cmoe --swa-compress` is the trade-off: the shared copy now lives until its last consumer instead of dying with each layer's split, so one CSA mask + one HCA mask stay resident. Everything else drops, and the buffer no longer depends on expert placement.

Throughput, same binary, `-ub 8192`, 131072, temperature 0, `-nkvo -ncmoe 19` (my production profile), patched / old:

| depth | prefill t/s | generation t/s |
|---|---|---|
| ~20k | 1 852.2 / 1 766.5 (+4.9 %) | 19.95 / 19.38 (+2.9 %) |
| ~122k | **1 710.6 / 1 382.1 (+23.8 %)** | **18.99 / 17.25 (+10.1 %)** |

Completions are byte-identical patched vs old at temperature 0 (13.6k-token prompt, two u-batches; this profile and `--swa-compress -ncmoe 21`).

What the patch leaves: the window cut is still a strided view of the host mask, so one span copy per graph remains (~1–2 GiB at reserve). A CPU-side `ggml_cont` would remove it, but that needs the build callback to pin one named tensor the way `kqv_merged_cont` is pinned for `-nkvo`, so I left it to you. Happy to open a PR if you prefer that form.

Dumps and arithmetic: [RESULTS.md](https://github.com/daimonionnn/ik-llama-toolkit/blob/main/docs/RESULTS.md) §49.9–§49.10.

---
*Driven together with Claude (via Claude Code), as before.*
