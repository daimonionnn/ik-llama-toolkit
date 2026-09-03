# How this box compares

This machine against the 2× DGX Spark TP=2 setups that have become the popular
way to run large MoE models locally. Two models are covered, and **they reach
opposite verdicts**, which is the most useful thing in this document:

* **§1–§4 DeepSeek-V4-Flash** (collected 2026-08-19) — the Sparks win generation
  by 2–4×.
* **§5–§8 Qwen3.8-Flash-Next** (collected 2026-09-03) — this box wins, and at Q8
  it is a wash.

The difference is not the hardware, and it is not how much spills into DDR5
either — Qwen at Q8_0 spills *more* than DeepSeek and still generates twice as
fast. It is **expert geometry**: DeepSeek moves 2.75× more routed-expert weight
per token than Qwen does. §8 works that through.

> **This is one machine's answer, not a hardware review.** Every "measured"
> number here comes from a single RTX PRO 6000 Blackwell 96 GiB box with 244 GiB
> of DDR5, and the tuning that produced them is specific to that VRAM budget. The
> toolkit itself runs on any CUDA GPU (and on ROCm, Vulkan and Metal through
> ik_llama.cpp) — on a smaller card the same profiles simply push more expert
> layers to the host. What §8 shows is that the *verdict* moves with the model
> too, so do not read any row here as "hardware X beats hardware Y".
>
> **Read the provenance column before the numbers.** Everything in §1 and §5 was
> measured here — `tools/depthbench.sh` for DeepSeek, `llama-sweep-bench` for Qwen
> (RESULTS §51) — and can be reproduced from this repository. Everything in §2,
> §3 and §6 is third-party, taken from public write-ups, measured on hardware
> nobody here has touched, with engines and quants that differ from ours and from
> each other. They are not directly comparable and should not be treated as if
> they were.

# Part I — DeepSeek-V4-Flash

## 1. This machine — measured

RTX PRO 6000 Blackwell 96 GB, Core Ultra 7 270K Plus, 244 GiB DDR5-6667,
ik_llama.cpp, MXFP4, the `gpu-experts` profiles (RESULTS §21–§29).

| depth | prefill tok/s | generation t/s |
|---:|---:|---:|
| 4k | 1380 | 20.5 |
| 16k | **1910** | 21.1 |
| 32k | 1830 | 19.3 |
| 128k | 1346 | 17.4 |
| 32k, 524288 profile | 1721 | 16.3 |

## 2. 2× DGX Spark, TP=2 — published

Two GB10 nodes, 128 GB LPDDR5X each, joined over a 200 Gb QSFP56 link. The model
is split across both, so each node holds ~79 GiB entirely in its own unified
memory.

| configuration | depth | prefill | generation |
|---|---:|---:|---:|
| DSpark NVFP4, Stage C, speculative | 100k | **2639** | 84.3 peak / 67.6 mean |
| DSpark FP4, vLLM, k=5 | 2k | 1.8k ×2 | 47.4 ×2 |
| | 8k | 1.9k ×2 | 48.0 ×2 |
| | 32k | 1400 | 52.2 |
| | 128k | 1800 | 53.2 |
| official FP8, TP=2 | 8k | ~1970 | 40 |
| | 100k | ~1900 | 40 |
| | 200k | ~1750 | 38 |
| | 500k | ~1380 | 32 |
| FP8 + MTP (forum, arthurdroz) | 0 | 1098 | 37.3 |
| | 32k | **318** | 33.5 |
| | 65k | **176** | 29.5 |
| FP8, 256k, tuned (serapis) | 4k | 889 | 37.6 |
| 4 nodes, not 2 | ~50k | 2500 | 90 |

`×2` marks an aggregate over two concurrent streams rather than single-stream.

**The spread is the first thing to notice: 176 to 2639 tok/s of prefill on the
same two machines.** Fifteen-fold, and none of it is hardware. It is quantisation,
engine, and whether anyone tuned it. Any single "2× Spark does N tok/s" figure —
including the flattering ones — says more about that setup than about the box.

## 3. The same card, and near neighbours — published

| machine | engine + quant | prefill | generation |
|---|---|---:|---:|
| **RTX PRO 6000 — here, tuned** | ik_llama, MXFP4 | **1830** | 19.3 |
| RTX PRO 6000 — published | `ds4.c`, IQ2_XXS | 344 | 46.9 |
| Mac Studio M2 Ultra 192 GB | `ds4.c`, IQ2_XXS | 389 | 29.7 |
| 1× DGX Spark | `ds4.c`, IQ2_XXS | 410 | ~14 |

**The published figure for this card is 5.3× below what the card actually does**,
and at worse quality: 344 tok/s is `ds4.c` on a 2-bit quant, against 1830 on
effectively lossless MXFP4. Anyone comparing hardware from that table would
conclude a pair of Sparks is five times the RTX PRO 6000 at prefill. It is not.

That is not a criticism of the people who published it — `ds4.c` is what they had
running. It is a caution about cross-machine tables in general, this one included.

## 4. What the comparison actually shows

**Prefill: comparable.** This single card sits inside the band the Spark pairs
occupy — above the poorly-tuned FP8 runs, level with the good ones, below the
NVFP4 DSpark record. Against two machines.

**Generation: they win, by 2–4×, and it is structural.** Each Spark node holds
half the model in unified memory at ~273 GB/s, so the pair has ~546 GB/s against
this box's ~107 GB/s of DDR5 for the 63 GiB of experts that live in host RAM.
Decode is bandwidth-bound (RESULTS §25, §26), so that ratio is the answer. No
amount of tuning here closes it.

Two things would narrow it, neither free:
* **speculative decoding.** The DSpark runs use it (k=5); our MTP profiles measure
  27 t/s against 20 without (§27.4), but they cannot use the `gpu-experts`
  placement — the draft model's weights and KV leave too little VRAM.
* **faster host memory.** §26 fixed the exchange rate: +18 % of bandwidth bought
  +5.5 % of generation. Getting to Spark's decode from here is not a memory
  upgrade, it is a different memory architecture.

**Quality is not held constant anywhere in §2 or §3.** Ours is MXFP4, effectively
lossless QAT. Theirs is NVFP4 experts with FP8 attention, or full FP8, or a 2-bit
`ds4.c` quant. Comparing throughput across those is comparing different models.

# Part II — Qwen3.8-Flash-Next

Collected 2026-09-03, after the profiles in RESULTS §51 were built. The question
that prompted it: would trading this card for 2× DGX Spark be an upgrade, given
that Q8_0 spills ~94 GiB of experts into DDR5 here?

## 5. This machine — measured

RTX PRO 6000 Blackwell 96 GB, ik_llama.cpp, `llama-sweep-bench`, `-ctk/-ctv
q8_0`, `-t 8 -tb 24`, card otherwise idle. Full curves; the profiles are
`config/models/qwen38-flash-next-*.env`.

Headline figures in this repository take **prefill from the N_KV 2048 step** (the
N_KV 0 row is a cold first prefill and reads low) and **generation from the N_KV 0
step** (generation decays with depth, so the shallowest row is the honest one).
Both are shown here so the convention is visible rather than looking like a
discrepancy — see RESULTS §51.

| depth | Q4_K_M @128k pp | tg | Q8_0 @128k pp | tg | Q8_0 @256k pp | tg |
|---:|---:|---:|---:|---:|---:|---:|
| 0 *(cold pp)* | 3083 | **128.9** | 2193 | **40.6** | 2082 | **36.9** |
| 2 048 | **3486** | 118.9 | **2303** | 40.1 | **2163** | 35.6 |
| 8 192 | — | — | 2168 | 38.2 | 2048 | 34.5 |
| 16 384 | — | — | 2062 | 37.1 | 1936 | 33.4 |
| 32 768 | — | — | 1854 | 35.3 | 1757 | 32.4 |
| 49 152 | — | — | 1670 | 33.7 | 1580 | 30.7 |
| 75 776 | — | — | 1368 | 30.8 | — | — |
| ~96k | 1440 | 62.2 | — | — | — | — |

Q4_K_M keeps 79.7 GiB on the card and spills 33.9; Q8_0 keeps 83.6 and spills
95.9. That single difference is the whole 3.2× generation gap between the two
columns.

## 6. DGX Spark — published

**No one has published FP8 Qwen3.8-Flash-Next on 2× Spark.** Every dual-Spark
deployment found is NVFP4. The single-Spark runs offload to NVMe or SSD because
the model does not fit. Read the quant column carefully:

| configuration | quant | prefill | generation |
|---|---|---:|---:|
| 2× Spark TP=2, SGLang | NVFP4 | not published | 18.5–20 base, 41–50 MTP, ~70 peak |
| 2× Spark TP=2, 8 streams | NVFP4 | — | 119–149.5 aggregate |
| 1× Spark (blazux) | NVFP4 + NVMe offload | 1500–2000 | ~26 (MTP=2), 31 hybrid |
| 1× Spark (DwarfStar/ds4) | Q5 mixed + SSD-PLE | 1024 | ~28 |
| 1× Spark, **gpt-oss-120b** | MXFP4, 59 GiB, fits | 1956 → **1027** @32k | 60.6 → **40.6** @32k |

The last row is the honest reference point, and it matters more than the rest.
It is a comparable-class MoE (116.8 B, ~5.1 B active) that **fits entirely** in
one Spark's 128 GB, running mature llama.cpp rather than day-0 patched SGLang
kernels for a brand-new SM121 target. Against it this box is ~1.8× on prefill
and ~2.1× on generation — not the 3–7× the Qwen-specific NVFP4 numbers suggest.

**That 18.5–20 tok/s baseline is a software artifact, not a hardware ceiling.**
One Spark's 273 GB/s over ~3.4 GB of active NVFP4 weights allows ~80 tok/s;
the mature stack reaches 60.6 on gpt-oss, i.e. 60 % of nominal bandwidth. Any
conclusion drawn from the 19 tok/s figure is a conclusion about SGLang's
day-0 state.

## 7. The Q8 / FP8 question specifically

The intuition worth testing is "256 GB of unified memory lets the Sparks hold Q8
where this card cannot." It does not survive contact with two facts.

**First, FP8 is slower than NVFP4 on Spark for MoE.** Measured on a Spark MoE
(Qwen3.6-35B-A3B): **FP8 52.0 tok/s against NVFP4 66.9**, with NVFP4 also using
16 GB less. MoE fires many small per-expert kernels and the NVFP4 path captures
them into CUDA graphs better. So the Spark ecosystem pushes back down to 4 bits —
the same quality compromise, on hardware bought to avoid it.

**Second, the estimate lands on a tie.** Nobody has run it, so this is arithmetic
from their own measured efficiency (60 % of nominal, from the gpt-oss row):

| | this card, measured | 2× Spark FP8, **estimated** |
|---|---:|---:|
| bytes/token, ~6 B active | 6.38 GB @ Q8_0 | 6.00 GB, TP2 → 3.00 GB/rank |
| generation | **40.6 t/s** | 273 × 0.6 / 3.0 = 54.6, less TP2 sync → **~36–44** |
| prefill | **2193–2303 t/s** | **~1000–1500** |

Generation a wash inside the error bars; prefill ours by ~1.6–2×.

## 8. What the two models together show

**It is not spill, and it is not the hardware. It is expert geometry.**

The obvious explanation — "the verdict follows how much spills into DDR5" — is
wrong, and this table is what kills it:

| model + quant | on card | spilled | our tg | verdict vs 2× Spark |
|---|---:|---:|---:|---|
| DeepSeek-V4-Flash MXFP4 | ~83 GiB | ~63 GiB | 19.3 | **they win 2–4×** (§4) |
| Qwen3.8 Q8_0 | 83.6 GiB | **95.9 GiB** | 40.6 | wash |
| Qwen3.8 Q4_K_M | 79.7 GiB | 33.9 GiB | **128.9** | **we win ~2×** |

Qwen at Q8_0 spills half again as much as DeepSeek and still generates twice as
fast. Spill cannot be the mechanism.

**What decides it is routed-expert bytes read per token**, and the two models are
built very differently:

| | layers | experts used | n_embd | expert FFN | active expert params |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 43 | 6 of 256 | 4096 | 2048 | **6.49 B** |
| Qwen3.8-Flash-Next | 48 | 10 of 512 | 2560 | 640 | **2.36 B** |

DeepSeek uses fewer, fatter experts on a wider residual: 43 × 6 × 3 × 4096 ×
2048 against 48 × 10 × 3 × 2560 × 640. **It moves 2.75× more expert weight per
token.** That dominates everything else, including the quant:

| | bytes/token (routed experts) |
|---|---:|
| DeepSeek MXFP4 (4.40 bpw) | **3.57 GB** |
| Qwen Q8_0 (8.51 bpw) | 2.51 GB |
| Qwen Q4_K_M (5.387 bpw) | 1.59 GB |

Qwen at **Q8_0 — nearly double the bits per weight — still reads 30 % fewer bytes
per token than DeepSeek at MXFP4.** Decode is bandwidth-bound, so that is the
answer to why a much bigger file runs much faster.

Two secondary effects push the same way. Qwen's hybrid layout keeps full
attention on only 12 of 48 layers, so its KV cache is 2 224 MiB at 131072 where
DeepSeek needs 5 504 (and the `--swa-compress` / `-nkvo` gymnastics of §49 to
place it at all). And the smaller per-token read leaves more headroom before the
PCIe link saturates.

**So why do the Sparks win DeepSeek?** Because on their side nothing spills at
all — the pair's 256 GB of unified memory holds the whole model, every byte
served at 273 GB/s. Our box has to drag 43 % of a *large* per-token expert read
across PCIe from DDR5. When the per-token read is big, that penalty dominates and
they win. When it is small, our resident GDDR7 portion dominates and we win. The
crossover is a property of the model, not of either machine.

**Depth is a second, quieter advantage here.** Over 0 → 32k this box loses 15 %
prefill and 13 % generation on Qwen Q8; the Spark reference loses **47 % and
33 %**. Structural: our 2 224 MiB KV cache sits in GDDR7 and does not compete
with weights, while on Spark the cache and the weights draw on the same 273 GB/s
pool. The deeper the context, the wider this gets.

**Economics, September 2026.** NVIDIA lists this card at $16 000 (from $8 565 at
launch, GDDR7 shortage); 2× DGX Spark is $9 398. Trading down in price, down in
single-stream throughput, and up in operational complexity — two nodes and a
200 Gb fabric instead of one card.

**Where the Sparks would still be right:** a model too large to spill usefully
into 244 GiB of DDR5; concurrent serving, where 119–149.5 tok/s aggregate over 8
streams is genuinely strong; or 340 W and two silent boxes against 600 W in one.

**The real upgrade path here is software, not hardware.** `build_qwen4exp.cpp`
has no MTP — zero references, against ten architectures that have it. Speculative
decoding is where the Spark deployments get 2–3× (18.5 → 41–70 tok/s). The same
lever is unbuilt on our side, and it is worth more than any hardware in this
price range.
