# How this box compares

DeepSeek-V4-Flash on other hardware, chiefly the 2× DGX Spark TP=2 setups that
have become the popular way to run it. Collected 2026-08-19.

> **Read the provenance column before the numbers.** Everything in §1 was measured
> here with `tools/depthbench.sh` and can be reproduced from this repository.
> Everything in §2 and §3 is third-party, taken from public write-ups, measured on
> hardware nobody here has touched, with engines and quants that differ from ours
> and from each other. They are not directly comparable and should not be treated
> as if they were.

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
