# Comment prepared for ikawrakow/ik_llama.cpp#2320

*(`Bug: --cache-ram size limit is not enforced?`, opened by @Skelectric. Not a new
issue — this adds an independent reproduction, a measurement that bears on the
"functions as designed" reading, and a tested patch.)*

---

Independent reproduction here, on a very different configuration from
@Skelectric's: `-c 131072` (not 1M), no `--swa-compress`, 96 GiB VRAM / 224 GiB
RAM. Same model, same outcome — so it is not specific to huge contexts or SWA.

```
 - cache state: 1 prompts, 18803.422 MiB (limits: 8192.000 MiB, 0 tokens, 23208 est)
   - prompt 0x72af040c5180:   53271 tokens,       0 discarded, checkpoints: 18, 18803.422 MiB
```

2.3× the default limit; another run reached 16 116 MiB.

## The part that may bear on "functions as designed"

Keeping one state regardless of the limit is clearly deliberate. But the size of
that one state is governed by a *second* limit which knows nothing about the
first, and I think the interaction is what makes the default surprising rather
than the rule itself.

**A checkpoint is a fixed size regardless of how many tokens it covers.** Measured
at `-c 131072` on DeepSeek-V4-Flash MXFP4:

```
  4 096 tokens -> 871.673 MiB
 12 288 tokens -> 871.735 MiB
 27 916 tokens -> 871.854 MiB
```

~872 MiB each, about 1/6 of the whole 5504 MiB KV cache. `server_prompt::size()`
counts them:

```cpp
size_t server_prompt::size() const {
    size_t res = data.size();
    for (const auto& checkpoint : checkpoints) res += checkpoint.size();
    return res;
}
```

So the default `--ctx-checkpoints 32` authorises **~27 GiB of checkpoints inside a
cache whose default byte budget is 8 GiB** — the two defaults are 3.4× apart, and
neither is expressed in terms of the other. At `-c 262144` a checkpoint is
~1744 MiB and 32 of them is ~55 GiB; that matches a 54 GiB RSS drop I measured
when turning both off.

Put another way: `--cache-ram` reads like a memory budget, but with one state it
is `--ctx-checkpoints` that actually decides the memory. A user lowering
`--cache-ram` to bound RAM gets no effect at all.

## A patch, if the strict-limit direction is acceptable

Checkpoints are an optimisation for partial reuse; `data` is the state itself. So
when a single state is over budget, trimming its checkpoints oldest-first keeps
the "always keep one state" rule intact while making the byte limit mean
something:

```diff
             states.pop_front();
         }
+
+        // The loop above cannot help when a *single* state is over the limit,
+        // and one easily is: a state carries up to --ctx-checkpoints checkpoints,
+        // and a checkpoint is a fixed size regardless of how many tokens it
+        // covers, so 32 of them can be several times limit_size.
+        //
+        // Checkpoints are an optimisation for partial reuse, not the state
+        // itself -- `data` is. So trim them oldest-first to get back under the
+        // limit, rather than letting the cache silently ignore it.
+        for (auto& state : states) {
+            while (size() > limit_size && !state.checkpoints.empty()) {
+                LLAMA_LOG_INFO(" - cache size limit reached, dropping a checkpoint (size = %.3f MiB)\n",
+                    state.checkpoints.front().size() / (1024.0 * 1024.0));
+
+                state.checkpoints.pop_front();
+            }
+        }
     }
```

Verified with `--cache-ram 2048 --ctx-checkpoints 8` and four distinct 20k-token
prompts:

```
  cache state records  : 3
  max cache size       : 1704.7 MiB   (limit 2048)
  checkpoints dropped  : 21
  whole prompts evicted: 2
  PASS: no record above the limit
```

It has since run four days of normal interactive use with the cache on, no ill
effects.

Trimming is only one option — capping how many checkpoints `alloc()` copies would
be another, and deriving one default from the other (or just documenting that
`--ctx-checkpoints` is the effective memory knob) might be enough. Happy to send
whichever you prefer as a PR, or to leave it here if you would rather @Skelectric
carry the PR since it is their issue.

**Environment:** ik_llama.cpp `8337e4cd` (found on `2cda8d2d`, still present),
DeepSeek-V4-Flash-0731 MXFP4, `-c 131072 -mla 3 -fidx -nkvo --n-cpu-moe 19 -ub 8192`,
RTX PRO 6000 Blackwell 96 GiB, driver 610.43.02, CUDA 13.3, Ubuntu 26.04.
