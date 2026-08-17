# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| baseline nkvo n18 | (profile defaults) | 1375.0 / 1526.8 | 20.66 / 19.63 | cb 3520.02 MiB, host 59762.00 MiB |
| no-nkvo n19 | IK_NCMOE=19,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek | — | — | **did not load** — `out of memory` |
| no-nkvo n20 | IK_NCMOE=20,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek | — | — | **did not load** — `out of memory` |
| no-nkvo n22 | IK_NCMOE=22,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek | — | — | **did not load** — `out of memory` |
