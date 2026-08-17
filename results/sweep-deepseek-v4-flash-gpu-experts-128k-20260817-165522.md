# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| no-nkvo n18 amb512 | IK_NCMOE=18,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -amb 512 | — | — | **did not load** — `out of memory` |
| no-nkvo n19 amb512 | IK_NCMOE=19,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -amb 512 | — | — | **did not load** — `out of memory` |
| no-nkvo n20 amb512 | IK_NCMOE=20,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -amb 512 | — | — | **did not load** — `out of memory` |
| nkvo n18 amb512 | IK_NCMOE=18,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo -amb 512 | 1372.3 / 1532.9 | 20.53 / 19.72 | cb 3520.02 MiB, host 59762.00 MiB |
