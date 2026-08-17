# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| ncmoe 17 | IK_NCMOE=17,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | — | — | **did not load** — `out of memory` |
