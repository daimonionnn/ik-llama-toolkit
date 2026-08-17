# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| ncmoe 14 | IK_NCMOE=14,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | — | — | **did not load** — `out of memory` |
| ncmoe 16 | IK_NCMOE=16,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | — | — | **did not load** — `out of memory` |
| ncmoe 18 (current) | IK_NCMOE=18,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1366.0 / 1528.4 | 20.54 / 19.83 | cb 3520.02 MiB, host 59762.00 MiB |
| ncmoe 20 | IK_NCMOE=20,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1307.9 / 1457.4 | 19.35 / 18.52 | cb 3520.02 MiB, host 66290.00 MiB |
| ncmoe 24 | IK_NCMOE=24,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1218.0 / 1353.6 | 17.22 / 16.44 | cb 3520.02 MiB, host 79346.00 MiB |
