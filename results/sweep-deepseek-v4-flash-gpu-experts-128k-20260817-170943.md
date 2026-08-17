# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| ub2048 b8192 n18 | IK_UBATCH=2048,IK_BATCH=8192,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1029.3 / 1122.6 | 20.75 / 19.73 | cb 2496.01 MiB, host 59762.00 MiB |
| ub4096 b8192 n18 (current) | IK_UBATCH=4096,IK_BATCH=8192,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1370.8 / 1520.7 | 20.79 / 19.62 | cb 3520.02 MiB, host 59762.00 MiB |
| ub4096 b4096 n18 | IK_UBATCH=4096,IK_BATCH=4096,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1371.9 / 1435.9 | 20.63 / 19.77 | cb 3520.02 MiB, host 59762.00 MiB |
| ub6144 b8192 n18 | IK_UBATCH=6144,IK_BATCH=8192,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | — | — | **died mid-run** — `RemoteDisconnected('Remote end closed connection without response')` |
| ub8192 b8192 n19 | IK_UBATCH=8192,IK_BATCH=8192,IK_NCMOE=19,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1336.7 / 1792.9 | 19.88 / 19.15 | cb 7040.03 MiB, host 63026.00 MiB |
