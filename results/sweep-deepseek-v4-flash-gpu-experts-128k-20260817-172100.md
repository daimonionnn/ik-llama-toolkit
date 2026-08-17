# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768,128000. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| ub8192 n19 | IK_UBATCH=8192,IK_BATCH=8192,IK_NCMOE=19,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | 1329.4 / 1793.6 / 1327.7 | 19.98 / 19.14 / 17.20 | cb 7040.03 MiB, host 63026.00 MiB |
| ub12288 n20 | IK_UBATCH=12288,IK_BATCH=12288,IK_NCMOE=20,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | — | — | **died mid-run** — `RemoteDisconnected('Remote end closed connection without response')` |
| ub16384 n21 | IK_UBATCH=16384,IK_BATCH=16384,IK_NCMOE=21,IK_EXTRA_ARGS=-mla 3 -fidx --reasoning-format deepseek -nkvo | — | — | **died mid-run** — `RemoteDisconnected('Remote end closed connection without response')` |
