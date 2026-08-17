# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| t24 tb24 (current) | (profile defaults) | 1334.3 / 1795.3 | 20.05 / 19.20 | cb 7040.03 MiB, host 63026.00 MiB |
| t24 tb8 | IK_THREADS_BATCH=8 | 1331.2 / 1794.7 | 20.05 / 19.09 | cb 7040.03 MiB, host 63026.00 MiB |
| t24 tb4 | IK_THREADS_BATCH=4 | 1315.9 / 1791.9 | 20.18 / 19.16 | cb 7040.03 MiB, host 63026.00 MiB |
| t8 tb24 | IK_THREADS=8 | 1353.4 / 1801.5 | 19.23 / 18.04 | cb 7040.03 MiB, host 63026.00 MiB |
