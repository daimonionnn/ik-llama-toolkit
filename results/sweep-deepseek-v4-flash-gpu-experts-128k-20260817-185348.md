# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,16384,32768. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| tuned n19 ub8192 | (profile defaults) | 1347.5 / 1887.1 / 1804.1 | 19.92 / 19.81 / 19.05 | cb 7040.03 MiB, host 63026.00 MiB |
