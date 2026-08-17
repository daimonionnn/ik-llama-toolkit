# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,16384,32768,128000. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| DDR5-7400 2x64GB | (profile defaults) | 1382.1 / 1909.7 / 1845.8 / 1362.2 | 21.23 / 21.07 / 20.22 / 18.00 | cb 7040.03 MiB, host 63026.00 MiB |
