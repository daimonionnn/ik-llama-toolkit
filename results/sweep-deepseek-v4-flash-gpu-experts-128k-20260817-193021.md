# sweep: `deepseek-v4-flash-gpu-experts-128k`

Depths 4096,16384,32768,128000. Two repeats each, `max_tokens` 160, temperature 0, unique salt
per request. One fresh server per arm.

| arm | overrides | prefill tok/s | generation t/s | notes |
|---|---|---|---|---|
| new build (DSA f32) | (profile defaults) | 1335.4 / 1867.2 / 1801.7 / 1330.9 | 20.07 / 19.85 / 19.18 / 17.17 | cb 7040.03 MiB, host 63026.00 MiB |
