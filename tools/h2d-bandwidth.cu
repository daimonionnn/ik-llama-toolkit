// h2d-bandwidth -- what one host-to-device copy costs on this box, pinned and
// pageable, 1 GiB, best of 5. Used in RESULTS 49.10 to price the per-layer
// mask copies the scheduler was making (50.0 / 18.0 GB/s on the RTX PRO 6000).
//   nvcc -O2 -o h2d-bandwidth tools/h2d-bandwidth.cu && ./h2d-bandwidth
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main() {
    const size_t n = 1ull << 30;
    void * d = nullptr; cudaMalloc(&d, n);
    void * hp = nullptr; cudaMallocHost(&hp, n);
    void * hg = malloc(n);
    memset(hg, 1, n); memset(hp, 1, n);
    cudaMemcpy(d, hp, n, cudaMemcpyHostToDevice); // warm-up
    for (int kind = 0; kind < 2; kind++) {
        void * h = kind ? hg : hp;
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            auto t0 = std::chrono::steady_clock::now();
            cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, cudaStreamPerThread);
            cudaStreamSynchronize(cudaStreamPerThread);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (s < best) best = s;
        }
        printf("%s H2D 1 GiB: best %.3f s = %.1f GB/s\n", kind ? "pageable" : "pinned  ", best, n / best / 1e9);
    }
    return 0;
}
