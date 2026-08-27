// =============================================================================
// fattn-nan-repro -- fully self-contained reproducer, no external files needed
// =============================================================================
// FLASH_ATTN_EXT (the new-MMA path taken for DeepSeek-shaped heads, K=V=512)
// emits NaN for query rows whose mask is entirely -inf when they form the
// LEADING block of the batch. Such rows are legitimate in DeepSeek V4 DSA:
// the raw branch can select nothing for a token, and with attention sinks
// attached the correct output is 0 -- which is exactly what mid-batch
// fully-masked rows and masked padding rows produce. The leading block gets
// NaN instead.
//
// Everything below is generated: random q/k, a mask with rows 0..14 fully
// masked and an 8-wide causal window after. Shapes mirror a real failing
// graph (n_tokens 1060, n_kv 1280, 64 heads, head dim 512).
//
//   g++ -O2 fattn-nan-repro.cpp -I ggml/include -lggml -o repro && ./repro
//   expected on sm_120: "NaN 458752 / 34734080 ... tokens 0..13"
// -----------------------------------------------------------------------------
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

int main() {
    const int64_t hd = 512, n_tok = 1060, n_head = 64, n_kv = 1280;
    const int64_t rows_padded = 1072;              // n_tok padded as the server pads it
    const float   scale = 1.0f/sqrtf((float) hd);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!backend) { fprintf(stderr, "cuda init failed\n"); return 1; }

    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, n_tok, n_head, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, n_kv, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, rows_padded, 1, 1);
    ggml_tensor * sink = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_head);

    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, k /* v==k, MLA-style */, mask, scale, 0.0f, 0.0f);
    fa->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
    ggml_flash_attn_ext_add_sinks(fa, sink);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic fill
    uint32_t st = 0x12345678u;
    auto rnd = [&]() { st ^= st<<13; st ^= st>>17; st ^= st<<5; return (st >> 8)*(1.0f/16777216.0f) - 0.5f; };

    { std::vector<float> h(ggml_nelements(q));  for (auto & x : h) x = rnd()*4.0f;
      ggml_backend_tensor_set(q, h.data(), 0, h.size()*4); }
    { std::vector<ggml_fp16_t> h(ggml_nelements(k)); for (auto & x : h) x = ggml_fp32_to_fp16(rnd()*2.0f);
      ggml_backend_tensor_set(k, h.data(), 0, h.size()*2); }
    { std::vector<float> h(n_head); for (auto & x : h) x = rnd();
      ggml_backend_tensor_set(sink, h.data(), 0, h.size()*4); }
    {
        const ggml_fp16_t NEG_INF = ggml_fp32_to_fp16(-INFINITY), ZERO = ggml_fp32_to_fp16(0.0f);
        std::vector<ggml_fp16_t> h((size_t) n_kv*rows_padded, NEG_INF);
        for (int64_t r = 15; r < n_tok; ++r) {              // rows 0..14 stay fully masked
            const int64_t hi = n_kv - (n_tok - r);
            for (int64_t c = (hi >= 8 ? hi - 8 : 0); c <= hi && c < n_kv; ++c) h[r*n_kv + c] = ZERO;
        }
        ggml_backend_tensor_set(mask, h.data(), 0, h.size()*2);
    }

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(ggml_nelements(fa));
    ggml_backend_tensor_get(fa, out.data(), 0, out.size()*4);
    size_t nan = 0; int first = -1, last = -1;
    for (size_t i = 0; i < out.size(); ++i) {
        if (std::isnan(out[i])) {
            ++nan;
            const int tok = (int)(i / (hd*n_head));
            if (first < 0) first = tok;
            last = tok;
        }
    }
    printf("NaN %zu / %zu%s", nan, out.size(), nan ? ", tokens " : "\n");
    if (nan) printf("%d..%d  <-- fully-masked leading rows; expected output for them is 0, not NaN\n", first, last);
    return nan ? 42 : 0;
}
