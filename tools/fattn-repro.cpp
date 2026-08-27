// =============================================================================
// fattn-repro -- standalone FLASH_ATTN_EXT reproducer (RESULTS 48)
// =============================================================================
// Feeds the staged origin-capture bytes (exact device bytes fattn-0 read, taken
// in-stream at the moment of a NaN event) to a single FLASH_ATTN_EXT node on
// the CUDA backend, rebuilt exactly as build_deepseek4.cpp builds the raw
// branch: v == k, op_params[4] = IQK_DISABLED, sinks attached, prec F32.
//
//   ./fattn-repro <capture-dir> [--scale S]... [--runs N]
//
// The kq_scale is not in the capture (older dumps); pass candidates to sweep.
// Output: per scale, NaN count in the output. Any NaN = reproduction.
// -----------------------------------------------------------------------------
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

struct dumped { int64_t hdr[10]; std::vector<uint8_t> bytes; bool ok=false; };

static dumped load(const std::string & path) {
    dumped d;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return d;
    if (fread(d.hdr, sizeof(d.hdr), 1, f) != 1) { fclose(f); return d; }
    d.bytes.resize((size_t) d.hdr[9]);
    d.ok = fread(d.bytes.data(), 1, d.bytes.size(), f) == d.bytes.size();
    fclose(f);
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <capture-dir> [--scale S]... [--runs N]\n", argv[0]); return 1; }
    std::string dir = argv[1];
    std::vector<float> scales; int runs = 3; bool synth = false, synthmask = false;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--scale") && i+1 < argc) scales.push_back((float) atof(argv[++i]));
        else if (!strcmp(argv[i], "--runs") && i+1 < argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--synthetic")) synth = true;   // random q/k, real mask
        else if (!strcmp(argv[i], "--synthmask")) synthmask = true; // generated mask too
    }
    if (scales.empty()) scales = { 1.f/sqrtf(512.f), 1.f/sqrtf(576.f), 0.0865f, 0.0915f, 0.0625f };

    dumped q = load(dir + "/staged_src0.bin");
    // --synthetic: keep the mask geometry, replace q/k values with a fixed-seed
    // uniform fill. If NaN survives, the trigger is the mask alone and the
    // reproducer carries no private data.

    dumped k = load(dir + "/staged_src1.bin");
    dumped m = load(dir + "/staged_src3.bin");
    dumped s4 = load(dir + "/staged_src4.bin");
    if (!q.ok || !k.ok || !m.ok || !s4.ok) { fprintf(stderr, "capture load failed\n"); return 1; }
    if (synth) {
        uint32_t st = 0x12345678u;
        auto rnd = [&]() { st ^= st<<13; st ^= st>>17; st ^= st<<5; return (st >> 8) * (1.0f/16777216.0f) - 0.5f; };
        float * qf = (float *) q.bytes.data();
        for (size_t i = 0; i < q.bytes.size()/4; ++i) qf[i] = rnd() * 4.0f;
        uint16_t * kh = (uint16_t *) k.bytes.data();
        for (size_t i = 0; i < k.bytes.size()/2; ++i) kh[i] = ggml_fp32_to_fp16(rnd() * 2.0f);
        fprintf(stderr, "synthetic q/k substituted (seed fixed)\n");
    }
    if (synthmask) {
        // fully generated mask with the captured geometry's essence:
        //   rows 0..14 fully -inf (the leading fully-masked block)
        //   rows 15+   a small causal local window of 8 allowed positions
        //   rows >= n_tokens (padding) fully -inf
        const int64_t n_kv = m.hdr[1], n_rows = m.hdr[2];
        const size_t row_h = (size_t) m.hdr[6] / 2;
        uint16_t * mh = (uint16_t *) m.bytes.data();
        const uint16_t NEG_INF = 0xFC00, ZERO = 0x0000;
        const int64_t n_tok = q.hdr[2];
        for (int64_t r = 0; r < n_rows; ++r) {
            for (int64_t c = 0; c < n_kv; ++c) mh[r*row_h + c] = NEG_INF;
            if (r >= 15 && r < n_tok) {
                const int64_t hi = n_kv - (n_tok - r);       // causal-ish anchor
                for (int64_t c = (hi >= 8 ? hi - 8 : 0); c <= hi && c < n_kv; ++c)
                    mh[r*row_h + c] = ZERO;
            }
        }
        fprintf(stderr, "synthetic mask generated (rows 0-14 fully masked, 8-wide window after)\n");
    }
    fprintf(stderr, "loaded: q ne=[%lld,%lld,%lld] k ne=[%lld,%lld] mask ne=[%lld,%lld]\n",
            (long long)q.hdr[1],(long long)q.hdr[2],(long long)q.hdr[3],
            (long long)k.hdr[1],(long long)k.hdr[2],(long long)m.hdr[1],(long long)m.hdr[2]);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!backend) { fprintf(stderr, "cuda init failed\n"); return 1; }

    for (float scale : scales) {
        // fresh context per scale so nothing carries over
        ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        auto mk_base = [&](dumped & d, const char * name) {
            ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) d.hdr[9]);
            ggml_set_name(t, name);
            return t;
        };
        ggml_tensor * qb = mk_base(q, "qbase");
        ggml_tensor * kb = mk_base(k, "kbase");
        ggml_tensor * mb = mk_base(m, "mbase");
        ggml_tensor * sb = mk_base(s4, "sbase");

        auto view_of = [&](ggml_tensor * base, dumped & d, const char * name) {
            ggml_tensor * v = ggml_view_4d(ctx, base, d.hdr[1], d.hdr[2], d.hdr[3], d.hdr[4],
                                           (size_t) d.hdr[6], (size_t) d.hdr[7], (size_t) d.hdr[8], 0);
            v->type = (ggml_type) d.hdr[0];
            v->nb[0] = (size_t) d.hdr[5];
            ggml_set_name(v, name);
            return v;
        };
        ggml_tensor * qv = view_of(qb, q, "q");
        ggml_tensor * kv = view_of(kb, k, "k");
        ggml_tensor * vv = view_of(kb, k, "v");     // v == k, as in the model
        ggml_tensor * mv = view_of(mb, m, "mask");
        ggml_tensor * sv = view_of(sb, s4, "sinks");

        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qv, kv, vv, mv, scale, 0.0f, 0.0f);
        fa->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
        ggml_flash_attn_ext_add_sinks(fa, sv);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, fa);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
        ggml_backend_tensor_set(qb, q.bytes.data(), 0, q.bytes.size());
        ggml_backend_tensor_set(kb, k.bytes.data(), 0, k.bytes.size());
        ggml_backend_tensor_set(mb, m.bytes.data(), 0, m.bytes.size());
        ggml_backend_tensor_set(sb, s4.bytes.data(), 0, s4.bytes.size());

        for (int r = 0; r < runs; ++r) {
            ggml_backend_graph_compute(backend, gf);
            std::vector<float> out(ggml_nelements(fa));
            ggml_backend_tensor_get(fa, out.data(), 0, out.size()*sizeof(float));
            size_t nan = 0; float mx = 0;
            std::vector<int> nan_rows;
            const int64_t d0 = fa->ne[0], nh = fa->ne[1];
            for (size_t ii = 0; ii < out.size(); ++ii) {
                if (std::isnan(out[ii])) {
                    ++nan;
                    int row = (int)(ii / (d0*nh));
                    if (nan_rows.empty() || nan_rows.back() != row) nan_rows.push_back(row);
                } else if (fabsf(out[ii]) > mx) mx = fabsf(out[ii]);
            }
            if (!nan_rows.empty()) {
                printf("  NaN tokens (%zu):", nan_rows.size());
                for (size_t z = 0; z < nan_rows.size() && z < 40; ++z) printf(" %d", nan_rows[z]);
                printf("\n");
            }
            printf("scale %-9.6f run %d: NaN %zu / %zu  max|out| %.4f %s\n",
                   scale, r, nan, out.size(), mx, nan ? " <<<<<<<< REPRODUCED" : "");
            if (nan) { printf("\nREPRODUCED standalone at scale %.6f\n", scale); return 42; }
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    printf("\nnot reproduced with these scales\n");
    return 0;
}
