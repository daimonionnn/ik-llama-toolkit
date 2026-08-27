// =============================================================================
// replay-nan-dump -- offline reproducer for the NaN abort (RESULTS 47.1)
// =============================================================================
// 46.4 proved fattn-0 computes NaN deterministically from clean inputs, and 47
// dumps the poisoned state at abort time. This replays the dumped token history
// through llama_decode with controllable ubatch boundaries and reports the
// first chunk whose logits contain NaN.
//
//   ./replay-nan-dump --dump logs/nan-dump-<ts> [--align-end N] -- <server args>
//
// --align-end N places chunk boundaries so the FINAL chunk has exactly N tokens
// (the aborting graph's n_tokens, printed by IK_NAN_CHECK). Without it, chunks
// run from position 0 in n_ubatch steps.
// The KV dumps (kv_l0.bin) are not loaded -- the cache is rebuilt by replaying
// the same tokens, which is exactly what the server's re-prefill did when it
// re-poisoned at the same spot (+1 token, 47). They remain as ground truth to
// diff against if a replay ever fails to reproduce.
// -----------------------------------------------------------------------------
#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

extern "C" int llama_ik_load_kv_layer(struct llama_context *, int, const char *);

static std::vector<llama_token> read_tokens(const std::string & path) {
    std::vector<llama_token> toks;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    int32_t t;
    while (fread(&t, sizeof(t), 1, f) == 1) toks.push_back(t);
    fclose(f);
    return toks;
}

int main(int argc, char ** argv) {
    std::string dump_dir; int align_end = 0; bool inject_kv = false;
    std::vector<char *> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump")      && i+1 < argc) { dump_dir  = argv[++i]; continue; }
        if (!strcmp(argv[i], "--align-end") && i+1 < argc) { align_end = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--inject-kv")) { inject_kv = true; continue; }
        if (!strcmp(argv[i], "--")) continue;
        rest.push_back(argv[i]);
    }
    if (dump_dir.empty()) { fprintf(stderr, "usage: %s --dump <dir> [--align-end N] -- <server args>\n", argv[0]); return 1; }

    auto toks = read_tokens(dump_dir + "/tokens.bin");
    fprintf(stderr, "replay: %zu tokens from %s\n", toks.size(), dump_dir.c_str());

    gpt_params params;
    if (!gpt_params_parse((int) rest.size(), rest.data(), params)) return 1;

    llama_init_result ir = llama_init_from_gpt_params(params);
    if (!ir.model || !ir.context) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context * ctx = ir.context;
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_ub    = params.n_ubatch;

    // chunk plan
    std::vector<std::pair<int,int>> plan;  // (p0, n)
    const int T = (int) toks.size();
    if (align_end > 0 && align_end < T) {
        const int body = T - align_end;
        for (int p = 0; p < body; ) { int n = std::min(n_ub, body - p); plan.push_back({p, n}); p += n; }
        plan.push_back({body, align_end});
    } else {
        for (int p = 0; p < T; ) { int n = std::min(n_ub, T - p); plan.push_back({p, n}); p += n; }
    }
    fprintf(stderr, "replay: %zu chunks, final chunk %d tokens\n", plan.size(), plan.back().second);

    for (size_t c = 0; c < plan.size(); ++c) {
        auto [p0, n] = plan[c];
        if (inject_kv && c == plan.size() - 1) {
            // body replay only populated the cells; now make layers 0/1
            // bit-identical to the aborting server before the fatal chunk
            const int mb0 = llama_ik_load_kv_layer(ctx, 0, (dump_dir + "/kv_l0.bin").c_str());
            const int mb1 = llama_ik_load_kv_layer(ctx, 1, (dump_dir + "/kv_l1.bin").c_str());
            fprintf(stderr, "injected server KV: l0=%d MB, l1=%d MB\n", mb0, mb1);
            if (mb0 < 0 || mb1 < 0) return 3;
        }
        llama_batch batch = llama_batch_get_one(toks.data() + p0, n, p0, 0);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at chunk %zu (p0=%d)\n", c, p0); return 2; }
        const float * logits = llama_get_logits(ctx);
        int n_nan = 0; float mx = 0.f;
        for (int i = 0; i < n_vocab; ++i) {
            if (std::isnan(logits[i])) ++n_nan;
            else if (std::fabs(logits[i]) > mx) mx = std::fabs(logits[i]);
        }
        fprintf(stderr, "chunk %3zu  p0=%7d  n=%5d  max|logit|=%8.3f  nan=%d/%d %s\n",
                c, p0, n, mx, n_nan, n_vocab, n_nan ? " <<<<<<<< NaN REPRODUCED" : "");
        if (n_nan) { fprintf(stderr, "\nREPRODUCED at chunk %zu, tokens [%d, %d)\n", c, p0, p0+n); return 42; }
    }
    fprintf(stderr, "\nNOT reproduced with this chunk plan\n");
    return 0;
}
