#!/usr/bin/env python3
"""mtp-depth.py <port> <label> <depth> [<depth> ...]
Real-text context (repo docs + source) cut to a token depth, then two tasks on
the same prefix: 'code' first (its prefill is the measured one) and 'prose'.
Temperature 0.7, 400 tokens, thinking off, unique salt per depth."""
import json, sys, time, urllib.request, os, random
port, label, depths = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
S = os.path.dirname(os.path.abspath(__file__))
corpus = open(os.path.join(S, "corpus.txt"), encoding="utf-8", errors="replace").read()
def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])
TASKS = [
 ("code",  "Using only the source code above, write a Python script that reads a llama-sweep-bench markdown table from stdin and prints, for each N_KV row, S_PP and S_TG as CSV. Include type hints. Only code, no explanation."),
 ("prose", "In about 300 words of plain prose, summarize what the document above says about how expert placement affects generation speed on this machine."),
]
out = []
for depth in depths:
    salt = f"{label}-{depth}-{random.randrange(1<<30)}"
    head = f"Reference material, session {salt}.\n"
    tail_len = max(ntok(t) for _, t in TASKS) + 64
    # cut the corpus by characters, then correct once by tokens
    chars = int(depth * 3.2)
    doc = corpus[:chars]
    n = ntok(head + doc)
    doc = corpus[:int(chars * (depth - tail_len) / n)]
    for name, task in TASKS:
        msg = head + doc + "\n\n===== END OF MATERIAL =====\n\n" + task
        body = {"model": "x", "messages": [{"role": "user", "content": msg}], "max_tokens": 400,
                "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time(); d = post("/v1/chat/completions", body); wall = time.time() - t
        tm = d.get("timings", {})
        row = dict(label=label, depth=depth, task=name, prompt_n=tm.get("prompt_n"),
                   pp=round(tm.get("prompt_per_second", 0), 1), gen_n=tm.get("predicted_n"),
                   tg=round(tm.get("predicted_per_second", 0), 2), draft_n=tm.get("draft_n"),
                   accepted=tm.get("draft_n_accepted"), wall=round(wall, 1))
        out.append(row); print(json.dumps(row), flush=True)
