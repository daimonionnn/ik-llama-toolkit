import json, sys, time, urllib.request
port = sys.argv[1]; temp = float(sys.argv[2])
prompts = {
 "code":  "Write a Python function that reads a CSV file and returns per-column mean, min and max for numeric columns. Include type hints and a docstring. Only code, no explanation.",
 "prose": "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
 "extract": "Extract every person, date and amount from this text as JSON: 'On 3 March 2024 Anna Novak paid 1,250 EUR to Peter Kral; on 17 April Peter refunded 300 EUR, and on 2 May 2024 Maria Horvat invoiced Anna 780 EUR.'",
}
for name, p in prompts.items():
    body = {"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t
    tm = d.get("timings", {})
    txt = d["choices"][0]["message"].get("content") or ""
    print(f"{name:8s} tokens={d['usage']['completion_tokens']:4d} tg={tm.get('predicted_per_second', 0):6.2f} t/s  "
          f"draft_n={tm.get('draft_n','-')} accepted={tm.get('draft_n_accepted','-')}  wall={dt:5.1f}s | {txt[:70]!r}")
