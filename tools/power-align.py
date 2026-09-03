#!/usr/bin/env python3
"""Align an nvidia-smi power log with a server log's per-request timings.

Sample the card alongside a benchmark:
    nvidia-smi --query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu \
        --format=csv,noheader -l 1 > power.csv
then, per server log, print what the GPU drew during each request's prefill
and generation window (reconstructed from the print_timing block and the
request line's epoch timestamp):
    tools/power-align.py power.csv logs/server-*.log
Used in RESULTS 49.10 to show that a 400 W cap clipped prefill (peaks 450-572 W)
but never touched generation (160-200 W).
"""
import csv, datetime, re, statistics, sys

def load(path):
    out = []
    for r in csv.reader(open(path)):
        if len(r) < 7:
            continue
        try:
            ts = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            out.append((ts, float(r[1].split()[0]), float(r[3].split()[0])))
        except ValueError:
            pass
    return out

def window(samples, a, b):
    p = [s[1] for s in samples if a <= s[0] <= b]
    c = [s[2] for s in samples if a <= s[0] <= b]
    if not p:
        return "no samples"
    over = 100 * sum(1 for x in p if x > 400) / len(p)
    return (f"mean {statistics.mean(p):5.0f} W  max {max(p):3.0f} W  >400 W {over:3.0f} %  "
            f"clk {statistics.mean(c):4.0f} MHz  ({len(p)} s)")

PAT = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?\n\s+eval time =\s+([\d.]+) ms /"
                 r"\s+(\d+) tokens.*?\n\s+total time =.*?\n.*?timestamp=(\d+)", re.S)

def main():
    samples = load(sys.argv[1])
    min_tokens = 5000
    for log in sys.argv[2:]:
        txt = open(log, "rb").read().decode("utf-8", "replace")
        print(f"## {log}")
        for m in PAT.finditer(txt):
            tp, ntok, tg, end = float(m[1]) / 1000, int(m[2]), float(m[3]) / 1000, int(m[5])
            if ntok < min_tokens:
                continue
            g0 = end - tg
            p0 = g0 - tp
            print(f"  {ntok:6d} tok  prefill {tp:6.1f} s  {window(samples, p0 + 1, g0 - 1)}")
            print(f"  {'':6s}      gen     {tg:6.1f} s  {window(samples, g0 + 1, end - 1)}")

if __name__ == "__main__":
    main()
