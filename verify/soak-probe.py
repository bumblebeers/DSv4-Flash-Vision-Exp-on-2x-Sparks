#!/usr/bin/env python3
"""Post-launch soak gate (LAUNCH-CHECKLIST), generalized from
runs/2026/08/19-relaunch/soak-probe.py (probe mix unchanged for comparability): ~2 min mixed-length concurrent requests at
c=8, varied prompts/budgets/thinking — the DSpark uniform-context EngineDeadError
watch (vLLM >=0.21 class). Output: soak-summary.json. Genre: gate evidence."""
import argparse, json, random, threading, time, urllib.request
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("--base", default="http://localhost:8000")
_p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
_p.add_argument("--seconds", type=int, default=130)
_p.add_argument("--out", required=True)
_a = _p.parse_args()
BASE, MODEL, OUT = _a.base, _a.model, Path(_a.out)
DEADLINE = time.time() + _a.seconds
LOCK = threading.Lock()
stats = {"ok": 0, "err": 0, "errors": [], "finishes": {}, "walls": []}

FILLER = ("The quarterly report shows revenue of %d thousand across region %d, "
          "with logistics costs rising %d percent and inventory turns at %d. ")

def post(body, timeout=180):
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def make_req(rng):
    kind = rng.randrange(4)
    if kind == 0:      # short factual, no think
        return {"messages": [{"role": "user", "content": rng.choice([
                    "Name three rivers in Europe.", "What is 91*7?",
                    "Give one synonym for 'rapid'."])}],
                "max_tokens": rng.choice([32, 96]),
                "chat_template_kwargs": {"thinking": False}}
    if kind == 1:      # long context, varied length
        n = rng.choice([20, 60, 140])
        ctx = " ".join(FILLER % (rng.randrange(900), i % 9, rng.randrange(40), rng.randrange(12))
                       for i in range(n))
        return {"messages": [{"role": "user", "content":
                    ctx + "\n\nSummarize the overall trend in two sentences."}],
                "max_tokens": rng.choice([64, 128]),
                "chat_template_kwargs": {"thinking": False}}
    if kind == 2:      # thinking on, small math
        return {"messages": [{"role": "user", "content": rng.choice([
                    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
                    "A rectangle has area 84 and one side 7. What is its perimeter?"])}],
                "max_tokens": rng.choice([512, 1024])}
    return {"messages": [{"role": "user", "content":                     # agentic-ish
                "You have tools: search(q), read(id). Plan (in words, no tool syntax) "
                "how you would find the population of the largest city in Estonia."}],
            "max_tokens": rng.choice([256, 384]),
            "chat_template_kwargs": {"thinking": rng.random() < 0.5}}

def worker(wid):
    rng = random.Random(1000 + wid)
    while time.time() < DEADLINE:
        t0 = time.time()
        try:
            r = post({"model": MODEL, **make_req(rng)})
            fin = r["choices"][0].get("finish_reason")
            with LOCK:
                stats["ok"] += 1
                stats["finishes"][fin] = stats["finishes"].get(fin, 0) + 1
                stats["walls"].append(round(time.time() - t0, 1))
        except Exception as e:
            with LOCK:
                stats["err"] += 1
                if len(stats["errors"]) < 10:
                    stats["errors"].append(str(e)[:120])

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
t0 = time.time()
for t in threads: t.start()
for t in threads: t.join()
stats["duration_s"] = round(time.time() - t0, 1)
stats["walls"] = {"n": len(stats["walls"]), "max": max(stats["walls"], default=None)}
OUT.write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
raise SystemExit(1 if stats["err"] else 0)
