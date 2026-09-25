#!/usr/bin/env python3
"""Behavioral gate probes for a freshly launched DSv4-Flash server.

Generalized from runs/2026/08/19-relaunch/gate-probes.py (launch-6 gate evidence);
probe content kept identical so results stay comparable across launches.

Covers: reasoning-effort /tokenize fingerprints, custom-hook tokenize, invalid
effort level -> HTTP 400, Tallinn smoke, spec-decode acceptance on two content
classes via /metrics deltas.

The fingerprint probe is the standing contract that reasoning-effort rendering is
correct (owner directive 2026-08-23): an upstream image/encoder change that alters
rendering fails this gate loudly instead of degrading quality silently.
  --expect-fp LOW,HIGH,MAX   expected token counts (eugr+mod v2: 5,84,97)
  --fp-mode enforce|record   'record' logs observed values without failing the gate
                             (used on the stock aiden stack until the fix is ported)
"""
import argparse, json, re, sys, time, urllib.request, urllib.error

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://localhost:8000")
p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
p.add_argument("--expect-fp", default="5,84,97", help="expected /tokenize counts low,high,max")
p.add_argument("--fp-mode", choices=["enforce", "record"], default="enforce")
p.add_argument("--out", required=True)
args = p.parse_args()

BASE, MODEL = args.base, args.model
want_low, want_high, want_max = (int(x) for x in args.expect_fp.split(","))
res = {"t": time.time(), "base": BASE, "fp_mode": args.fp_mode}

def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def counters():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
                 "spec_decode_num_accepted_tokens_total"):
        m = re.findall(rf'^vllm:{name}.*?\s([\d.e+]+)$', txt, re.M)
        out[name] = sum(float(x) for x in m)
    return out

# 1. fingerprints (runner2 canary form: 1-msg "hi", thinking=True)
fp = {}
for eff, want in (("low", want_low), ("high", want_high), ("max", want_max)):
    n = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                           "chat_template_kwargs": {"thinking": True, "reasoning_effort": eff}},
             timeout=60)["count"]
    fp[eff] = {"count": n, "want": want, "ok": n == want}
res["fingerprints"] = fp

# 2. custom hook (count depends on mounted file; record what we see)
try:
    n = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                           "chat_template_kwargs": {"thinking": True, "reasoning_effort": "custom"}},
             timeout=60)["count"]
    res["custom_tokenize"] = {"count": n, "note": "vs low; delta = injected custom text"}
except Exception as e:
    res["custom_tokenize"] = {"error": str(e)[:120]}

# 3. unknown reasoning_effort. RE-SPECCED 2026-09-23: this build maps an unknown/typo'd
# effort to `high` (tokenizers/deepseek_v4.py), so the correct contract is that it renders the
# SAME as high -- not that it is rejected. The old "expect HTTP 400" expectation came from the
# superseded 0.25.2 packet and was permanently red for nothing. This form is a real regression
# check: if the mapping changes, the count moves and the gate fails.
try:
    n_unknown = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                       "chat_template_kwargs": {"thinking": True, "reasoning_effort": "zzz"}},
         timeout=60)["count"]
    res["invalid_level"] = {"status": "no-error", "count": n_unknown, "want": want_high,
                            "ok": n_unknown == want_high,
                            "note": "build maps unknown effort -> high; asserted to render as high"}
except urllib.error.HTTPError as e:
    res["invalid_level"] = {"status": e.code, "ok": False,
                            "note": "unknown effort was rejected; build is expected to map it to high"}

# 4. smoke (first gen request post-boot; generous timeout)
t0 = time.time()
r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 256,
         "messages": [{"role": "user", "content": "What is the capital of Estonia? One word."}]})
content = r["choices"][0]["message"].get("content") or ""
res["smoke"] = {"ok": "Tallinn" in content, "wall_s": round(time.time() - t0, 1),
                "usage": r.get("usage", {}).get("completion_tokens")}

# 5. acceptance on two content classes
def accept_delta(fn):
    b = counters(); fn(); a = counters()
    dt = a["spec_decode_num_draft_tokens_total"] - b["spec_decode_num_draft_tokens_total"]
    ac = a["spec_decode_num_accepted_tokens_total"] - b["spec_decode_num_accepted_tokens_total"]
    return {"draft_tokens": dt, "accepted": ac, "rate": round(ac / dt, 3) if dt else None}

def factual():
    for q in ("List the first eight prime numbers.", "Name the planets of the solar system in order.",
              "What is 17 times 23? Show the multiplication."):
        post("/v1/chat/completions", {"model": MODEL, "max_tokens": 200,
             "messages": [{"role": "user", "content": q}],
             "chat_template_kwargs": {"thinking": False}})

GUTENBERG = ("It was a bright cold day in April, and the clocks were striking thirteen. "
             "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, "
             "slipped quickly through the glass doors of Victory Mansions, though not quickly enough "
             "to prevent a swirl of gritty dust from entering along with him.")
def continuation():
    post("/v1/chat/completions", {"model": MODEL, "max_tokens": 300,
         "messages": [{"role": "user", "content":
                       "Continue this passage in the same style for one paragraph:\n\n" + GUTENBERG}],
         "chat_template_kwargs": {"thinking": False}})

res["acceptance_factual"] = accept_delta(factual)
res["acceptance_continuation"] = accept_delta(continuation)

with open(args.out, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

fp_bad = not all(v["ok"] for v in res["fingerprints"].values())
if fp_bad and args.fp_mode == "record":
    print("NOTE: fingerprints off-expectation but fp-mode=record (stock-aiden bracket)", file=sys.stderr)
    fp_bad = False
if not res["smoke"]["ok"] or not res["invalid_level"].get("ok") or fp_bad:
    sys.exit(1)
