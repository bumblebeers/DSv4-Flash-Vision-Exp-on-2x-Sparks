#!/usr/bin/env python3
"""Long-context recall and reasoning-effort probes for the packet stack.

`gate-probes.py` checks that the deployment is *up and configured*; `vision-probes.py`
checks the image path; the soak probes check it stays up. This one measures the two
things `RESULTS.md` lists as not measured: **long-context recall** and the cost/accuracy
profile of the thinking levels.

It is a measurement instrument, not a gate: recall at depth is a gradient, and a single
threshold would hide the shape of it. It exits non-zero only if the endpoint did not
answer at all. Read the numbers, and compare them against the reference in
`docs/CONTEXT.md`.

    python3 verify/context-probes.py --out /tmp/context.json
    python3 verify/context-probes.py --out /tmp/context.json --sizes 2000 8000 20000
    python3 verify/context-probes.py --out /tmp/context.json --skip effort decode cache

Preflight note: the two deep rows are prefill-bound. At the packet's measured ~1,540 tok/s
at depth 0 falling to ~860 at 980K, the default 8,000-line ladder costs roughly 3 minutes
of prefill in total, and a 20,000-line row costs ~5 more on its own. Budget accordingly.

Vendor-comparison note: `--effort-style` exists because the two stacks shape the thinking
knobs differently. This packet's stack takes `chat_template_kwargs` (see `gate-probes.py`);
the hosted DeepSeek API takes a top-level `reasoning_effort` and a `thinking: {type:
"disabled"}` object. The default matches this packet. Pointed at the hosted API, use
`--effort-style top_level`.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://localhost:8000")
p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
p.add_argument("--out", required=True)
p.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 8000],
               help="haystack sizes in filler lines, ascending (~25 tokens/line)")
p.add_argument("--seed", type=int, default=11)
p.add_argument("--nonce", type=int, default=0,
               help="added to --seed; use a fresh nonce when prefix caching would otherwise "
                    "serve a haystack this endpoint has already seen")
p.add_argument("--effort-style", choices=["chat_template", "top_level", "none"],
               default="chat_template")
p.add_argument("--effort-reps", type=int, default=8,
               help="runs per thinking level (keep >=8; the per-level spread is wide and "
                    "small samples have produced confidently wrong readings)")
p.add_argument("--skip", nargs="*", default=[], choices=["models", "recall", "effort", "decode", "cache"])
p.add_argument("--timeout", type=int, default=3600)
p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY",
               help="name of an env var holding a bearer token, for pointing this at a "
                    "hosted OpenAI-compatible endpoint. Unset means no auth header is sent, "
                    "which is what this packet's own stack needs. Never pass a key on argv.")
args = p.parse_args()
KEY = os.environ.get(args.api_key_env) or None

BASE, MODEL = args.base.rstrip("/"), args.model
SKIP = set(args.skip)
res = {"t": time.time(), "base": BASE, "model": MODEL, "effort_style": args.effort_style,
       "seed": args.seed, "nonce": args.nonce, "document_seed": args.seed + args.nonce,
       "sizes": args.sizes, "effort_reps": args.effort_reps, "errors": []}


def _headers():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def get(path, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(BASE + path, headers=_headers()),
                                  timeout=timeout)


def post(path, body, timeout=None, stream=False):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), _headers())
    return urllib.request.urlopen(req, timeout=timeout or args.timeout)


def chat(messages, **params):
    """Non-streaming call -> (body, elapsed). Errors are returned, never raised."""
    body = {"model": MODEL, "messages": messages}
    body.update({k: v for k, v in params.items() if v is not None})
    t0 = time.time()
    try:
        with post("/v1/chat/completions", body) as r:
            return json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return {"error": e.read()[:400].decode("utf8", "replace"), "status": e.code}, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}, time.time() - t0


def usage_of(body):
    return (body or {}).get("usage") or {}


def reasoning_tokens(body):
    """(tokens, reported). reported=False as distinct from a reported zero -- a serving
    stack that has no thinking plumbing at all reports neither field."""
    det = usage_of(body).get("completion_tokens_details")
    if det is None:
        return 0, False
    return det.get("reasoning_tokens", 0), True


def cache_read_tokens(body):
    u = usage_of(body)
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0


def text_of(body):
    try:
        return body["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


# --- fixed inputs (changing these invalidates comparison to docs/CONTEXT.md) --------

WORDS = ("regional logistics audit variance warehouse quarterly ledger depot manifest freight "
         "consolidated routing surcharge invoice pallet throughput dockyard customs tariff "
         "shipment waybill carrier transit reconciliation discrepancy overage shortage").split()

NEEDLES = [(0.05, "Anchorage", "KILO-7319"), (0.25, "Brasilia", "TANGO-2044"),
           (0.50, "Zagreb", "MIKE-8850"), (0.75, "Osaka", "SIERRA-1177"),
           (0.95, "Lyon", "PAPA-6031")]

NEEDLE_QUESTION = ("The document below contains lines beginning 'IMPORTANT RECORD:'. "
                   "List every depot name with its access code, in the order they appear. "
                   "Output only the list.")

EFFORT_LEVELS = ["none", "minimal", "low", "medium", "high", "max"]
HARD_PROBLEM = ("A 4x4 grid has a bug at each of 3 distinct cells, chosen uniformly at random. "
                "What is the probability that no two bugs share a row or a column? "
                "Reason carefully, then end with 'ANSWER: <fraction>'.")
HARD_ANSWER = "6/35"

DECODE_PROMPTS = {
    "diverse": "Write a detailed essay on the history of cartography. Aim for about 400 words.",
    "predictable": "Write the integers from 1 to 300, one per line, with no other text.",
}


def build_document(n_lines, seed):
    rnd = random.Random(seed)
    lines = [f"Line {i:05d}: " + " ".join(rnd.choice(WORDS) for _ in range(13)) +
             f" ref {rnd.randrange(100000, 999999)}." for i in range(n_lines)]
    truth = {}
    for frac, depot, code in NEEDLES:
        lines.insert(int(len(lines) * frac),
                     f"IMPORTANT RECORD: the access code for depot {depot} is {code}.")
        truth[depot] = code
    return "\n".join(lines), truth


def effort_params(level):
    """Thinking knobs in the shape this packet's stack expects."""
    if args.effort_style == "none":
        return {}
    if args.effort_style == "chat_template":
        if level == "none":
            return {"chat_template_kwargs": {"thinking": False}}
        return {"chat_template_kwargs": {"thinking": True, "reasoning_effort": level}}
    return {"reasoning_effort": level}


# --- sections ----------------------------------------------------------------------

def s_models():
    out = {}
    try:
        with get("/v1/models") as r:
            d = json.loads(r.read())
        for m in d.get("data", []):
            out[m.get("id")] = {"max_model_len": m.get("max_model_len"),
                                "root": m.get("root"), "parent": m.get("parent")}
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    res["models"] = out
    for k, v in out.items():
        print(f"model {k}: max_model_len={v.get('max_model_len') if isinstance(v, dict) else v}")


def s_recall():
    rows = []
    for n_lines in args.sizes:
        doc, truth = build_document(n_lines, res["document_seed"])
        body, dt = chat([{"role": "user", "content": NEEDLE_QUESTION + "\n\n" + doc}],
                        temperature=0, max_tokens=1024)
        if "error" in body:
            rows.append({"lines": n_lines, "chars": len(doc), "status": body.get("status"),
                         "error": body["error"]})
            print(f"recall {n_lines:>6} lines: REJECTED ({str(body['error'])[:100]})")
            break  # sizes ascend; a rejection means the ceiling is below this
        text = text_of(body)
        u = usage_of(body)
        ptok = u.get("prompt_tokens") or 0
        hit = cache_read_tokens(body)
        found = {d: (c.lower() in text.lower()) for d, c in truth.items()}
        rows.append({
            "lines": n_lines, "chars": len(doc), "prompt_tokens": ptok,
            "cache_read_tokens": hit,
            # a warm prefix makes the prefill time meaningless; use --nonce to avoid it
            "warm": bool(ptok) and hit > 0.05 * ptok,
            "seconds": round(dt, 1),
            "chars_per_token": round(len(doc) / ptok, 2) if ptok else None,
            "needles_found": sum(found.values()), "needles_total": len(found), "found": found,
            "prefill_tok_per_s": round(ptok / dt) if dt > 0 else None,
            "answer": text[:300],
        })
        print(f"recall {n_lines:>6} lines: {ptok:>8} tok  {dt:6.1f}s  "
              f"{sum(found.values())}/{len(found)} needles"
              f"{'  (warm prefix)' if rows[-1]['warm'] else ''}")
    res["recall"] = rows


def s_effort():
    rows = []
    for level in EFFORT_LEVELS:
        for rep in range(args.effort_reps):
            body, dt = chat([{"role": "user", "content": HARD_PROBLEM}],
                            temperature=0, max_tokens=4096, **effort_params(level))
            if "error" in body:
                rows.append({"level": level, "rep": rep, "status": body.get("status"),
                             "error": body["error"]})
                break
            rtok, reported = reasoning_tokens(body)
            rows.append({"level": level, "rep": rep, "reasoning_tokens": rtok,
                         "reasoning_reported": reported,
                         "completion_tokens": usage_of(body).get("completion_tokens"),
                         "correct": HARD_ANSWER in text_of(body), "seconds": round(dt, 1)})
        ok = [r for r in rows if r["level"] == level and "error" not in r]
        if ok:
            toks = sorted(r["reasoning_tokens"] for r in ok)
            print(f"effort {level:>8}: reasoning tokens {toks} "
                  f"correct {sum(1 for r in ok if r['correct'])}/{len(ok)}")
        else:
            print(f"effort {level:>8}: REJECTED ({str(rows[-1].get('error'))[:90]})")
    res["effort"] = rows


def s_decode():
    out = {}
    for name, prompt in DECODE_PROMPTS.items():
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 1024, "stream": True,
                "stream_options": {"include_usage": True}}
        t0 = time.time()
        ttft = None
        n_delta = 0
        usage = None
        try:
            with post("/v1/chat/completions", body) as r:
                for raw in r:
                    line = raw.decode("utf8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        o = json.loads(payload)
                    except Exception:
                        continue
                    if o.get("usage"):
                        usage = o["usage"]
                    for ch in o.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content") or d.get("reasoning_content"):
                            if ttft is None:
                                ttft = time.time() - t0
                            if d.get("content"):
                                n_delta += 1
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"decode {name:>12}: FAILED {e}")
            continue
        elapsed = time.time() - t0
        ctok = (usage or {}).get("completion_tokens") or n_delta
        gen = (elapsed - ttft) if ttft else elapsed
        out[name] = {"prompt": prompt, "ttft_ms": round(ttft * 1000) if ttft else None,
                     "completion_tokens": ctok, "seconds": round(elapsed, 2),
                     "usage_streamed": usage is not None,
                     "decode_tok_per_s": round(ctok / gen, 1) if gen > 0 else None}
        print(f"decode {name:>12}: {out[name]['decode_tok_per_s']} tok/s over {ctok} tokens, "
              f"TTFT {out[name]['ttft_ms']} ms")
    res["decode"] = out


def s_cache():
    doc, _ = build_document(800, seed=99)
    msgs = [{"role": "user", "content": "Reply with the single word OK.\n\n" + doc}]
    chat(msgs, temperature=0, max_tokens=16)
    body, dt = chat(msgs, temperature=0, max_tokens=16)
    u = usage_of(body)
    res["cache"] = {"prompt_tokens": u.get("prompt_tokens"),
                    "cached_tokens": cache_read_tokens(body),
                    "second_call_seconds": round(dt, 2)}
    print(f"cache: {res['cache']['cached_tokens']} of {res['cache']['prompt_tokens']} tokens "
          f"cached on the repeat call")


for name, fn in (("models", s_models), ("recall", s_recall), ("effort", s_effort),
                 ("decode", s_decode), ("cache", s_cache)):
    if name in SKIP:
        continue
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - one broken section must not lose the others
        res["errors"].append(f"{name}: {type(e).__name__}: {e}")
        print(f"! section {name} failed: {e}", file=sys.stderr)

Path(args.out).write_text(json.dumps(res, indent=2))
print(f"\nwrote {args.out}")
if res["errors"]:
    print("section errors: " + "; ".join(res["errors"]), file=sys.stderr)
sys.exit(0)
