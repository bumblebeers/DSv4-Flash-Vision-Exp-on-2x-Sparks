#!/usr/bin/env python3
"""Interface-basics validation for DSv4-Flash-0731 on the head node.

Run ON the head node (imports the official encoder from the HF snapshot).
Checks template conformance, API shapes, streaming, tool calls, sampling.
Prints PASS/FAIL per check; exit 1 if any FAIL. Results are evidence for
REPORT.md — genre: validation battery, run after any relaunch.
"""
import json, sys, urllib.request
from pathlib import Path

import os
SNAP = Path(os.environ.get("BASICS_SNAP") or str(Path.home() / (".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731"
                      "/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062")))
BASE = os.environ.get("BASICS_BASE", "http://localhost:8000")
MODEL = os.environ.get("BASICS_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")
sys.path.insert(0, str(SNAP / "encoding"))
import encoding_dsv4  # official encoder

results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def sse(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks

MSGS = [{"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Name one prime number."}]

# --- 1. Template conformance: official render -> /tokenize(prompt) must equal
#        /tokenize(messages) for every thinking x effort combo.
for thinking, effort in [(True, "low"), (True, "high"), (True, "max"), (False, None)]:
    kwargs = {"thinking": thinking}
    enc_kwargs = {"thinking_mode": "thinking" if thinking else "chat"}
    if effort:
        kwargs["reasoning_effort"] = effort
        enc_kwargs["reasoning_effort"] = effort
    official = encoding_dsv4.encode_messages(MSGS, **enc_kwargs)
    t_off = post("/tokenize", {"model": MODEL, "prompt": official})
    t_srv = post("/tokenize", {"model": MODEL, "messages": MSGS,
                               "add_generation_prompt": True,
                               "chat_template_kwargs": kwargs})
    same = (t_srv.get("tokens") == t_off.get("tokens")) if t_off.get("tokens") else \
           (t_srv["count"] == t_off["count"])
    check(f"template thinking={thinking} effort={effort}",
          same, f"server={t_srv['count']} official={t_off['count']} ids_compared={bool(t_off.get('tokens'))}")

# --- 2. Non-streaming shape
r = post("/v1/chat/completions", {"model": MODEL, "messages": MSGS, "max_tokens": 400})
msg = r["choices"][0]["message"]
check("nonstream content present", bool(msg.get("content")), repr(msg.get("content"))[:60])
rkey = "reasoning" if msg.get("reasoning") else ("reasoning_content" if msg.get("reasoning_content") else None)
check("nonstream reasoning key", rkey is not None, f"key={rkey}")
check("nonstream usage+finish", bool(r.get("usage")) and r["choices"][0].get("finish_reason") in ("stop", "length"),
      f"finish={r['choices'][0].get('finish_reason')}")

# --- 3. Streaming shape (no tools): reasoning delta key; no tool_calls key on content deltas
chunks = sse("/v1/chat/completions", {"model": MODEL, "messages": MSGS, "max_tokens": 400, "stream": True})
deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
rkeys = {k for d in deltas for k in d if k in ("reasoning", "reasoning_content")}
bad_tc = sum(1 for d in deltas if d.get("content") and "tool_calls" in d)
check("stream reasoning delta key", bool(rkeys), f"keys={rkeys}")
check("stream no tool_calls[] on content deltas (helge bug)", bad_tc == 0, f"bad deltas={bad_tc}")

# --- 4. Tool call round trip
TOOLS = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}}]
TMSG = [{"role": "user", "content": "What is the weather in Tallinn? Use the tool."}]
r = post("/v1/chat/completions", {"model": MODEL, "messages": TMSG, "tools": TOOLS, "max_tokens": 600})
tcs = r["choices"][0]["message"].get("tool_calls") or []
ok_args = False
if tcs:
    try:
        a = json.loads(tcs[0]["function"]["arguments"])
        ok_args = "city" in a and "arguments" not in a
    except Exception:
        pass
check("tool call nonstream", bool(tcs) and tcs[0]["function"]["name"] == "get_weather" and ok_args,
      f"n={len(tcs)} args_ok={ok_args}")

chunks = sse("/v1/chat/completions", {"model": MODEL, "messages": TMSG, "tools": TOOLS,
                                      "max_tokens": 600, "stream": True})
deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
stream_tc = [d for d in deltas if d.get("tool_calls")]
bad_tc2 = sum(1 for d in deltas if d.get("content") and d.get("tool_calls") == [])
check("tool call streaming deltas", bool(stream_tc), f"tc_deltas={len(stream_tc)}")
check("stream(tools) no empty tool_calls[] on content (helge bug)", bad_tc2 == 0, f"bad={bad_tc2}")

# --- 5. Sampling: generation_config.json vs server; param acceptance
gc = json.loads((SNAP / "generation_config.json").read_text()) if (SNAP / "generation_config.json").exists() else {}
print(f"INFO  generation_config.json: {gc}")
ok = True
for params in ({"temperature": 1.0, "top_p": 0.95}, {"temperature": 0.0}, {"seed": 42}):
    try:
        post("/v1/chat/completions", {"model": MODEL, "messages": MSGS, "max_tokens": 16, **params})
    except Exception as e:
        ok = False
        print(f"      rejected: {params} -> {e}")
check("sampling params accepted (t=1/t=0/seed)", ok)

# --- 6. thinking=false must not inject effort prefix and must not reason
r = post("/v1/chat/completions", {"model": MODEL, "messages": MSGS, "max_tokens": 400,
                                  "chat_template_kwargs": {"thinking": False}})
msg = r["choices"][0]["message"]
rlen = len(msg.get("reasoning") or msg.get("reasoning_content") or "")
check("thinking=false yields no reasoning", rlen == 0 and bool(msg.get("content")), f"rlen={rlen}")

fails = [n for n, ok, _ in results if not ok]
print(f"\n{len(results)-len(fails)}/{len(results)} passed" + (f"; FAILS: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
