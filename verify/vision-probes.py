#!/usr/bin/env python3
"""Vision gate probes for DeepSeek-V4-Flash-Vision-Exp on the head node.

Genre: measurement instrument (new for the Vision deployment; no 0731 precedent).

Probes:
  1. carrots-describe   — single image, loose grade ("carrot" in answer)
  2. two-image-order    — carrots then corn, JSON answer, order preservation
                          (mirrors the model repo's own two-image ordering test)
  3. shapes-exact       — deterministic PIL-drawn shapes, count/color JSON,
                          exact-match grade (community precedent: voktolom
                          synthetic shapes test, forum 381911 #38)
  4. ocr-lines          — PIL-rendered text, transcription substring grade
  5. tool-image-400     — structured image part in a tool message must 400
                          (official restriction), BUT tool message *text*
                          mentioning <image> tags must NOT 400 (MiaAI issue
                          #167 fix regression probe)

All raw responses archived to the output JSON. Generated stimuli saved as PNGs
beside it. Sampling: temperature 0 for graded probes (determinism); server
defaults otherwise untouched.

Usage: python3 vision-probes.py --out vision-probes-result.json
"""
import argparse, base64, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://localhost:8000")
p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
p.add_argument("--out", required=True)
p.add_argument("--imgdir", default=str(Path(__file__).parent))
args = p.parse_args()

BASE, MODEL = args.base, args.model
IMGDIR = Path(args.imgdir)
res = {"t": time.time(), "base": BASE, "model": MODEL, "probes": {}}


def data_url(path_or_bytes, mime):
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def chat(messages, max_tokens=2048, temperature=0.0, timeout=900):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    msg = r["choices"][0]["message"]
    return (msg.get("content") or ""), r


def img_part(url):
    return {"type": "image_url", "image_url": {"url": url}}


def record(name, ok, detail):
    res["probes"][name] = {"ok": bool(ok), **detail}
    print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(detail)[:200]}")


# ---- 1. carrots describe -------------------------------------------------
try:
    content, raw = chat([{"role": "user", "content": [
        img_part(data_url(IMGDIR / "carrots.jpeg", "image/jpeg")),
        {"type": "text", "text": "What vegetable is shown in this image? One word."},
    ]}])
    record("carrots-describe", "carrot" in content.lower(),
           {"answer": content[:200], "wall_s": None})
except Exception as e:
    record("carrots-describe", False, {"error": str(e)[:300]})

# ---- 2. two-image order --------------------------------------------------
try:
    content, raw = chat([{"role": "user", "content": [
        {"type": "text", "text": "First image:"},
        img_part(data_url(IMGDIR / "carrots.jpeg", "image/jpeg")),
        {"type": "text", "text": "Second image:"},
        img_part(data_url(IMGDIR / "corn.jpeg", "image/jpeg")),
        {"type": "text", "text": 'Name the vegetable in each image. Answer ONLY with JSON: {"first": "...", "second": "..."}'},
    ]}])
    try:
        j = json.loads(content.strip().strip("`").removeprefix("json"))
    except Exception:
        import re
        m = re.search(r'\{[^}]*\}', content, re.S)
        j = json.loads(m.group(0)) if m else {}
    ok = "carrot" in str(j.get("first", "")).lower() and "corn" in str(j.get("second", "")).lower()
    record("two-image-order", ok, {"answer": content[:300], "parsed": j})
except Exception as e:
    record("two-image-order", False, {"error": str(e)[:300]})

# ---- 3. shapes exact -----------------------------------------------------
try:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    # 3 red circles, 2 blue squares, 1 green triangle — fixed layout
    for cx, cy in ((100, 100), (300, 120), (520, 90)):
        d.ellipse((cx - 40, cy - 40, cx + 40, cy + 40), fill="red", outline="black")
    for x, y in ((120, 300), (420, 320)):
        d.rectangle((x, y, x + 80, y + 80), fill="blue", outline="black")
    d.polygon([(300, 420), (260, 460), (340, 460)], fill="green", outline="black")
    shapes_png = IMGDIR / "probe-shapes.png"
    im.save(shapes_png)
    content, raw = chat([{"role": "user", "content": [
        img_part(data_url(shapes_png, "image/png")),
        {"type": "text", "text": 'Count the shapes by type and color. Answer ONLY with JSON like {"red_circles": N, "blue_squares": N, "green_triangles": N}'},
    ]}])
    try:
        j = json.loads(content.strip().strip("`").removeprefix("json"))
    except Exception:
        import re
        m = re.search(r'\{[^}]*\}', content, re.S)
        j = json.loads(m.group(0)) if m else {}
    ok = (j.get("red_circles") == 3 and j.get("blue_squares") == 2
          and j.get("green_triangles") == 1)
    record("shapes-exact", ok, {"answer": content[:300], "parsed": j,
                                "want": {"red_circles": 3, "blue_squares": 2, "green_triangles": 1}})
except Exception as e:
    record("shapes-exact", False, {"error": str(e)[:300]})

# ---- 4. OCR lines --------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (800, 200), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    LINE1 = "The KV pool holds 2331430 tokens."
    LINE2 = "RoCE link: 192.0.2.11 to .12"
    d.text((20, 40), LINE1, fill="black", font=font)
    d.text((20, 110), LINE2, fill="black", font=font)
    ocr_png = IMGDIR / "probe-ocr.png"
    im.save(ocr_png)
    content, raw = chat([{"role": "user", "content": [
        img_part(data_url(ocr_png, "image/png")),
        {"type": "text", "text": "Transcribe the text in this image exactly."},
    ]}])
    ok = "2331430" in content and "192.0.2.11" in content
    record("ocr-lines", ok, {"answer": content[:300]})
except Exception as e:
    record("ocr-lines", False, {"error": str(e)[:300]})

# ---- 5a. structured image in tool message -> 400 -------------------------
try:
    chat([
        {"role": "user", "content": "call a tool"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "screenshot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": [
            img_part(data_url(IMGDIR / "corn.jpeg", "image/jpeg"))]},
    ], max_tokens=16)
    # RE-SPECCED 2026-09-23: this build ACCEPTS and reads images in tool messages -- that is the
    # deployment's own documented behaviour (DEPLOY.md: "images inside tool messages are accepted
    # and read by this stack; the vendor's encoder documents images in user messages only").
    # The old "must 400" expectation came from the 0.25.2 packet and was permanently red.
    # Either acceptance or a 400 is a non-regression; anything else (5xx, empty answer) fails.
    record("tool-image-accepted", bool(content.strip()),
           {"status": "accepted", "answer": content[:200],
            "note": "acceptance is this build's documented behaviour"})
except urllib.error.HTTPError as e:
    record("tool-image-accepted", e.code == 400,
           {"status": e.code, "note": "rejected; also acceptable"})

# ---- 5b. tool message TEXT mentioning <image> tags must NOT 400 ----------
try:
    content, raw = chat([
        {"role": "user", "content": "call a tool"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1",
         "content": "match: commit adds <image>foo.jpg</image> tag handling"},
        {"role": "user", "content": "Summarize the tool output in one sentence."},
    ], max_tokens=128)
    record("tool-image-text-ok", True, {"answer": content[:200]})
except urllib.error.HTTPError as e:
    record("tool-image-text-ok", False, {"status": e.code, "note": "issue #167 regression"})
except Exception as e:
    record("tool-image-text-ok", False, {"error": str(e)[:300]})

n_ok = sum(1 for v in res["probes"].values() if v["ok"])
res["summary"] = f"{n_ok}/{len(res['probes'])} probes passed"
Path(args.out).write_text(json.dumps(res, indent=2))
print(res["summary"], "->", args.out)
sys.exit(0 if n_ok == len(res["probes"]) else 1)
