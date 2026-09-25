# Benchmarks

Everything here was measured on the configuration in
[`config/R-baseline.env`](../config/R-baseline.env) — `GPU_UTIL=0.88`, `MAX_NUM_SEQS=16`,
`MNBT=2048`, `LPT` unset, `expandable_segments:True`, fp8 KV, prefix caching on,
KV pool **3,785,457 tokens**.

## Methodology

- **Client runs on a third machine**, never on the Sparks. The servers are an appliance;
  no benchmark process, probe or client ran on them.
- Endpoint `http://<head>:8000/v1`, model `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.
- Tools pinned: [llama-benchy](https://github.com/eugr/llama-benchy) `0.4.0`,
  [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) `v2.7.0`.
- llama-benchy corpus: Project Gutenberg *Complete Works of Shakespeare* (pg100),
  **1,832,549 tokens**. The tool's default corpus is only 142,813 tokens, which caps
  `--depth` far below this model's context.
- Tokenizer verified as the real one — the tool silently falls back to `gpt2` otherwise,
  which invalidates every throughput figure. Warm-up delta on this model:
  `Server: 105, Local: 22`.
- `--latency-mode generation`, `--skip-coherence`, 3 runs per cell unless stated.

## Throughput vs concurrency

pp 2048 / tg 128, 3 runs per point, at two context depths.

| Streams | Depth 0 agg | Depth 0 per-stream | Depth 0 TTFR | 64K agg | 64K per-stream | 64K TTFR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.69 | 41.69 | 1.06 s | 43.31 | 43.31 | 36.1 s |
| 2 | 58.72 | 29.36 | 1.91 s | 6.56 | 3.28 | 54.7 s |
| 4 | 60.45 | 15.11 | 3.20 s | 4.55 | 1.14 | 91.2 s |
| 8 | 73.56 | 9.20 | 5.47 s | 3.94 | 0.49 | 164.3 s |
| 16 | 80.97 | 5.06 | 9.66 s | 3.69 | 0.23 | 310.5 s |

**Concurrency helps at short context and destroys throughput at long context.** At depth
0, 16 streams give 1.94× single-stream throughput. At 64K the same 16 streams give
**0.085×** — a 12× collapse, and most of the damage is done by the *second* stream
(43.31 → 6.56). Prefill throughput is flat across every level (1,852–1,881 tok/s at 64K),
so the engine is not prefill-bound; decode is starved because long prefills hold the
scheduler.

Single-stream 64K decode (43.31) is slightly *higher* than depth 0 (41.69). Adding
context costs nothing on its own; adding a *concurrent* long-context stream costs
almost everything.

## Throughput vs context depth

Single stream, pp 2048 / tg 128, 3 runs per point.

| Depth | Prefill tok/s | Decode tok/s | Peak | TTFR |
|---:|---:|---:|---:|---:|
| 0 | 2,375.6 | 44.48 | 51.0 | 1.09 s |
| 4,096 | 2,063.1 | 41.20 | 45.0 | 3.20 s |
| 16,384 | 1,962.1 | 46.91 | 54.7 | 9.62 s |
| 32,768 | 1,909.1 | 45.01 | 51.7 | 18.46 s |
| 65,536 | 1,845.1 | 46.66 | 54.3 | 36.85 s |
| 131,072 | 1,734.6 | 40.77 | 51.7 | 76.97 s |
| 262,144 | 1,538.0 | 43.62 | 50.7 | 172.00 s |
| 524,288 | 1,233.8 | 34.34 | 44.0 | **428.62 s** |

**Decode is essentially context-independent to 262K.** Prefill degrades roughly linearly.
A 512K-token prompt costs 7.1 minutes to first token.

## Depth × concurrency

2 runs per cell. This is the table that matters for planning long-context serving.

| Depth | Streams | Aggregate decode tok/s | Per stream | TTFR |
|---:|---:|---:|---:|---:|
| 16,384 | 4 | 15.46 | 3.87 | 24.8 s |
| 16,384 | 8 | 14.03 | 1.75 | 44.3 s |
| 16,384 | 16 | 13.44 | 0.84 | 83.2 s |
| 65,536 | 4 | 3.94 | 0.99 | 93.4 s |
| 65,536 | 8 | 3.85 | 0.48 | 167.6 s |
| 65,536 | 16 | 3.61 | 0.23 | **317.4 s** |

**Concurrency stops helping at depth.** At depth 0, 16 streams nearly double aggregate
throughput (41.7 → 81.0 tok/s). At 64K it *inverts*: aggregate falls to ~3.6 tok/s and
adding streams only adds latency, because prefills serialise. Plan long-context capacity
by stream count, not by aggregate throughput.

## Quality

### tool-eval-bench — hard mode, 88 scenarios

| Run | Score | Points | Errors | Safety gate |
|---|---:|---:|---:|---|
| 3 trials, sequential | **90**/100 | 158/176 | 0 | FAIL (2 warnings) |
| 3 trials, `--parallel 6` | **89**/100 | 157/176 | 0 | FAIL (2 warnings) |

One point apart across 264 scenario runs each — **no quality degeneracy at concurrency 6**.

The safety warnings are the same behaviour twice: the model **batches a dependent tool
call** (`send_email` alongside `create_calendar_event` in one turn, without waiting for
the result). Worth knowing if you drive this model with tools — it is a correctness
issue for dependent call chains, not a refusal failure.

### Accuracy suite

| Benchmark | Result |
|---|---|
| GSM8K | **95.5 %** (191/200) |
| IFEval | **75.8 %** prompt-level (410/541), **78.9 %** instruction-level (658/834) |
| MMLU | **not measured** — HuggingFace rate-limited the dataset download (HTTP 429), repeatedly |

### Needle-in-a-haystack

**Full context — 1,046,528 tokens:**

| Needle position | 1K haystack | 1,022K haystack |
|---|---|---|
| 0 % (start) | ● | ● |
| 50 % (middle) | ● | ● |
| 100 % (end) | ● | ● |

**100 % (6/6)**, rating ★★★★★. Effective context **1,046,528 tokens** — the largest
haystack was retrieved at every depth. Duration 2,531 s, 2,626,394 tokens processed.

Separately, at 5 positions × 4 haystack sizes up to 81,200 tokens: **100 % (20/20)**,
identical sequential and at `--parallel 6`.

**This is a heavy operation.** The run sustained the board at up to **96 °C** with both
fans at maximum, and held ~1M of the 3.79M-token KV pool per request. See
[`FAN-CONTROL.md`](FAN-CONTROL.md) — running this without the fan curve is how a node
gets hard-stopped.

### Speculative decoding (DSpark k=3)

| Metric | Value |
|---|---|
| Acceptance, structured prompts | 69.9 % |
| Acceptance, filler prompts | 47.9–53.2 % |
| Acceptance, code prompts | 46.5 % |
| Draft window utilisation | 2.7 / 3 positions (90 %) |
| Speedup ceiling | 2.4–3.0× |

## Caveats — read before quoting these

- **Single-trial tool-call scores are very noisy.** The three-trial runs above are the
  only ones we would quote. Treat any single run as ±3 points.
- **The needle result is deep but small-n.** 6 needles at full context, 20 at ≤81 K.
  Every one was found; treat it as a strong signal, not a statistical sweep.
- **The 1M throughput point was not completed.** A `--depth 1044480` llama-benchy run
  sustained ~18 minutes and then the head node hard-stopped — no ICMP, both RoCE links
  dropped, physical power cycle required, **with the stock fan curve**. Host memory was
  flat and `/health` was 200 until ~20 s before death, so it was not memory exhaustion.
  The needle run above reached the same context depth *with the fan curve active*,
  peaked at 96 °C, and completed cleanly. The throughput curve is extrapolated above 524K.
- **Boot-to-boot KV pool varies ±3 %.** That is the CUDA-graph memory profiler, not drift.
- **MMLU is absent**, so the accuracy table is incomplete by one entry.

## Reproducing

```bash
# throughput
uvx --from llama-benchy==0.4.0 llama-benchy \
  --base-url http://<head>:8000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --pp 2048 --tg 128 --depth 0 4096 16384 32768 65536 131072 262144 524288 \
  --concurrency 1 2 4 8 16 --runs 3 \
  --latency-mode generation --skip-coherence \
  --book-url https://www.gutenberg.org/cache/epub/100/pg100.txt \
  --format json --save-result benchy.json

# quality
uvx --from git+https://github.com/SeraphimSerapis/tool-eval-bench.git@v2.7.0 tool-eval-bench \
  run --hardmode --trials 3 --seed 42 --timeout 600 \
  --base-url http://<head>:8000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --json-file hardmode.json
```

Set the client timeout to **≥600 s** for reasoning workloads.
