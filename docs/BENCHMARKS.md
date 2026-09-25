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

pp 2048 / tg 128, 3 runs per point, at two context depths. **Read peak, not aggregate** —
see the note below.

| Streams | Depth 0 agg | Depth 0 peak | Depth 0 TTFR | 64K agg | 64K peak | 64K TTFR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.69 | 52.0 | 1.06 s | 43.31 | 50.7 | 36.1 s |
| 2 | 58.72 | 75.7 | 1.91 s | 6.56 | 73.0 | 37.0 – 72.5 s |
| 4 | 60.45 | 110.7 | 3.20 s | 4.55 | 77.0 | 37.0 – 145.3 s |
| 8 | 73.56 | 161.7 | 5.47 s | 3.94 | 74.0 | 37.1 – 291.5 s |
| 16 | 80.97 | 230.0 | 9.66 s | 3.69 | 74.3 | 37.2 – 584.0 s |

### How to read this

**Decode capability scales with concurrency at both depths.** Peak decode throughput at
64K rises 50.7 → 77.0 tok/s and saturates near 4 streams. The engine is not decode-limited.

**The 64K aggregate is low because the cell is prefill-dominated.** At 16 streams the
cell carries 1.08 M prefill tokens and 2 K decode tokens — roughly 95 % prefill. An
aggregate figure computed over the cell's wall clock therefore reports mostly the
prefill, not the decode.

**Prefills serialise completely at this configuration.** The arithmetic is exact:
`16 × 67,584 ÷ 1,852 tok/s = 583.9 s`, against a measured worst TTFR of **584.0 s**.
Each request's prefill runs to completion before the next begins, so request *N* waits
`N × ~36 s` for its first token. Prefill *rate* is unaffected — 1,852–1,881 tok/s at
every concurrency level — it is prefill *concurrency* that is absent.

**This is probably a tunable, not an engine limit — but we did not test it here.**
`LPT` is unset in `config/R-baseline.env`, so a long prefill receives the whole per-step
budget (`min(MNBT, LPT)` = `MNBT`) and cannot be interleaved with other requests. Setting
`LPT` below `MNBT` should chunk long prefills and spread first-token latency instead of
stacking it; total prefill work is unchanged either way, so aggregate throughput should
not move.

What is **measured** is the serialisation, the TTFR spread, and the flat prefill rate.
What is **inferred** is that `LPT` removes the head-of-line blocking — the reasoning comes
from the scheduler implementation plus a separate measurement on this stack where
`LPT=1024` cut worst-case short-request latency during a 100K prefill from 39.9 s to
1.9 s. That is a different workload from 16 concurrent 64K requests, so treat the fix as
untested here and re-measure if you set it.

Single-stream 64K decode (43.31) is slightly above depth 0 (41.69): context on its own
costs nothing. What costs is a *queue* of long-context requests behind it.

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

## Depth × concurrency (independent grid)

A separate 2-run grid at intermediate depths, to check the trend between the two points
above. Same picture: aggregate falls as streams rise, while per-request first-token
latency stacks.

| Depth | Streams | Aggregate decode tok/s | TTFR |
|---:|---:|---:|---:|
| 16,384 | 4 | 15.46 | 24.8 s |
| 16,384 | 8 | 14.03 | 44.3 s |
| 16,384 | 16 | 13.44 | 83.2 s |
| 65,536 | 4 | 3.94 | 93.4 s |
| 65,536 | 8 | 3.85 | 167.6 s |
| 65,536 | 16 | 3.61 | **317.4 s** |

The TTFR growth is the signal: 16,384 × 16 gives 83 s, 65,536 × 16 gives 317 s, and the
full-range run above reaches 584 s at the same stream count. Latency scales with the
*total prefill queued ahead of a request*, which is `streams × depth` — the definition of
head-of-line blocking. Note this grid predates the `--context-size` correction, so treat
its absolute numbers as indicative and the trend as the finding.

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
