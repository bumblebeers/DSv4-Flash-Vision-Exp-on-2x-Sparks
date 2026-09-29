# Benchmarks

Everything here was measured on the deployed configuration in
[`config/R-mnbt2304-nccl-u924.env`](../config/R-mnbt2304-nccl-u924.env) — `GPU_UTIL=0.924`,
`MAX_NUM_SEQS=16`, `MAX_BATCHED=2304`, `LPT=2048`, `expandable_segments:True`, fp8 KV,
prefix caching on, NCCL pinned-buffer reclaim, KV pool **4,682,729 tokens** (4.47× the full
1M context).

## Methodology

- **Client runs on a third machine**, never on the Sparks. The servers are an appliance;
  no benchmark process, probe or client ran on them.
- Endpoint `http://<head>:8000/v1`, model `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.
- Tools pinned: [llama-benchy](https://github.com/eugr/llama-benchy) `0.4.0`,
  [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) `v2.7.0`.
- **Decode is quoted as a sustained rate on natural text.** The community-standard figure
  is published on a random-token corpus, which cannot be speculated. On this model that
  understates decode throughput by **55–69 %**. Two corpora, two different problems —
  see the note at the end of this section.
- llama-benchy corpus: Project Gutenberg *Complete Works of Shakespeare*
  (`https://www.gutenberg.org/cache/epub/100/pg100.txt`), 1,473,993 tokens, tokenized with
  the model's own tokenizer. The tool's **default** corpus is Sherlock Holmes at
  **142,813 tokens**, which caps `--depth` far below this model's context — always check the
  `Total tokens available in text corpus` line in your own run.
- Tokenizer verified as the real one — the tool silently falls back to `gpt2` otherwise,
  which invalidates every throughput figure. Warm-up delta on this model:
  `Server: 105, Local: 22`.
- Unless stated, runs use `--latency-mode generation --skip-coherence` and 3 runs per cell.

## The single-stream headline

The forum-standard single request — 1024 prompt / 4096 output tokens, concurrency 1, natural
text:

| | random-token corpus | **natural text** | Δ |
|---|---:|---:|---:|
| output tok/s | 39.43 | **60.73** | **+54.0 %** |
| mean TPOT | 25.07 ms | **16.42 ms** | −34.5 % |
| mean TTFT | 1,192 ms | **225 ms** | −81.1 % |
| acceptance length | 2.22 | **3.36** | +51 % |

**Quote the natural-text figure.** The random-token number is the one the community
publishes, and it is conservative by roughly half on this model because DSpark cannot
speculate random token IDs (acceptance is flat at ~1.5 tokens/draft on the random corpus vs
~2.6–3.4 on natural text).

## Throughput vs concurrency (natural text, 1K-token prompts)

`vllm bench serve`, 1024 in / 256 out, natural text (Shakespeare), JSONL corpus.

| Streams | out tok/s | per-stream | scaling | mean TTFT | TPOT | acc len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.80 | 40.80 | 1.00× | 701 ms | 21.9 ms | 2.58 |
| 2 | 65.17 | 32.59 | 1.60× | 352 ms | 29.3 ms | 2.58 |
| 4 | 93.73 | 23.43 | 2.30× | 436 ms | 40.1 ms | 2.61 |
| 8 | 133.44 | 16.68 | 3.27× | 620 ms | 55.1 ms | 2.63 |
| 16 | **186.46** | 11.65 | **4.57×** | 847 ms | 74.3 ms | 2.62 |

**Decode scales with concurrency.** Aggregate throughput at 16 streams is 4.57× the
single-stream figure. Throughput saturates at c16 — a c32 run reads 104.81 tok/s, no better,
and its mean TTFT is queue-dominated (26.7 s → 36.6 s) and should not be read as a latency
figure.

## Throughput vs concurrency (natural text, 8K-token prompts)

The depth datapoint that matters for real workloads.

| Streams | out tok/s | per-stream | scaling | mean TTFT | p99 TTFT | TPOT | acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (warm) | **41.20** | 41.20 | 1.00× | 268 ms | 293 ms | 23.3 ms | 2.44 |
| 2 | 61.29 | 30.65 | 1.49× | 370 ms | 550 ms | 31.2 ms | 2.43 |
| 4 | 86.50 | 21.62 | 2.10× | 477 ms | 736 ms | 43.4 ms | 2.41 |
| 8 | 124.10 | 15.51 | 3.01× | 629 ms | 859 ms | 59.0 ms | 2.44 |
| 16 | **172.10** | 10.76 | 4.18× | 890 ms | 1,169 ms | 80.9 ms | 2.41 |

**Depth is nearly free at realistic concurrency**: ~7 % below the 1K sweep at c2–c16, and at
c1 the two are equal (40.80 vs 41.20). Acceptance is essentially unchanged.

## Time to first token across prompt sizes

Application-shaped prompts at c1 (prefix-cache warmed):

| prompt tokens | TTFT median | TTFT range | E2EL median | TPOT median | out tok/s median |
|---:|---:|---:|---:|---:|---:|
| 58 | 301 ms | 298–306 ms | 5.6 s | 20.8 ms | 45.53 |
| 1,800 | 268 ms | 264–682 ms | 6.0 s | 22.3 ms | 42.77 |
| 7,200 | 307 ms | 302–2,461 ms | 6.3 s | 23.2 ms | **40.51** |

**TTFT is roughly flat at ~300 ms from 58 to 7,200 prompt tokens** — 125× the prompt length
for the same time-to-first-token. That is the prefix-cache-and-chunked-prefill path working.
The `out tok/s` column is warm steady state; the first exposure of each length reads lower
(e.g. 30.72 vs 40.51 at 7,200).

## Throughput vs context depth (llama-benchy, single stream)

`llama-benchy --pp 2048 --tg 128 --depth … --concurrency 1 --runs 3`, the exact form the
NVIDIA forum thread uses. llama-benchy drives the model with real prose, so this is natural
text and directly comparable.

| Depth | Prefill tok/s | Decode tok/s | TTFR |
|---:|---:|---:|---:|
| 0 | 2,596 | 41.1 | 1.0 s |
| 4,096 | 2,035 | 45.4 | 3.2 s |
| 16,384 | 1,931 | 40.3 | 9.8 s |
| 32,768 | 1,871 | 43.1 | 18.8 s |
| 65,536 | 1,803 | 42.1 | 37.7 s |
| 131,072 | 1,682 | 38.8 | 79.4 s |
| 262,144 | 1,527 | 43.1 | 173.2 s |
| 524,288 | 1,244 | 34.9 | **423.4 s** |

**Decode is essentially context-independent to ~262K**, and prefill degrades roughly
linearly. A 524K-token prompt costs **7.1 minutes to first token**; a cold 1M-token prefill
costs ~17 minutes (measured 1,052 s at 950 tok/s). Compared to the 0.88 reference, this
config is statistically indistinguishable at depth (prefill +0.2 %, decode −4.1 %, inside the
run-to-run noise) — its advantage is concentrated at shallow context and low concurrency.

> **Note on `tg128`.** It is a 128-token *burst*, not a sustained rate. Quoting it as
> sustained decode was the single worst methodology error in the original corpus (44.5 tok/s
> burst vs 22.7–35.1 sustained on random tokens). The sustained figures are in the concurrency
> tables above — use those for any honest claim. The depth curve is for *shape*, not headline
> rates.

## Quality

### tool-eval-bench — hard mode, 88 scenarios

| Run | Score | Points | Errors | Safety gate |
|---|---:|---:|---:|---|
| 3 trials, sequential | **90**/100 | 158/176 | 0 | FAIL (2 warnings) |
| 3 trials, `--parallel 6` | **89**/100 | 157/176 | 0 | FAIL (2 warnings) |

One point apart across 264 scenario runs each — **no quality degeneracy at concurrency 6**.
The safety warnings are the same behaviour twice: the model **batches a dependent tool call**
(`send_email` alongside `create_calendar_event` in one turn, without waiting for the result).
Worth knowing if you drive this model with tools.

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

**100 % (6/6)**, rating ★★★★★. Effective context **1,046,528 tokens** — the largest haystack
was retrieved at every depth. Duration 2,531 s, 2,626,394 tokens processed. Separately, at 5
positions × 4 haystack sizes up to 81,200 tokens: **100 % (20/20)**.

**This is a heavy operation.** The run sustained the board at up to **96 °C** with both fans
at maximum, and held ~1M of the KV pool per request. Running this without the fan curve is
how a node gets hard-stopped. See [`FAN-CONTROL.md`](FAN-CONTROL.md).

### Speculative decoding (DSpark k=3)

| Metric | Value |
|---|---|
| Acceptance, structured prompts | 69.9 % |
| Acceptance, filler prompts | 47.9–53.2 % |
| Acceptance, code prompts | 46.5 % |
| Draft window utilisation | 2.7 / 3 positions (90 %) |
| Speedup ceiling | 2.4–3.0× |

## Caveats — read before quoting these

- **Single-trial tool-call scores are very noisy.** The three-trial runs above are the only
  ones we would quote. Treat any single run as ±3 points.
- **The needle result is deep but small-n.** 6 needles at full context, 20 at ≤81 K. Every
  one was found; treat it as a strong signal, not a statistical sweep.
- **The 1M throughput point was not completed.** A `--depth 1044480` llama-benchy run
  sustained ~18 minutes and then the head node hard-stopped — no ICMP, both RoCE links
  dropped, physical power cycle required, **with the stock fan curve**. Host memory was flat
  and `/health` was 200 until ~20 s before death, so it was not memory exhaustion. The needle
  run above reached the same context depth *with the fan curve active*, peaked at 96 °C, and
  completed cleanly.
- **Boot-to-boot KV pool varies ±3 %.** That is the CUDA-graph memory profiler, not drift.
- **MMLU is absent**, so the accuracy table is incomplete by one entry.
- **Random vs natural corpus.** Any figure produced on a random-token corpus (`--dataset-name
  random`) is conservative by roughly half on this model. That includes the concurrency rows
  and the forum-standard stage. The public figures in this document are natural text.

## Reproducing

```bash
# throughput (natural text — pass a large corpus; the tool's default caps --depth at ~142K)
uvx --from llama-benchy==0.4.0 llama-benchy \
  --base-url http://<head>:8000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --pp 2048 --tg 128 --depth 0 4096 16384 32768 65536 131072 262144 524288 \
  --concurrency 1 --runs 3 \
  --latency-mode generation --skip-coherence \
  --book-url https://www.gutenberg.org/cache/epub/100/pg100.txt \
  --format json --save-result benchy.json

# quality
uvx --from git+https://github.com/SeraphimSerapis/tool-eval-bench.git@v2.7.0 tool-eval-bench \
  run --hardmode --trials 3 --seed 42 --timeout 600 \
  --base-url http://<head>:8000/v1 \
  --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --context-size 1048576 \
  --json-file hardmode.json
```

Set the client timeout to **≥600 s** for reasoning workloads. Always pass `--context-size
1048576` to tool-eval-bench — it auto-detects the context window as `num_gpu_blocks ×
block_size`, which on this hybrid sliding-window model is not the token capacity.
