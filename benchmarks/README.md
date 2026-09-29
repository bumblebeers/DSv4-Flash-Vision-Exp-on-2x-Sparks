# Raw benchmark data

`docs/BENCHMARKS.md` summarises these. This directory holds the raw results.

## `u924/` — the deployed config

`vllm bench serve` output from the deployed `R-mnbt2304-nccl-u924` config
(`GPU_UTIL 0.924`, NCCL reclaim, KV pool 4,682,729 tokens). Each file is one stage; the
stage index and exact commands are in the suite's `stages.tsv` / `stages-partA.tsv`.

- `conc{1,2,4,8,16}.json` — concurrency sweep, natural text, 1K-token prompts
- `nt8k-c{1,2,4,8,16}*.json` — concurrency sweep, natural text, 8K-token prompts
- `app-{58,1800,7200}-*.json` — application-shaped prompts at c1 (warm + 3 runs)
- `forum-single.json`, `nt-forum-single.json` — the forum-standard 1024 in / 4096 out request
- `benchy-depth.json` — llama-benchy depth curve (Part C)

The suite's prompt generator and harnesses are in `results/bench-20260926/` of the source
cluster (not shipped here). The `--dataset-name custom` loader requires JSONL plus `pandas`,
and does not truncate to `--input-len`, so one JSONL per target length is required.

## Top-level `*.json` — the reference config

llama-benchy `0.4.0` runs on the reference `R-baseline` config (`GPU_UTIL 0.88`, KV pool
3,785,457 tokens) on natural text (Shakespeare), for the depth/concurrency grid that
established the prefill-serialisation finding. Retained as the reference-config data; the
deployed numbers are in `u924/`.
