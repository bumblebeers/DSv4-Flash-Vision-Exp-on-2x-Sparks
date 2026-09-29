# Agent guide

This repo is a complete, reproducible deployment of DeepSeek-V4-Flash-Vision-Exp on a 2× DGX
Spark (GB10) cluster. You can stand it up from scratch by following the runbook.

## To stand the stack up on new hardware

Read in order:

1. `docs/SETUP.md` — the step-by-step recreation: image, weights, install, serve, thermal.
2. `docs/CONFIG.md` — every serving knob, the patches, and why each is set that way.
3. `docs/TROUBLESHOOTING.md` — failure modes, symptoms, and fixes.

The short path (on the head node, from a clone of this repo):

```bash
HEAD=<head> WORKER=<worker> SSH_USER=<user> ./supervision/install.sh
sudo systemctl start stack029-vllm
```

The launcher (`supervision/stack029-launch.sh`) runs preflight → render → launch → `/health`
→ record → a 5-gate battery, then monitors the pair. It refuses to start on 7 classes of
preflight failure and destroys the pair if any gate fails. Run the battery against a live
server at any time with:

```bash
./supervision/stack029-launch.sh gates config/<profile>.env
```

## Key invariants

- **Image pinned by digest, weights by revision.** Do not use `:latest` or an unpinned
  `main` — the Hub repo is updated in place, and an unpinned `main` points vLLM at a
  snapshot that is not on disk (boot fails with `Invalid repository ID or local directory
  specified`).
- **The vision fix needs both halves.** The overlays (`overlays/attn_utils__mmspan.py`,
  `overlays/default__mmspan.py`) plus the two `--hf-overrides` flags in the profile's
  `EXTRA_ARGS`. Either alone is a silent no-op. `MMPREFIX_FIX=1` is the launcher switch.
- **`--kv-cache-dtype fp8` is required** for the full 1M context; the engine refuses to start
  with `estimated maximum model length is 981376` otherwise.
- **The gate battery is the definition of "up".** `/health` returning 200 is necessary but not
  sufficient. Any gate failure stops the cluster.

## Reproducing benchmarks

`docs/BENCHMARKS.md` has the methodology, exact commands, and the raw data in `benchmarks/`.
Decode figures must be read as **sustained rates on natural text** — random token IDs cannot
be speculated and understate decode by 55–69 %. Always pass `--context-size 1048576` to
tool-eval-bench; it auto-detects the context window from the KV pool and would otherwise test
at ~8 % of the intended depth.

## Files

| path | what |
|---|---|
| `config/` | serving profile — every knob with rationale in comments |
| `supervision/` | `install.sh`, `stack029-launch.sh`, `stack029-vllm.service.template` |
| `overlays/` | the vision + parser vLLM source files (bind-mounted read-only) |
| `patches/` | the upstream-shaped patch the vision overlay is generated from |
| `verify/` | gate battery + independent probes |
| `benchmarks/` | raw benchmark JSON behind `docs/BENCHMARKS.md` |
| `docs/` | the runbook: SETUP, CONFIG, BENCHMARKS, TROUBLESHOOTING, FAN-CONTROL |
