# Configuration

The active profile is [`config/R-baseline.env`](../config/R-baseline.env). Every value
below is either a knob we set deliberately or a patch we apply, with the reason.

Read the profile itself too — it carries the same rationale inline, so the file is
self-explanatory on the node.

---

## Serve shape

| Setting | Value | Why |
|---|---|---|
| `GPU_UTIL` | **0.88** | GPU and host share **one** 121.6 GiB pool. 0.88 asks for 107.03 GiB (79.5 weights + 21.3 KV), leaving the host ~14.6 GiB nominally. Higher does not boot: the engine checks free device memory at startup and 0.92 fails cleanly with `Free memory on device cuda:0 (110.48/121.63 GiB) … less than desired`. The practical ceiling is ≈0.908. |
| `MAX_NUM_SEQS` | **16** | Concurrency. Each running sequence submits `k+1 = 4` verify rows under DSpark, so 16 sequences = 64 rows — exactly FlashInfer's decode-dispatch boundary (`_DECODE_MAX_TOKENS = 64`; above it, requests route to the prefill orchestrator instead of the dedicated decode kernels). |
| `MAX_BATCHED` | **2048** | `max_num_batched_tokens`. For this hybrid sliding-window model the batch budget is counted **in the KV pool sizing**, so lowering it *increases* the pool. 4096 → 2048 bought ≈ +0.9 M tokens of pool. |
| `_LPT_` (`--long-prefill-token-threshold`) | **unset** | The effective per-step prefill cap is `min(MNBT, LPT)`. Setting `LPT` below `MNBT` reserves budget for short requests arriving during a long prefill — at the cost of ~10–20 % on the long prefill itself, and of KV pool. Left unset here; set it to ~1024 if you serve interactive traffic alongside long prompts. |
| `K` | **3** | DSpark speculative tokens. Matches the model's `num_nextn_predict_layers`. |
| `DRAFT_METHOD` | `probabilistic` | DSpark drafting mode. |
| `EFFORT` | `high` | Default reasoning effort. Per-request overrides work. |
| `LIMIT_IMAGES` | **8** | Max images per prompt. |
| `RETENTION` | **4096** | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`. Prefix caching is on. |
| `FORCE_A16` | **1** | `VLLM_B12X_MOE_FP4_FORCE_A16`. Required for the MXFP4 MoE path on this build. |
| `OVERRIDE_GEN` | `temperature 1.0, top_p 0.95` | The model card's agentic recommendation, applied server-side so clients need not send it. |
| `EP` | **0** | Expert parallelism off — TP=2 only. |

## Allocator

```
EXTRA_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
```

Set **explicitly** rather than left to the launcher's default, so an upstream default
change cannot silently move it.

This one is worth understanding, because it is the single largest capacity lever and
it trades against something real. vLLM sizes the KV pool as
`(total × util) − non-KV memory measured during a profiling pass` — the pool is a
*remainder*. `expandable_segments:True` packs allocations into growable virtual
segments to fight fragmentation, and retains freed memory inside them rather than
returning it, so the profiler measures a larger non-KV footprint and less is left for KV.

| `expandable_segments` | KV pool | Note |
|---|---:|---|
| `True` | 3,401,076 | less fragmentation-prone; **what the reference deployment ran** |
| `False` | 4,017,245 | **+616,169 tokens (+18.1 %)**, more fragmentation-prone |

We run `True`. The +18 % is real and measured, but `False` is the more
fragmentation-prone mode, and fragmentation is the documented mechanism behind the
prefill memory high-water mark on this stack: with the native allocator a freed
smaller block cannot serve the next larger request, so *reserved* grows with prompt
length. Capacity bought that way is capacity that can be eaten back by workload.

If you want the extra 18 % and can bound your prompt lengths, flip it — it is one line.

## Patches and overlays

Two patches are applied at container start, both bind-mounted read-only over the
image's own files. They are **not** upstream — they are ours, kept in this repo so the
running stack is reproducible rather than hand-edited on the node.

### Vision mm-prefix-span fix — `MMPREFIX_FIX=1`

**The defect.** With the V2 model runner, `compute_mm_prefix_ranges` derives the
mm-prefix bidirectional attention ranges from `PlaceholderRange.extract_embeds_range()`
— per-row-pair runs — instead of the full sentinel block `[pad + IMAGE_START … IMAGE_END]`.
Image rows therefore leak into causal attention, and **predicted image coordinates
collapse to the centre of the image**.

Measured with a coordinate probe (marker at known positions, slope of predicted-vs-true):

| | before | after |
|---|---:|---:|
| y-slope | 0.287 | **0.981** (r 0.999) |
| y mean absolute error | 19.09 % (146.6 px) | **0.79 %** (6.1 px) |

Before the fix, predicted y was a constant `50.0` while x tracked its marker at
r = 0.971 — which is what identifies it as y-specific rather than a general vision fault.

**Both halves are required.** The overlay reads the two flags via
`getattr(hf_text_config, …, 0/False)` and silently takes the *unfixed* branch when they
are absent — and they are absent from this checkpoint's `config.json`. So the code
overlay alone is a no-op; the flags must be injected as well:

```
--hf-overrides '{"mm_prefix_span_leading_pad_modulus":4,"mm_prefix_clamp_sliding_window":true}'
```

Supplied via the profile's `EXTRA_ARGS`; `MMPREFIX_FIX=1` makes the launcher bind-mount
both overlay files and check they exist on **both** nodes during preflight.

- Sources: [`overlays/attn_utils__mmspan.py`](../overlays/attn_utils__mmspan.py),
  [`overlays/default__mmspan.py`](../overlays/default__mmspan.py)
- Patch of record: [`patches/dsv4-mm-prefix-span.patch`](../patches/dsv4-mm-prefix-span.patch)

> Note: the standard gate battery passes vision **with this defect live**, because
> every probe in it is describe / count / OCR — none of which the defect perturbs.
> That is why a coordinate probe exists separately.

### Parser patch — `PARSER_FIX=1`

Replaces `vllm/parser/deepseek_v4.py` with
[`overlays/parser_deepseek_v4__d98c8c0.py`](../overlays/parser_deepseek_v4__d98c8c0.py).
Reasoning and tool-call parsing on this build.

## Things we deliberately did *not* change

Measured and rejected, recorded so you do not have to re-derive them:

| | Why not |
|---|---|
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | +0.39 M pool tokens, but removes the CUDA-graph memory reserve and drops the host memory floor from 3.8 GiB to 1.2 GiB. |
| `--kv-cache-dtype` other than `fp8` | fp8 costs essentially nothing in throughput here and sizes the pool for the full 1M context. |
| SM12x `o_proj` repair patches | Already baked into this image. Verified: the `o_proj.py` recipe and 3-D reshape are present. |
| FlashInfer "resolved plan" patches | Target a newer FlashInfer than this image's 0.6.18 (flat `_sparse_mla_sm120.py`). The boot completes `FULL_AND_PIECEWISE` capture without them. |

## Environment variables the launcher sets

`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`,
`VLLM_B12X_MOE_FP4_FORCE_A16`, `PYTORCH_CUDA_ALLOC_CONF`, and the two overlay mounts.
Reproduce the exact command for any boot from its record:

```
~/stack029/records/<timestamp>__<profile>/serve.sh
```

---

## Settings we investigated before publishing

Three things in the launch command are non-obvious. Each was traced to the source in
the image, because guessing on a forum thread is worse than saying "we checked".

### `--block-size 256` does not set the effective block size

The engine reports `block_size=4` while accepting `--block-size 256`. Not a bug and not
a silent clamp — the effective value is computed after argument parsing:

```python
# vllm/v1/engine/core.py:345
participating = [g.kv_cache_spec.block_size for g in kv_cache_groups
                 if g.kv_cache_spec.prefix_cacheable]
vllm_config.cache_config.block_size = min(participating or [...])
```

The block size is the **minimum across all prefix-cacheable KV-cache groups**. This
model's group sizes are hardcoded, fixed by tensor sharing between compressor state and
KV blocks:

| Group | `block_size` | Why |
|---|---:|---|
| C4 compressor (`compress_ratio=4`) | **4** | C4 block shape `[4, 2*512*2*4]` |
| C128 compressor (`compress_ratio=128`) | 8 | C128 block shape `[8, 512*2*4]` |
| SWA | 64 | shares a page with the C4A KV blocks |

`min(4, 8, 64) = 4`. Both source sites carry the same note: *"TODO(yifan): make block
size automatically determined and configurable."*

The model's KV block shape is hardcoded as `[256//4, head_dim] = [64, 584]`, so `256`
appears to be a structural constant of the architecture rather than an arbitrary flag.
**We verified that the reported block size is the group minimum; we did not verify
whether `--block-size 256` is load-bearing for coherence with those hardcoded shapes.**
It is inherited from the vendor sample, not a value we chose.

### The empty reasoning strings are intentional

```
--reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
```

Empty looks wrong but is the correct encoding for "let the parser decide". The config
only *fills in* a string when its own is falsy:

```python
# vllm/config/reasoning.py:88
start_token = reasoning_parser.reasoning_start_str
if start_token and not reasoning_start_str:      # "" is falsy -> parser wins
    reasoning_start_str = start_token
```

and the `deepseek_v4` parser hardcodes its delimiters:

```python
# vllm/parser/deepseek_v4.py:43
DSML_THINK_START = "<think>"
DSML_THINK_END   = "</think>"
```

So thinking is delimited by `<think>` / `</think>`, supplied by the parser. Passing
non-empty strings here would **override** the parser, not enable anything.

### Inherited engine flags, and one that is load-bearing

| Flag | What it does | Provenance |
|---|---|---|
| `--skip-mm-profiling` | Omits the multimodal encoder from the memory-profiling pass, so the KV pool is not reduced by the vision encoder's peak. A capacity/margin trade — this is part of why the pool reaches 3.86 M tokens. | vendor sample |
| `--enable-prompt-tokens-details` | Adds the prompt-token breakdown to `usage`, which is where `multimodal_tokens` comes from — useful for confirming a model actually conditioned on an image. | vendor sample |
| `VLLM_USE_V2_MODEL_RUNNER=1` | Selects the V2 model runner (tri-state in `envs.py`; `1` forces it on). **This is load-bearing for our vision fix** — the overlay patches `vllm/v1/worker/gpu/model_states/default.py`, a V2-runner file. Turning it off would silently disable the mm-prefix-span repair. | image |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | Enables `compilation/breakable_cudagraph.py` (default `False`). Boot log confirms `Breakable CUDA graph enabled`. | image |

`config/serve.sh.example` is the fully rendered engine command from a working boot, so
you can diff it against your own without standing the stack up first.
