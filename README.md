# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark (GB10)

Serve a 305B-parameter MoE with **1,048,576-token context** and native image input
across two NVIDIA DGX Sparks at tensor-parallel 2 over a direct ConnectX link.

This repo is the whole deployment: pinned image, serving config, supervisor, the
patches this stack needs, and a verification battery. Point an agent at it and it
should be able to stand the stack up.

---

## Performance

Measured on 2× DGX Spark (GB10, 128 GB unified each) over the direct 200 Gb/s link.
Client on a third machine; nothing benchmarked on the servers themselves.

**Decode throughput vs concurrency** (pp 2048 / tg 128, 3 runs each). Concurrency scales
decode well at short context. At 64K the cost is not decode — it is that **prefills
serialise**, so every request waits for the one before it to finish prefilling:

| Streams | Depth 0 agg tok/s | Depth 0 peak | 64K agg tok/s | 64K peak | 64K TTFR (first→last req) |
|---:|---:|---:|---:|---:|---:|
| 1 | 41.7 | 52 | 43.3 | 51 | 36 s |
| 2 | 58.7 | 76 | 6.6 | 73 | 37 – 73 s |
| 4 | 60.5 | 111 | 4.6 | 77 | 37 – 145 s |
| 8 | 73.6 | 162 | 3.9 | 74 | 37 – 292 s |
| 16 | **81.0** | **230** | 3.7 | 74 | 37 – **584 s** |

Read the **peak** column, not the aggregate. Peak decode throughput *rises* with
concurrency at 64K too (51 → 77 tok/s, saturating near 4 streams) — the engine decodes
fine. The aggregate is low because a 64K cell is ~95 % prefill: 16 requests carry
1.08 M prefill tokens against 2 K decode tokens.

The serialisation is exact. `16 × 67,584 tokens ÷ 1,852 tok/s = 583.9 s`, and the
measured worst TTFR is **584.0 s**: each prefill runs to completion before the next
begins. Prefill rate itself is flat (~1,850 tok/s) at every concurrency level; single
stream at depth 0 manages 2,453 tok/s only because it has no queue behind it.

**This is very likely tunable — and worth testing on your hardware.** `LPT`
(`--long-prefill-token-threshold`) is unset in `config/R-baseline.env`, so a long prefill
receives the entire per-step budget and cannot be interleaved. The scheduler caps new
prefill tokens per step at `min(MNBT, LPT)`, so setting `LPT` below `MNBT` should chunk
long prefills and spread first-token latency rather than stacking it.

To be explicit about the limits of what we measured: the serialisation and the TTFR
spread above are measured; **the fix is inferred, not measured at this concurrency and
depth.** The mechanism is from the scheduler implementation, and `LPT=1024` was measured
on this stack cutting worst-case short-request latency during a 100K prefill from 39.9 s
to 1.9 s — a different scenario. If you set `LPT`, re-measure.

**Practical guidance:** size a long-context deployment by stream count, not aggregate
throughput, and set `LPT` deliberately for your latency profile.

**Prefill and decode vs context depth** (single stream, 3 runs each):

| Context depth | Prefill tok/s | Decode tok/s | Time to first token |
|---:|---:|---:|---:|
| 0 | **2,376** | **44.5** | 1.1 s |
| 16,384 | 1,962 | 46.9 | 9.6 s |
| 65,536 | 1,845 | 46.7 | 36.9 s |
| 262,144 | 1,538 | 43.6 | 172.0 s |
| 524,288 | 1,234 | 34.3 | **428.6 s** |

Decode is essentially flat to 256K context. Prefill degrades gracefully. A 512K-token
prompt costs **7.1 minutes** to first token.

**Quality** (tool-eval-bench v2.7.0, hard mode, 88 scenarios × 3 trials):

| Run | Score | Points | Errors |
|---|---:|---:|---:|
| Sequential | **90**/100 | 158/176 | 0 |
| `--parallel 6` | **89**/100 | 157/176 | 0 |

| Benchmark | Result |
|---|---|
| GSM8K | **95.5 %** (191/200) |
| IFEval | **75.8 %** prompt (410/541), **78.9 %** instruction (658/834) |
| Needle-in-a-haystack, ≤81K | **100 %** (20/20), all positions |
| **Needle-in-a-haystack, 1,022K context** | **100 %** (6/6) — start, middle and end of a 1.04M-token haystack |
| Speculative decoding (DSpark k=3) | 46.5–69.9 % acceptance, 2.4–3.0× speedup ceiling |

**Serving envelope:** KV pool **3,785,457 tokens** — 3.60× concurrency at full 1M context.

Full methodology and raw data: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Requirements

| | |
|---|---|
| Nodes | 2× NVIDIA DGX Spark (GB10), 128 GB unified memory each |
| Interconnect | Direct ConnectX-7 link between them (RoCE), port-to-port |
| OS | DGX OS / Ubuntu 24.04, kernel `6.17.0-1026-nvidia`, driver `580.173.02` |
| Disk | ~165 GB per node for the weights (each node holds the full set) |
| Docker | present by default on DGX OS |

Both nodes load the same weights — tensor parallelism shards *memory*, not disk.

---

## Quick start

On **both** nodes, unless stated otherwise. Replace `HEAD` and `WORKER` with your
two hostnames and `USER` with your account.

**1. Pull and pin the image.** Registry-published, pinned by digest:

```bash
IMAGE="0rand/vllm_spark_dsv4-0.29-b12x@sha256:85eb91eec0e7a70c7c287ec04c9699f9604ebbd90b15b6b370104b60b605be18"
docker pull "$IMAGE"
docker tag  "$IMAGE" stack029/orand:85eb91ee
```

**2. Fetch the weights** (~165 GB) into the local HuggingFace cache:

```bash
export HF_HUB_DISABLE_XET=1
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 86f746b36186f0e567729a5c06a8c918caba82a9
```

Download once, then `rsync` to the peer over the direct link. Pin the revision — the
repo is updated in place, and an unpinned `main` will point at a snapshot you do not have.

**3. Install the stack** on the head node:

```bash
git clone <this repo> && cd dsv4-flash-2x-spark-deployment
HEAD=spark1 WORKER=spark2 SSH_USER=$USER ./supervision/install.sh
```

**4. Serve it.** The launcher runs preflight → launch → health → gates:

```bash
sudo systemctl start stack029-vllm
journalctl -u stack029-vllm -f          # ~6–10 min cold to /health
curl -s localhost:8000/v1/models | jq -r '.data[].id'
```

Full walkthrough including the fan-control and Secure Boot steps:
[`docs/SETUP.md`](docs/SETUP.md).

---

## Layout

```
config/R-baseline.env       the serving profile — every knob, with rationale in comments
supervision/
  install.sh                deploy launcher + unit to the head node
  stack029-launch.sh        gated launcher: preflight, launch, health, 5-gate battery, monitor
  stack029-vllm.service.template
overlays/                   three vLLM source files bind-mounted read-only (see docs/CONFIG.md)
patches/
  dsv4-mm-prefix-span.patch upstream-shaped patch the vision overlay is generated from
verify/                     gate battery and independent probes
benchmarks/                 raw result JSON behind docs/BENCHMARKS.md
docs/
  SETUP.md                  step-by-step recreation, including Secure Boot + fan control
  CONFIG.md                 what each serving knob does and why it is set that way
  BENCHMARKS.md             methodology, full result tables, raw data
  TROUBLESHOOTING.md        known failure modes and what they look like
  FAN-CONTROL.md            thermal management: fan floor curve + why it is needed
```

---

## Verification

Every start runs a 5-gate battery before the server is considered up:

```bash
./supervision/stack029-launch.sh gates config/R-baseline.env   # re-run against a live server
```

1. Model fingerprints + smoke + unknown-reasoning-level handling
2. 14-check basics battery (vision-parametrised)
3. 6 vision probes
4. 130 s c8 mixed soak with clock sampling (must peak ≥ 2400 MHz)
5. NVRM allocation-failure counts and host-memory floor (recorded)

Any gate failure stops the cluster. Independent probes in `verify/` can be run
against a live endpoint at any time.

---

## Known issues

Summarised here, detailed in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md):

- **Full-context requests are heavy.** A 1M-token prefill sustains the board near its
  thermal limit and holds ~1M of the 3.79M-token KV pool. See `docs/FAN-CONTROL.md`.
- **Post-reboot firmware clock pinning.** After some reboots the GPU holds ~650–930 MHz
  under load instead of ~2500 MHz, which fails the clock gate. Another reboot clears it.
- **`expandable_segments` materially changes KV capacity** (+18 % when disabled) but
  interacts with the prefill memory high-water mark. Rationale in `docs/CONFIG.md`.
- **A featureless image is described wrongly** — a flat single-colour field answers
  "White" regardless of actual colour. Structured images are read correctly.

---

## Credits

This builds on other people's work; the parts that are ours are marked as such in the
docs.

- **Model:** [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai) (MIT)
- **Image:** [`0rand/vllm_spark_dsv4-0.29-b12x`](https://hub.docker.com/) — a registry-published
  community build of vLLM 0.28.1rc1 (B12X lineage) with native DSV4 vision + DSpark
- **Launch orchestration:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  — `launch-cluster.sh` drives the TP=2 pair
- **Fan control:** [thewh1teagle/sparkfan](https://github.com/thewh1teagle/sparkfan) — kernel
  gate + daemon; used unmodified except for a curve argument
- **EC protocol reverse engineering:** [Z841973620](https://github.com/Z841973620/dgx-spark-fan-override),
  [xXLegionBinFrogXx](https://github.com/xXLegionBinFrogXx/gb10-fan-control)
- **Benchmarks:** [eugr/llama-benchy](https://github.com/eugr/llama-benchy) (MIT),
  [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (MIT)

## License

MIT — see [LICENSE](LICENSE).
