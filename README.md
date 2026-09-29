# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark (GB10)

Serve the **DeepSeek-V4-Flash-Vision-Exp** MoE — 305 B parameters, native image input, and a
**1,048,576-token context** — across two NVIDIA DGX Spark (GB10) appliances at tensor
parallelism 2, over a direct 200 Gb/s ConnectX-7 RoCE link.

This repo is the whole deployment: the pinned serving image, the serving configuration with
every knob and the reason it is set that way, the two code patches this stack needs, a
supervisor that runs a five-gate verification battery before it will call the server up, and
the benchmark methodology and results.

## How this stack differs

Several public ways exist to run the DSv4-Flash family on a two-node Spark cluster. Most
target the earlier text-only releases (`V4-Flash`, `V4-Flash-0731`); this one targets
**Vision-Exp** and is set up so the vision is correct, not merely served.

| | **This repo** | MiaAI-Lab packet | eugr/spark-vllm-docker | Ollie | GPDev |
|---|---|---|---|---|---|
| Model | **Vision-Exp** | V4-Flash (text) | V4-Flash / -0731 / -Vision-Exp | Vision-Exp | Vision-Exp |
| Engine | `0rand/vllm_spark_dsv4-0.29-b12x` (vLLM 0.28.1rc1.dev475) | `ghcr.io/anemll/…` (vLLM 0.25.2) | built from source | same 0rand image + hot patches | prebuilt image |
| Spatial-vision fix | **in-repo overlay + flags** | — | recipe currently broken | hot-patch mod | — |
| KV pool | **4.68 M tokens** (4.47× full context) | shorter | varies | 1M | 1M |
| Health verification | **5-gate battery** on every start | — | — | partial | preflight |
| Thermal / fan handling | **documented + curve** | — | — | — | — |
| License | MIT | MIT | MIT | none | none |

**The vision fix is the load-bearing difference.** The model's native vision works for
describe / OCR out of the box, but vLLM's V2 model runner has a defect in
`compute_mm_prefix_ranges` that makes *spatial* prediction collapse: image rows leak into
causal attention, so the model pins every predicted y-coordinate to the image centre
(measured y-slope 0.287, 19 % mean error). This repo ships the fix in-repo — an overlay plus
two `--hf-overrides` flags — which brings the y-slope to 0.981 and the mean error to 0.8 %,
verified with a coordinate probe rather than by eye. A deployment that merely serves the model
has the defect live, and its vision gate still passes, because describe / count / OCR are not
perturbed by it.

Beyond the fix, this is the reproducible, verified deployment: the image is pinned by digest
and the weights by revision, every serving knob is explained, and a five-gate battery
(fingerprints, a 14-check basics battery, six vision probes, a 130 s clock soak that requires
≥ 2400 MHz, and NVRM / host-memory accounting) must pass on every start before the server is
declared up. Independent probes are included for re-running against a live endpoint.

## Performance

Measured on the deployed configuration (`GPU_UTIL 0.924`, NCCL pinned-buffer reclaim, KV pool
**4,682,729 tokens** = 4.47× the full 1M context), TP=2, client on a third machine.

**The honest headline — single stream, natural text, the forum-standard shape
(1024 in / 4096 out): 60.7 tok/s.** The same shape on random token IDs reads 39.4 — random
tokens cannot be speculated, so they understate decode by ~55–69 %.

| Metric | Value |
|---|---|
| Single stream, 1024 in / 4096 out | **60.7 tok/s** (TTFT 225 ms, TPOT 16.4 ms) |
| Single stream, 1024 in / 256 out | 40.8 tok/s |
| Aggregate, 16 streams, 1K-token prompts | **186.5 tok/s** (4.6× scaling from 1) |
| Aggregate, 16 streams, 8K-token prompts | 172.1 tok/s |
| Time to first token, 58 → 7,200 prompt tokens | **~300 ms, flat** |
| Prefill rate | 2,471 tok/s (`vllm bench`), 2,596 tok/s (depth-0 llama-benchy) |
| KV pool | **4,682,729 tokens** (4.47× the full context) |

**Quality** (model metrics, independent of serving config): tool-eval-bench hard mode
**90/100** (3 trials, 158/176 points, 0 errors); GSM8K **95.5 %**; IFEval **75.8 %**
prompt-level; needle-in-a-haystack **100 %** at 1,046,528 tokens (start, middle, end).
DSpark speculative decoding acceptance 46.5–69.9 %, a 2.4–3.0× speedup ceiling.

**The one thing to know about long context.** Prefill rate is roughly flat (~1,800–2,500 tok/s),
but a long prefill consumes most of the per-step budget, so concurrent long-context requests
queue behind it. Single-stream decode is context-independent to ~262K; a 524K-token prompt
costs **7.1 min to first token**, and a cold 1M-token prefill **~17 min**. The deployed profile
sets `LPT` to reserve a small per-step budget for short requests. Size a long-context
deployment by stream count, not aggregate throughput.

Full methodology, raw data, and the depth/concurrency grids:
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## What you need

| | |
|---|---|
| Nodes | 2× NVIDIA DGX Spark (GB10), 128 GB unified memory each |
| Interconnect | Direct ConnectX-7 RoCE link between them, port-to-port |
| OS / driver | DGX OS (Ubuntu 24.04), kernel `6.17.0-1026-nvidia`, driver `580.173.02` |
| Disk | ~165 GB per node for the weights (each node holds the full set) |
| Docker | present by default on DGX OS |

Plus a registry-published image (~24 GB) and the weights (~165 GB). First boot to a healthy
server is **6–10 minutes** (weights load, then memory profiling and CUDA-graph capture).

## Quick start

Run on **both** nodes unless stated. `HEAD` / `WORKER` are your two hostnames; `USER` your
account. `HEAD` and `WORKER` must `ssh` each other passwordlessly — the launcher drives the
worker over SSH.

```bash
# 1. Pull the image (pinned by digest) and tag it
IMAGE="0rand/vllm_spark_dsv4-0.29-b12x@sha256:85eb91eec0e7a70c7c287ec04c9699f9604ebbd90b15b6b370104b60b605be18"
docker pull "$IMAGE"
docker tag "$IMAGE" stack029/orand:85eb91ee

# 2. Fetch the weights into the local HF cache (each node holds the full set)
export HF_HUB_DISABLE_XET=1
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 86f746b36186f0e567729a5c06a8c918caba82a9

# 3. Install and start (install runs on the head node)
git clone <this repo> && cd <this repo>
HEAD=spark1 WORKER=spark2 SSH_USER=$USER ./supervision/install.sh
sudo systemctl start stack029-vllm
journalctl -u stack029-vllm -f     # watch the gated boot
```

The installer copies the launcher, serving profile, overlays and gate battery to `~/stack029/`
on both nodes, tags the pinned image by digest on each, and installs the systemd unit on the
head. The launcher then runs preflight, starts the TP=2 pair, and runs the gate battery. When
it prints `gates: PASS`, the server is up at `http://$HEAD:8000/v1`.

Full walkthrough (including the fan-control and Secure Boot steps):
[`docs/SETUP.md`](docs/SETUP.md).

## Repository layout

```
config/          serving profile — every knob, with rationale in comments
supervision/     install script, gated launcher, systemd unit template
overlays/        vLLM source files bind-mounted read-only (the vision + parser fixes)
patches/         the upstream-shaped patch the vision overlay is generated from
verify/          gate battery and independent probes
benchmarks/      raw result JSON behind docs/BENCHMARKS.md
docs/            SETUP, CONFIG, BENCHMARKS, TROUBLESHOOTING, FAN-CONTROL
```

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — recreate the stack step by step, including Secure Boot and fan control
- [`docs/CONFIG.md`](docs/CONFIG.md) — every serving knob and why it is set that way
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — methodology, full result tables, raw data
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — failure modes we have hit and what they look like
- [`docs/FAN-CONTROL.md`](docs/FAN-CONTROL.md) — thermal management: the fan floor curve and why it is needed

## Credits

- **Model:** [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai) (MIT)
- **Image:** [`0rand/vllm_spark_dsv4-0.29-b12x`](https://hub.docker.com/) — a registry-published community build of vLLM with native DSV4 vision + DSpark
- **Launch orchestration:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — `launch-cluster.sh` drives the TP=2 pair
- **Fan control:** [thewh1teagle/sparkfan](https://github.com/thewh1teagle/sparkfan) — kernel gate + daemon, used unmodified except for a curve argument
- **EC protocol reverse engineering:** [Z841973620](https://github.com/Z841973620/dgx-spark-fan-override), [xXLegionBinFrogXx](https://github.com/xXLegionBinFrogXx/gb10-fan-control)
- **Benchmarks:** [eugr/llama-benchy](https://github.com/eugr/llama-benchy) (MIT), [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (MIT)

## License

MIT — see [LICENSE](LICENSE).
