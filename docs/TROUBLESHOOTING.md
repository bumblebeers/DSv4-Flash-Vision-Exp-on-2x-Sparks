# Troubleshooting

Failure modes we have actually hit, what they look like, and what to do.

---

## The GPU holds ~650–930 MHz under load after a reboot

**Symptom.** The server starts and serves fine, but gate 4 fails:

```
gate 4: clock peaks under soak head=650 worker=929 (need >=2400)
```

Requests still complete — they are just ~2.4× slower (76 vs 135 requests in the same
130 s window). Idle `clocks.sm` reads ~650–930 MHz instead of ~2400 MHz.

**Cause.** A GB10 firmware clock-rate / power-delivery state that some reboots come up
in. It is not a software fault and nothing in the stack causes it.

**Fix.** Reboot. The healthy signature is `clocks.sm ≈ 2400 MHz` at idle and 2500+ MHz
under load, `pstate P0`, all throttle reasons `Not Active`. Confirm before blaming
anything else:

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,pstate --format=csv,noheader
nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader
```

**Do not** go hunting in the serving config for this. Work completes, so the engine is
healthy; a pinned clock has no software cause.

---

## Full-context requests are thermally marginal

**Symptom.** A ~1M-token prefill sustains the board at **89–92 °C** with both fans
already at maximum. On one occasion the node hard-stopped during such a request — no
ICMP, both RoCE links dropped, physical power cycle required. Memory was flat and
`/health` was 200 until ~20 s before death, so it was not memory exhaustion.

**Cause.** Sustained full-context prefill at maximum GPU utilization is close to this
board's thermal limit.

**Do this.**

- Run the fan curve in [`FAN-CONTROL.md`](FAN-CONTROL.md). It brings the fans to 100 %
  at 85 °C rather than the stock ~95–96 °C.
- Treat ~1M-token prompts as a deliberate operation, not routine traffic. 524K is
  comfortable; 1M is the edge.
- Watch the **board ACPI zones** (`/sys/class/thermal/thermal_zone*/temp`), not the GPU
  die. The board zones are the trip driver.

There is **no IPMI, no BMC and no remote power path** on a DGX Spark. If a node stops
answering, recovery is physically pulling power. Budget for that.

---

## Host memory runs out and the node livelocks

**Symptom.** The endpoint stops answering; SSH is dead; but the node still answers
`ping` and Tailscale, and TCP handshakes still complete. Kernel and network stack
alive, userspace never scheduled. No OOM-kill, no thermal event, no Xid.

**Cause.** GPU and host share one 121.6 GiB pool. At high `GPU_UTIL` the measured idle
`MemAvailable` can be only ~3 GiB, and sustained inference needs 2–4 GiB more. The
kernel starts swapping and never recovers.

**Recovery.** Power cycle. There is no remote path.

**Prevention.**

- Keep `GPU_UTIL` at 0.88 or below and know your idle `MemAvailable` floor.
- Note the **prefill memory high-water mark**: each new larger prompt permanently steps
  the idle floor down, because the GPU-side allocator's reserved pool grows with prompt
  length. Idle `MemAvailable` after a large request predicts the problem; the average does not.
- Swap is the trap. On unified memory swap converts a clean OOM kill into a ~30-minute
  livelock. Consider disabling it so failure is fast.

---

## Boot is slow, or appears hung

Cold boot is **6–10 minutes**: 65 weight shards at ~9 s each, then memory profiling and
CUDA graph capture (~45 s). A first boot with a cold JIT cache also compiles Triton,
CuTeDSL and TileLang kernels *during* the first requests — the engine logs
`jit_monitor … JIT compilation during inference` and warns about the latency spike.

If the container is `Up` but `/health` never returns, check the engine log rather than
waiting:

```bash
docker logs --tail 40 vllm_node
```

`Engine core initialization failed` means it will not recover on its own.

---

## `Invalid repository ID or local directory specified`

The weights snapshot vLLM resolved does not exist locally. The Hub repo is updated in
place — an unpinned `main` moves. Pin `REVISION` in the profile.

Related, from Xet: interrupted downloads leave **dangling symlinks** in the snapshot.

```bash
find ~/.cache/huggingface/hub -xtype l
find ~/.cache/huggingface/hub -path '*/blobs/*' -type l ! -exec test -e {} \; -delete
HF_HUB_DISABLE_XET=1 hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
```

---

## `NV_ERR_NO_MEMORY` in the kernel log

```
NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] …
```

**Expected at boot and during the first deep prefill** — this is first-touch backing of
the KV pool. Boots show anything from a handful to ~200 events. The gate records the
count rather than failing on it.

What matters is a **steady stream under normal load**, or a high count *after* the
server is ready. Compare against `facts.txt` in the boot record:

```
nvrm_boot_head=…  nvrm_boot_worker=…  nvrm_postready_head=…  nvrm_postready_worker=…
```

---

## A flat, featureless image is described wrongly

A single flat colour field with no structure is answered "White" regardless of its
actual colour. Any image with structure in it — including a large letter over a
coloured background — is read correctly, including the background colour.

This is an edge case, not a broken modality. The verification battery uses structured
images for its colour probe for this reason. Unresolved.

---

## Fan control

See [`FAN-CONTROL.md`](FAN-CONTROL.md) for the full set. The two that waste the most time:

- **`modprobe: Key was rejected by service`** — Secure Boot. The MOK is not enrolled, or
  the module was not signed with the enrolled key. Needs a keyboard at the next boot.
- **The tachometer does not move after a write** — some units acknowledge an override
  but keep the stock curve until a full cold power cycle: shut down, unplug, hold power
  ~10 s, plug back in, boot.

---

## Benchmarking gotchas

Two that silently invalidate results:

- **tool-eval-bench auto-detects the context window as `num_gpu_blocks × block_size`**
  (`20812 × 4 = 83,248`). On a hybrid sliding-window model that is *not* the token
  capacity — the real pool is 3.78 M and `max_model_len` is 1,048,576. Every
  context-sensitive test therefore runs at ~8 % of the intended depth unless you pass
  `--context-size 1048576`.
- **llama-benchy silently falls back to the `gpt2` tokenizer** if the model id does not
  resolve to a real HF tokenizer, which makes every throughput number wrong. Verify the
  warm-up line reports a sane delta (`Server: 105, Local: 22` on this model). It also
  reports **exit 0 with all-null measurements** on a connection failure, so check the
  data rather than the exit code.

Its default corpus (Sherlock Holmes) is only **142,813 tokens**, which caps `--depth`
well below the model's context. Supply a larger `--book-url` to test deep.
