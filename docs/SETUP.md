# Setup — recreate the stack

Two DGX Sparks, one serving a 1M-context MoE at TP=2. Everything below is run on
**both** nodes unless it says otherwise.

Throughout, substitute:

| Placeholder | Meaning | Example |
|---|---|---|
| `HEAD` | head node (rank 0) | `spark1` |
| `WORKER` | worker node (rank 1) | `spark2` |
| `SSH_USER` | your account on both nodes | `$USER` |

`HEAD` and `WORKER` must be able to `ssh` each other **passwordlessly** — the
launcher drives the worker over SSH.

---

## 1. Image

The image is registry-published and pinned by digest. Tags can move; the digest cannot.

```bash
IMAGE="0rand/vllm_spark_dsv4-0.29-b12x@sha256:85eb91eec0e7a70c7c287ec04c9699f9604ebbd90b15b6b370104b60b605be18"
docker pull "$IMAGE"
docker tag "$IMAGE" stack029/orand:85eb91ee
```

Verify both nodes agree — the launcher refuses to start if they do not:

```bash
docker image inspect --format '{{.Id}}' stack029/orand:85eb91ee
# sha256:deaff181941b58f68d74c17ff9a5b6efbcd75400b0739e629993594e5df9f13e
```

This image carries vLLM `0.28.1rc1.dev475+g6fbb00b18` with native DSV4 vision and
DSpark speculative decoding, FlashInfer 0.6.18, and B12X kernels.

## 2. Weights

~165 GB. Each node needs the **full** set — tensor parallelism shards memory, not disk.

```bash
export HF_HUB_DISABLE_XET=1
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --revision 86f746b36186f0e567729a5c06a8c918caba82a9
```

Download on `HEAD`, then copy to `WORKER` over the direct link (~25 GB/s, so minutes):

```bash
rsync -a --info=progress2 ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp \
  "$WORKER":.cache/huggingface/hub/
```

**Pin the revision.** The Hub repo is updated in place; if `refs/main` moves, the
snapshot vLLM resolves will not exist locally and the boot fails with
`Invalid repository ID or local directory specified`.

`HF_HUB_DISABLE_XET=1` matters: Xet writes some files as symlinks through a staging
directory, and if that directory is removed or the download is interrupted the
symlinks dangle. Detect and repair:

```bash
find ~/.cache/huggingface/hub -xtype l                      # dangling entries
find ~/.cache/huggingface/hub -path '*/blobs/*' -type l ! -exec test -e {} \; -delete
```

## 3. Install

From a clone of this repo, on `HEAD`:

```bash
HEAD=spark1 WORKER=spark2 SSH_USER=$USER ./supervision/install.sh
```

That copies the launcher, serving profile, overlay sources and gate battery to
`~/stack029/` on both nodes, tags the pinned image by digest on each, verifies the
image ID matches the expected one, and installs the systemd unit on `HEAD` (rendering
your username and home directory into it). It does **not** start anything.

Confirm the wiring before the first boot:

```bash
ssh "$HEAD" 'ls ~/stack029/{stack029-launch.sh,profiles,overlays,gates}'
```

## 4. Serve

```bash
ssh "$HEAD" 'sudo systemctl start stack029-vllm && journalctl -u stack029-vllm -f'
```

Cold start is **6–10 minutes**: 65 weight shards, then memory profiling and CUDA
graph capture. When the battery passes you will see `gates: PASS` and the launcher
enters its monitor loop.

```bash
curl -s localhost:8000/v1/models | jq -r '.data[].id'
curl -s localhost:8000/health -o /dev/null -w '%{http_code}\n'
```

Expected KV pool is around **3.78M tokens** (3.60× concurrency at 1M context).
It varies a few percent boot to boot — that is the CUDA-graph memory profiler, not a fault.

### Switching models

`stack029-vllm.service` declares `Conflicts=` against the other serving units, so
switching is one command either way:

```bash
sudo systemctl start stack029-vllm     # this stack up, others down
```

---

## 5. Thermal control (recommended)

**Do this.** Under sustained full-context prefill the board reaches the low 90s °C
even with fans at maximum, and the EC's stock fan curve does not reach 100 % until
~95–96 °C — late enough that the margin to a thermal trip is thin. See
[`FAN-CONTROL.md`](FAN-CONTROL.md) for the measurements behind that.

GB10 exposes no `hwmon` PWM, no i2c and no NVML fan object to userspace. The only
route is a kernel module that asks the board's embedded controller to raise its fan
RPM *floor*. We use [thewh1teagle/sparkfan](https://github.com/thewh1teagle/sparkfan)
— floor-only (it cannot command less cooling than stock), tested, and it ships a
temperature-driven daemon.

### 5a. Secure Boot

DGX OS boots with Secure Boot **enabled**, which rejects unsigned modules. Enrol a
Machine Owner Key once. This needs **a keyboard and monitor** at the next boot —
there is no remote path, by design.

```bash
sudo apt install -y mokutil openssl
cd ~ && mkdir -p sparkfan-mok && cd sparkfan-mok
openssl req -new -x509 -newkey rsa:2048 -nodes -days 36500 \
  -subj "/CN=sparkfan MOK/" -keyout MOK.priv -outform DER -out MOK.der
chmod 600 MOK.priv
sudo mokutil --import MOK.der     # choose a one-time password
sudo reboot
```

At boot: blue **MOK management** screen → `Enroll MOK` → `Continue` → `Yes` →
one-time password → reboot. The MOK screen appears only on the **first** boot after
the import; if you miss it, it times out harmlessly and you re-import.

Verify:

```bash
sudo mokutil --list-enrolled | grep -c sparkfan    # >= 1
```

### 5b. Build and install the module

```bash
git clone https://github.com/thewh1teagle/sparkfan && cd sparkfan
make -C kmod && make -C kmod test          # expect: all tests passed
K=$(uname -r)
sudo /lib/modules/$K/build/scripts/sign-file sha512 \
     ~/sparkfan-mok/MOK.priv ~/sparkfan-mok/MOK.der build/kmod/sparkfan.ko
sudo install -m 644 build/kmod/sparkfan.ko /lib/modules/$K/extra/
sudo depmod -a
echo sparkfan | sudo tee /etc/modules-load.d/sparkfan.conf
```

The userspace daemon is Rust (std-only, no dependencies):

```bash
cargo build --release --manifest-path cli/Cargo.toml --target-dir build/cli
sudo install -m 755 build/cli/release/sparkfan /usr/local/bin/
sudo install -m 644 sparkfan.service /etc/systemd/system/
```

If your `cargo` predates lockfile v4, delete `cli/Cargo.lock` first — the crate has
no dependencies, so it regenerates cleanly.

### 5c. The curve

We override the daemon's default curve. At **85 °C every fan goes to maximum**; below
that it ramps down. Values are RPM floors, chosen so both fans reach their real maxima
at the top gear (fan0 tops out at 9000, fan1 at ~12960 despite a nominal 13500).

```bash
sudo mkdir -p /etc/systemd/system/sparkfan.service.d
sudo tee /etc/systemd/system/sparkfan.service.d/curve.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/local/bin/sparkfan daemon 20:4050,40:6000,70:9450,85:13500
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now sparkfan
```

Confirm the tachometer actually moved, rather than trusting the write:

```bash
sparkfan status
# floor 4050   rpm fan0 <n> fan1 <n>
```

Stock idle is ~2700/4050 rpm; with the floor applied at a warm board you should see
both fans well above that. To hand control back to the EC: `sparkfan auto`.

## 6. Verify

```bash
cd ~/stack029 && ./stack029-launch.sh gates profiles/R-baseline.env
```

Runs the full 5-gate battery against the live server. Independent probes:

```bash
python3 verify/gate-probes.py    --base http://$HEAD:8000/v1 --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
python3 verify/vision-probes.py  --base http://$HEAD:8000/v1 --model deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
```

---

## 7. Point an agent at it

The endpoint is OpenAI-compatible at `http://$HEAD:8000/v1`, serving both
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` and `deepseek-v4-flash`.

```jsonc
{
  "baseURL": "http://spark1:8000/v1",
  "apiKey":  "EMPTY",                    // any non-empty string
  "model":   "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
  "contextWindow": 1048576,
  "input": ["text", "image"]
}
```

Two things that bite:

- **Thinking is toggled per request** via `chat_template_kwargs.enable_thinking`.
  Set a client timeout of **≥ 600 s** for reasoning workloads — the default 120 s
  truncates long thinking turns.
- **`text` and `image` are the only modalities most harnesses can declare.** The
  server also accepts video and audio over the raw API; check your harness's model
  schema before promising them.
