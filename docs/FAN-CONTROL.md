# Thermal control

**Why this exists.** Under sustained full-context prefill this board reaches **96 °C**
*with both fans already at maximum*, and the EC's stock fan curve does not command 100 %
until roughly 95–96 °C. The margin between "fans finally ramp" and "thermal trip" is
thin, and a trip is a hard stop. A fan floor moves the ramp earlier.

**What is borrowed and what is ours.** The kernel module, the sysfs gate and the
userspace daemon are [thewh1teagle/sparkfan](https://github.com/thewh1teagle/sparkfan),
used unmodified. The **curve** in `SETUP.md` §5c, the measurements on this page, and
the operational notes are ours.

---

## What GB10 actually exposes

There is no PWM, no i2c and no NVML fan object. `nvidia-smi` has no fan control, and
`-pl` (power limit) is `N/A`. NVIDIA's position is explicit: *"There is no method to
control fan speed on the Spark."*

The only route is the board's embedded controller, reached over ARM FF-A / eSPI. The
EC exposes two 16-bit override slots:

| Slot | Effect |
|---|---|
| `0x119190` | **cap** — limits the target from above. Can *reduce* cooling below stock. |
| `0x119192` | **floor** — raises the target from below. Cannot undercool. |

sparkfan writes **only the floor**. It implements no cap path, so it is incapable of
commanding less cooling than the stock curve — the property that made it the right
choice over the alternatives.

The EC keeps running its own curve and simply never goes below the floor:

```
applied_pwm(fan) = max( ec_curve_pwm(temperature), floor_pwm(fan) )
```

Ramp rates are **+10 % per tick up, −1 % per tick down**. The asymmetry matters: a
raised floor takes effect quickly, while lowering it takes minutes to settle. Do not
measure a floor change before it has settled.

## Measured limits on this hardware

```
caps   mode=0  fan0=1260-9000 rpm   fan1=1890-13500 rpm
```

| | nominal max | measured max |
|---|---:|---:|
| fan0 | 9000 | **9000** |
| fan1 | 13500 | **12960** |

fan1 plateaus at 12960 — 96 % of nominal — and does not go higher however large the
floor. Use 12960 as its real ceiling when reasoning about the top gear.

**Floor → RPM is per-fan and not exactly 1:1.** A single override value is converted
through each fan's own range, so you set one number and the EC decides both fans. Two
settled measurements:

| floor written | fan0 settled | fan1 settled |
|---:|---:|---:|
| 3582 | 3690 | 4185 |
| 13500 | 9000 | 12960 |

Practical consequence: **per-fan PWM is not independently settable** through this
interface, so a percentage-based fan table can only be approximated. Ours is expressed
in RPM for exactly that reason.

## Our curve

```
20:4050, 40:6000, 70:9450, 85:13500     # temp °C : floor rpm
```

- **85 °C → 13500**: both fans to maximum. Verified live — the daemon logged
  `smoothed 85 C -> floor 13500` and fan1 went to 13500 with fan0 to 9000.
- **70 °C → 9450** and **40 °C → 6000**: the ramp below the top gear.
- **20 °C → 4050**: a floor just above stock idle (2700/4050), so an idle box is not
  made louder for no reason.

The daemon applies a **5 °C hysteresis with a 60 s hold-down** and smooths the sensor
with a 20 s EMA, so a bursty load does not flap the fans. Note this is the daemon's
global behaviour, not a per-gear hysteresis — the shape of the ramp is ours, the
step-down policy is upstream's.

It reads the **hottest board ACPI zone**, which is the sensor that matters. The GPU
die is not the trip driver on this board, and it is the sensor the stock curve
over-reacts to being late on.

## What we measured

Board ACPI zone maximum, 1 Hz–0.5 Hz sampling, on a 2-node cluster:

| Condition | Board max | Fans |
|---|---:|---|
| Idle, stock curve | 36–40 °C | 2700 / 4050 rpm |
| Idle, floor 4050 | 37–40 °C | ~4050 rpm |
| Gate soak (130 s, c8 mixed) | 51–59 °C | stepping through gears |
| **Full-context needle, 1,022K-token haystacks** | **96 °C** | **fan1 13500 (max), fan0 8640** |

The last row is the one that matters: **a full-context workload drives this board to
96 °C with both fans already at maximum.** There is no fan headroom left above it — the
only remaining levers are not running 1M-token requests, or accepting the margin.

The comparison that justifies the whole exercise:

| Same workload, 1M-token context | Fans | Outcome |
|---|---|---|
| **Stock curve** | stock, ramping late | node **hard-stopped**; physical power cycle |
| **This curve** | at maximum from 85 °C | completed, **100 % retrieval**, 96 °C peak |

Everything below the top gear is the curve buying back margin before that point.

This is also why we treat the top gear as non-negotiable. The stock curve reaching
100 % at ~95–96 °C leaves very little room when a full-context request already
sustains 92 °C at full fan.

## Operating notes

- **The floor survives a warm reboot but not a power cycle.** Enrol the module in
  `/etc/modules-load.d/` and enable `sparkfan.service` so both come back by themselves.
- **`fault` latches.** If the FF-A transport fails or a reply times out, the sysfs
  device refuses all further requests until the module is reloaded — ideally after a
  power cycle. Check `sparkfan status`; a non-zero `fault` means stop, do not retry.
- **Some units acknowledge an override but keep the stock curve** until a full cold
  power cycle (shutdown, unplug, hold power ~10 s, plug, boot). If the tachometer does
  not move after a write, that is the first thing to try.
- **Fan wear is real.** A permanent high floor is more runtime. Pick the lowest floor
  that holds the temperatures you want; the curve above only pays for airflow when the
  board is actually warm.
- **The EC's own protections stay in force** underneath: its 100 % clamp and its
  thermal trips are not disabled by anything here.
- **Writes are volatile EC SRAM.** No flash is erased or written, no firmware image is
  modified. Nothing here can brick the controller.

## Alternatives we did not choose

Three other public projects address the same EC:

- [`christopherowen/dgx-spark-fan-control`](https://github.com/christopherowen/dgx-spark-fan-control)
  — the most rigorous engineering and documentation of the four, and it also never
  writes the cap slot. We passed on it because its issue history includes a **recurring
  firmware wedge** where a raw eSPI read triggered a watchdog reboot. Not something to
  put on a serving node.
- [`Z841973620/dgx-spark-fan-override`](https://github.com/Z841973620/dgx-spark-fan-override)
  and its fork [`mathieu-lacage`](https://github.com/mathieu-lacage/dgx-spark-fan-override)
  — these document the EC protocol well (our understanding of the two override slots
  comes from their write-ups) and ship ready-made fan profiles. We did not use them
  because they drive both slots, so they can command *less* cooling than stock, and the
  fork ships no installation documentation.

All four are the same underlying trick against the same EC mailbox. If sparkfan does
not work on your unit, `christopherowen` is the reasonable second choice.
