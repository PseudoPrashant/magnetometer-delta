# Architecture

System design, module relationships, and the reasoning behind them.

## System overview

```
[IIS2MDC magnetometer]  --I2C @400kHz-->  [ESP32]  --USB serial @115200-->  [Python host]
   raw field (LSB)                        ~100 Hz ASCII lines            numpy pipeline
                                                                         + matplotlib UI
```

The ESP32 firmware continuously samples the magnetometer and emits one line
per sample: `x,y,z` — three comma-separated numeric fields, raw LSB units,
terminated by a newline. The Python host is the tracker: it parses each line,
scales to mGauss (`LSB_TO_MGAUSS = 1.5`), runs the shared signal chain
(`core/pipeline.py`), shapes the resulting rotation deltas per-tracker, and
renders the integrated trajectory live.

A rolling magnetized sphere changes the *direction* of the measured dipole
field. That direction acts like a trackball encoder: consecutive direction
changes are converted to pointer steps.

## Module graph

```
config.py  <────────────── every module imports its knobs from here
   │
   ├── calibration/calibrate_offset.py       scipy.optimize, pyserial (lazy import)
   ├── calibration/train_axis_signatures.py  live / offline signature calibration
   │
   ├── core/pipeline.py ──► core/filters.py (OneEuroFilter ×3)
   │        ▲
   │        ├── trackers/tracker_v2.py
   │        ├── trackers/tracker_v3.py
   │        ├── trackers/tracker_v4_0.py
   │        ├── trackers/tracker_v4_1.py
   │        ├── trackers/tracker_v5.py
   │        └── trackers/tracker_v5_1.py (Stroke Recognizer + 3D Coupled)
   │
   ├── core/axis_signature.py (Accumulated Rotation-Axis Signatures Engine)
   │        ▲
   │        └── trackers/tracker_v6.py (3D Axis Sphere + Cosine Score HUD)
   │
   └── core/filters.py ◄─── trackers/tracker_v1.py (ExponentialMovingAverage)
```

- **`core/` has no matplotlib/pyserial imports** — pure signal processing,
  unit-testable offline.
- **`trackers/` own all I/O**: serial open at import time, matplotlib window,
  FuncAnimation loop. This split is why offline verification of the pipeline
  works without hardware.
- **`calibration/` is standalone** — shares only `config.py` constants with
  the trackers.

## Runtime model

Everything is single-threaded:

1. Entry point opens `SERIAL_PORT` immediately (import time) — no hardware,
   no program.
2. `FuncAnimation(update, interval=20)` fires `update()` every ~20 ms.
3. Each tick **drains the serial buffer completely** (`while ser.in_waiting:`)
   so processing tracks the data rate (~100 Hz), not the frame rate (~50 fps).
4. Per-sample work: parse → scale → `pipeline.feed()` / `recognizer.feed()` → gain/deadzone shaping
   → accumulate → update plot artists.

State lives in class-encapsulated state machines (e.g. `TrackerState` in v4_1/v5_1, `TrackerV6State` in v6)
plus the encapsulated state inside `RotationPipeline` / `RotationPipelineV5` / `AxisSignatureRecognizer`.
The `C` key calls `state.reset()`, which wipes both via `pipeline.reset()` / `recognizer.reset()`.

## Tracker variants

| | v1 | v2 (legacy prod) | v3_debug | v4 / v4_1 | v5_1 (stroke gesture + 3D) | v6 (accumulated rotation axes) |
|---|---|---|---|---|---|---|
| Spike filtering | none | 3-tap median | 3-tap median | adaptive delta gate | adaptive delta gate | noise floor delta gate |
| Filtering | plain EMA (α=0.7) | One Euro | One Euro | synchronous / 3D One Euro | 3D-coupled One Euro | **none (raw/unwarped field)** |
| Pipeline | own inline path | `RotationPipeline` | `RotationPipeline` | `RotationPipelineV4` / `V4_1` | `RotationPipelineV5` (burst-rectified) | `AxisSignatureRecognizer` |
| Kinematics | cross (small-angle) | cross (small-angle) | cross (small-angle) | exact arc-angle ($\arcsin$) | exact arc-angle ($\arcsin$) | **accumulated cross vector $\sum B_{prev} \times \Delta B$** |
| Gain | fixed 1500 | ballistic S-curve | ballistic S-curve | ballistic / smooth sigmoid | smooth $C^\infty$ sigmoid | **dot product cosine similarity** |
| Deadzone & Drift | angular (0.0035 rad), dropped | spatial (5 units), banked | none | leaky integration (v4_1) | leaky integration + noise floor | noise floor gating + silence taps |
| Gesture Recognition | none | none | none | none | stateful stroke accumulator | **3D unit-template axis matching** |
| Diagnostics HUD | no | no | console heartbeat | real-time on-canvas HUD | real-time HUD with stroke flash | **3D vector sphere + live score bars + HUD** |
| Channels | X, Y | X, Y | X, Y, Z in 3 planes | X, Y, Z in 3 planes | X, Y, Z in 3 planes | 3D axis vector $[u_x, u_y, u_z]$ |
| Purpose | historical baseline | stable v2 baseline | inspecting rotation space | production high-speed | production gesture & 3D tracker | **ultra-low compute swipe recognizer** |

v1 exists as a reference point: if v2 ever misbehaves you can diff behavior
against it. Do not extend v1 — new signal logic goes into `core/`.

## Design decisions

- **Direction-only tracking (normalization).** The magnet's distance and
  field strength vary; only orientation is stable. Normalizing to a unit
  vector makes everything downstream scale-invariant.
- **Cross-product deltas instead of absolute angles.** A dipole has 180°
  symmetry — absolute orientation is ambiguous. Consecutive-direction deltas
  sidestep that entirely, and a spin about the current field axis produces
  zero delta (correct: nothing observable changed).
- **One Euro over fixed low-pass.** A fixed filter must trade jitter at rest
  against lag while rolling. One Euro adapts cutoff to estimated speed:
  heavy smoothing when still, light when fast.
- **Fractional accumulation (stage 8).** A hard per-sample deadzone discards
  slow deliberate rolls forever. Banking sub-threshold motion in residues
  keeps slow motion working while noise never crosses the threshold
  coherently.
- **Shared pipeline, separate trackers.** Stages 1–6 were duplicated verbatim
  in two scripts; they now live once in `RotationPipeline`. Trackers keep
  only their differentiating stage-7+ behavior.
- **Absolute top-level imports.** `from config import ...` requires running
  as `python -m trackers.tracker_v2` from the project root. Chosen for
  readability; the constraint is documented everywhere it can bite.

## Known limitations

- `SERIAL_PORT` is hardcoded in `config.py` (currently `'COM7'`); Linux users
  edit it manually.
- Hardware-gated imports: CI/headless environments cannot even import the
  tracker modules (serial opens at import time). Only `core/` and
  `calibration/`'s math functions are hardware-free.
- Trajectories are visual-only (plot-units); there is no OS-pointer or HID
  output yet.
- No persistence: traces vanish on close; no session recording.
- No automated test suite. Offline checks feed synthetic rotating-dipole
  arrays through `RotationPipeline.feed()` (see git history).

## See also

- [Signal pipeline](signal-pipeline.md) — stage-by-stage math
- [Calibration guide](calibration-guide.md) — producing `B_OFFSET`
- [Configuration reference](configuration.md) — every knob explained
- [Hardware setup](hardware-setup.md) — firmware contract and environment
