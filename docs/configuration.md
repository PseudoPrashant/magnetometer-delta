# Configuration Reference

Every tunable lives in `config.py`. Modules never hardcode behavioral knobs;
the few module-local physics constants are listed at the bottom for
completeness. All values below are the current defaults.

## Serial

| Constant | Default | Effect |
|---|---|---|
| `SERIAL_PORT` | `'COM7'` | OS port of the ESP32. Windows: `COMx` (Device Manager). Linux/macOS: `/dev/ttyUSB0`-style path |
| `BAUD_RATE` | `115200` | Must match firmware. No reason to change unless firmware changes |

## Sensor & geometry

| Constant | Default | Effect |
|---|---|---|
| `LSB_TO_MGAUSS` | `1.5` | IIS2MDC scale factor. Fixed by hardware — change only if the sensor range config on the ESP32 changes |
| `B_OFFSET` | `[7.06, 483.78, 20.01]` mGauss | Hard-iron + ambient baseline subtracted at pipeline stage 3. **Calibration output** — see [calibration guide](calibration-guide.md) |
| `INV_A` | see config | Unwarps sensor frame → dipole space (stage 4). Encodes mount offset (0, 10, 5) mm |
| `A_FORWARD` | see config | Exact inverse of `INV_A`; used only by calibration to map the fitted center back |

## Filtering

| Constant | Default | Used by | Tuning guidance |
|---|---|---|---|
| `EMA_ALPHA` | `0.70` | v1 only | Higher = more responsive, noisier. Lower = smoother, laggier |
| `SPIKE_MEDIAN_TAPS` | `3` | pipeline stage 1 | Keep odd. Larger windows tolerate longer bursts but add latency. 3 is effectively lag-free |
| `OE_MIN_CUTOFF` | `1.2` Hz | One Euro base cutoff | Lower = calmer at rest, laggier overall. First knob to touch if rest jitter bothers you |
| `OE_BETA` | `0.015` | One Euro speed coefficient | Raise if fast rolls feel muted/laggy. Too high reintroduces jitter during motion |
| `OE_D_CUTOFF` | `1.0` Hz | One Euro internal speed LPF | Leave alone unless tuning feels binary; smooths how fast the adaptive cutoff itself moves |

## Velocity ballistics (v2/v3)

| Constant | Default | Effect |
|---|---|---|
| `GAIN_SLOW` | `700` plot-units/rad | Precision-regime gain, applied at/below `W_REF_LOW`. Scale up for a generally faster pointer |
| `GAIN_FAST` | `4200` plot-units/rad | Flick gain at/above `W_REF_HIGH`. Ratio GAIN_FAST/GAIN_SLOW defines the "throw" feel |
| `W_REF_LOW` | `0.35` rad/s | Speed where ramp-up begins. Raise = wider pure-precision zone |
| `W_REF_HIGH` | `5.0` rad/s | Speed where gain saturates. Lower = flicks max out sooner |
| `BALLISTICS_GAMMA` | `1.6` | Curve shape between refs. >1 = sub-linear (precision zone stays wide). 1.0 = linear. <1 = twitchy early ramp |

## Tracker behavior

| Constant | Default | Used by | Effect |
|---|---|---|---|
| `ROTATION_DEADZONE` | `0.0035` rad | v1 stage 7 | Angular threshold per sample; smaller motion discarded. v1 only discards — prefer v2's banking |
| `POINTER_GAIN` | `1500` plot-units/rad | v1 | Fixed gain (v1 has no ballistics) |
| `SPATIAL_DEADZONE` | `5.0` plot-units | v2 stage 8 | Banked residue must reach this before emitting a step. Noise floor vs slow-motion latency tradeoff |
| `DOMINANT_AXIS` | `True` | v2 stage 9 | Per-step axis suppression (staircase trajectories). Disable for continuous diagonal motion |

## Timing & plot

| Constant | Default | Used by | Effect |
|---|---|---|---|
| `RESYNC_GAP` | `0.5` s | pipeline | Serial stall longer than this drops the pending delta and reseeds direction instead of emitting a huge jump |
| `MAX_HISTORY` | `2000` points | v1/v2 | Trace memory cap; prevents unbounded growth and late-frame slowdowns |
| `VIEW_LIMIT` | `500` units | v1/v2 | Initial half-width of the view; autoscaling grows beyond it ×1.2 as needed |
| `TRAIL_LEN` | `600` frames | v3 | Per-plane trail length (deque-bounded) |
| `LIMIT_FLOOR` | `10.0` units | v3 | Closest auto-zoom allowed; keeps planes from zooming into noise |
| `CALIBRATION_SECONDS` | `120` s | calibration | Capture duration. Longer = better sphere coverage if your patience allows |

## Module-local physics constants (not daily knobs)

These live next to the math they stabilize and rarely need touching:

| Constant | Value | Location | Role |
|---|---|---|---|
| `DT_SEED` | `0.01` s | core/pipeline.py | Nominal interval seeding filters on first sample |
| `DT_MIN` / `DT_MAX` | `0.0005` / `0.25` s | core/pipeline.py | dt clamp bounds keeping One Euro α stable |
| `NORM_EPS` | `1e-4` | core/pipeline.py | Degenerate-sample guard at stage 5 |
| `LIMIT0` | `50.0` units | trackers/tracker_v3_debug.py | Initial per-plane half-width before autoscale takes over |

Changing any of these requires understanding the stage involved — start with
[signal-pipeline.md](signal-pipeline.md).
