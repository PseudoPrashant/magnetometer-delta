# magnetometer-delta

Live pointer-trajectory tracking from a **rolling magnet**: an IIS2MDC
magnetometer (ESP32 streaming over serial at 100 Hz) watches a magnetized
sphere roll against the PCB. The direction of the measured dipole field acts
like a trackball encoder - rotation deltas are filtered, unwarped, gain-shaped
and integrated into a live XY trajectory (or XYZ debug views).

## Documentation

Detailed documentation lives in [`docs/`](docs/):

| Doc | Contents |
|---|---|
| [Architecture](docs/architecture.md) | System overview, module graph, design decisions, limitations |
| [Signal pipeline](docs/signal-pipeline.md) | All 9 processing stages: math, rationale, tuning effects |
| [Calibration guide](docs/calibration-guide.md) | Sphere-fit theory, step-by-step procedure, validation |
| [Configuration reference](docs/configuration.md) | Every `config.py` knob: default, effect, tuning advice |
| [Hardware setup](docs/hardware-setup.md) | Firmware serial contract, ports, mounting geometry, environment |

## Project layout

```
magnetometer-delta/
├── config.py                      # every tunable in one place
├── calibration/
│   ├── analyze_recorded_streams.py# deep kinematic stream analyzer (variance, speed, replay)
│   ├── calibrate_offset.py        # hard-iron offset calibration (sphere fit)
│   ├── collect_calibration_data.py# 4-phase guided dataset collector
│   ├── interactive_calibration.py # guided swipe capture, variation analyzer, live prediction
│   ├── record_directional_stream.py# continuous directional stream recorder (2x 20s trials)
│   └── train_axis_signatures.py   # v6 rotation-axis signature trainer (live + CSV modes)
├── core/
│   ├── axis_signature.py          # Accumulated Rotation-Axis Signature engine (v6)
│   ├── filters.py                 # LowPassFilter, OneEuroFilter, VectorStreamDenoiseFilter, EMA
│   ├── kinematic_swipe.py         # Stream Kinematic Energy & Rotation-Plane Swipe Engine (v7)
│   └── pipeline.py                # RotationPipeline, V4, V4_1, V5, StrokeGestureRecognizer
├── trackers/
│   ├── tracker_v1.py              # baseline: EMA + fixed gain + deadzone
│   ├── tracker_v2.py              # production v2: median + One Euro + ballistics
│   ├── tracker_v3.py              # debug: XYZ rotation-delta planes
│   ├── tracker_v4_1.py            # v4.1: coupled One Euro + leaky damping
│   ├── tracker_v5.py              # v5: dual-branch rotation & swipe viewer
│   ├── tracker_v5_1.py            # production v5.1: stroke recognizer + 3D leaky integration
│   ├── tracker_v6.py              # production v6: accumulated rotation-axis swipe recognizer
│   └── tracker_v7.py              # production v7: stream kinematic energy & rotation-plane engine
├── tests/
│   └── test_axis_signature.py     # unit & dataset replay test suite
└── README.md
```

## Signal chain

Stages 1-6 live in `core/pipeline.py`, `core/axis_signature.py`, or `core/kinematic_swipe.py` (`StreamKinematicSwipeDetector.feed_raw()`):

```
[IIS2MDC Hardware @ 100 Hz / 400 kHz I2C]
    |
    v
1. Spike Rejection       (adaptive glitch gate / median / noise floor)
2. Adaptive Smoothing    (3D-coupled One Euro filter on raw B)
3. Baseline Correction   (B_clean = B - B_OFFSET)
4. Geometry Unwarping    (m = INV_A @ B_clean)
5. Normalization         (m_hat = m / ||m||)
6. Rotation Delta        (dTheta exact arc angle, burst-timing rectified)
7. Velocity Ballistics   (smooth C-infinity sigmoid curve)
8. Leaky Integration     (leaky damping + noise floor gating)         [v4_1, v5_1]
9. Stroke Recognition    (stateful displacement accumulator)          [v5_1]
10. Axis Signatures      (accumulated rotation-axis dot-product match)[v6]
11. Kinematic Swiping    (energy-gated velocity-weighted momentum)    [v7]
```

## Install

Python >= 3.10, then:

```bash
pip install numpy scipy matplotlib pyserial pandas
```

Set your port (`SERIAL_PORT`) in `config.py`.

**Important:** run everything from the project root so the top-level imports
(`config`, `core`, ...) resolve:

```bash
cd magnetometer-delta
```

## Usage

### 1. Calibrate & Record Data

Tape the PCB face-down, keep metal/watches away, roll continuously for
`CALIBRATION_SECONDS` while it collects:

```bash
python -m calibration.calibrate_offset
```

Paste the printed offset into `B_OFFSET` in `config.py`.

To record high-rate continuous directional streaming data (5s Still + 15s Swipe, 2 trials per direction):

```bash
python -m calibration.record_directional_stream                               # prompts for direction and records 2x trials
```

To run deep kinematic analysis on all recorded stream datasets:

```bash
python -m calibration.analyze_recorded_streams                                # offline stream analyzer & classifier benchmark
```

To train/update rotation-axis signature templates for Tracker v6 with interactive guided capture, pattern variation analysis, and live prediction testing:

```bash
python -m calibration.interactive_calibration                                # interactive guided capture & validator (recommended)
python -m calibration.train_axis_signatures                                  # live guided serial trainer
python -m calibration.train_axis_signatures --from-csv calibration_data_*.csv # offline replay mode
```

### 2. Track

```bash
python -m trackers.tracker_v7          # production v7 (stream kinematic energy & rotation-plane engine, recommended)
python -m trackers.tracker_v6          # production v6 (accumulated rotation-axis signatures, zero-filtering)
python -m trackers.tracker_v5_1        # production v5.1 (stroke gesture recognizer + 3D)
python -m trackers.tracker_v5          # v5 dual-branch tracker
python -m trackers.tracker_v4_1        # v4.1 leaky 3D viewer
python -m trackers.tracker_v2          # legacy v2 pipeline
python -m trackers.tracker_v1          # baseline EMA tracker
```

Hotkeys in plot windows:
- **C**: Clear the trace and reset pipeline filter and gesture state.
- **T**: Print current rotation-axis template vectors (v6/v7).
- **Q**: Quit window.

## Configuration guide (`config.py`)

| Block | Knobs | Effect |
|---|---|---|
| Serial | `SERIAL_PORT`, `BAUD_RATE` | Link to the ESP32 |
| Sensor | `LSB_TO_MGAUSS` | IIS2MDC raw-to-mGauss scale |
| Geometry | `INV_A`, `A_FORWARD`, `B_OFFSET` | Mounting unwarp + calibrated baseline |
| Filtering | `EMA_ALPHA` (v1), `SPIKE_MEDIAN_TAPS`, `OE_*` (One Euro) | Jitter vs responsiveness |
| Ballistics | `GAIN_SLOW/FAST`, `W_REF_LOW/HIGH`, `BALLISTICS_GAMMA` | Precision vs flick speed curve |
| Behaviour | `ROTATION_DEADZONE`, `POINTER_GAIN` (v1); `SPATIAL_DEADZONE`, `DOMINANT_AXIS` (v2) | Motion integration feel |
| Plotting | `MAX_HISTORY`, `VIEW_LIMIT`, `TRAIL_LEN`, `LIMIT_FLOOR` | Trace length / view scaling |

## Calibration math summary

Raw field samples are unwarped through `INV_A` into dipole space, where the
rotating sphere traces a sphere. A geometric least-squares sphere fit
(`Nelder-Mead` on `(||M - c|| - R)^2`) finds the center `c`; mapping back
through `A_FORWARD` yields the hard-iron offset to subtract at runtime.

## Refactor notes

- The original flat scripts were folded in 1:1:
  `test_offset.py -> calibration/calibrate_offset.py`,
  `test_script_1.py -> trackers/tracker_v1.py`,
  `test_script_2.py -> trackers/tracker_v2.py`,
  `test_script_3.py -> trackers/tracker_v3_debug.py`.
- `B_OFFSET` is now unified: v1 previously carried an older calibration value;
  all trackers share the latest one from `config.py`. Re-run calibration if
  your setup differs.
- Run via `python -m ...` from the project root; launching the files directly
  will fail on imports.
