# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-26
**Commit:** 45b27a8
**Branch:** main

## OVERVIEW

Trackball-style pointer tracker: an ESP32 streams IIS2MDC magnetometer data over serial at 100 Hz; the dipole direction of a rolling magnet is filtered, unwarped, and integrated into live XY (or XYZ debug) trajectories. Python >=3.10, numpy/scipy/matplotlib/pyserial.

## STRUCTURE

```
magnetometer-delta/
├── config.py                      # EVERY tunable: serial, geometry matrices, B_OFFSET, filter/gain/v6 knobs
├── calibration/
│   ├── analyze_recorded_streams.py# deep kinematic stream analyzer (variance, speed, replay)
│   ├── calibrate_offset.py        # sphere-fit hard-iron calibration → prints B_OFFSET for config.py
│   ├── collect_calibration_data.py# 4-phase guided dataset collector with live XY trail
│   ├── interactive_calibration.py # guided swipe capture, variation analyzer, live prediction
│   ├── record_directional_stream.py# continuous directional stream recorder (2x 20s trials)
│   └── train_axis_signatures.py   # v6 rotation-axis signature trainer (live + CSV modes)
├── core/
│   ├── axis_signature.py          # Accumulated Rotation-Axis Signatures Engine (v6)
│   ├── filters.py                 # LowPassFilter, OneEuroFilter, VectorStreamDenoiseFilter, EMA
│   ├── kinematic_swipe.py         # Stream Kinematic Energy & Rotation-Plane Swipe Engine (v7)
│   └── pipeline.py                # RotationPipeline, V4, V4_1, V5, StrokeGestureRecognizer, ballistic_gain()
├── trackers/
│   ├── tracker_v1.py              # baseline: own EMA path, fixed gain + angular deadzone
│   ├── tracker_v2.py              # legacy prod: RotationPipeline + ballistics + spatial deadzone
│   ├── tracker_v3.py              # debug viewer: XYZ accumulators in 3 projection planes
│   ├── tracker_v4_0.py            # v4: RotationPipelineV4 (vector gate + exact arc)
│   ├── tracker_v4_1.py            # v4.1: 3D-coupled filter + leaky integrator
│   ├── tracker_v5.py              # v5: Dual-branch with instantaneous swipe classification
│   ├── tracker_v5_1.py            # production v5.1: StrokeGestureRecognizer + 3D leaky integration
│   ├── tracker_v6.py              # production v6: Accumulated rotation-axis swipe recognizer
│   └── tracker_v7.py              # production v7: Stream kinematic energy & rotation-plane engine
├── tests/
│   └── test_axis_signature.py     # unit & dataset replay test suite
├── docs/                          # detailed docs: architecture, pipeline math, calibration, config ref
└── requirements.txt / .gitignore / README.md / AGENTS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change any tunable | `config.py` | Single source of truth; modules never hardcode knobs |
| Recalibrate offset | `calibration/calibrate_offset.py` | Paste printed value into `B_OFFSET`; not a hand-tuning knob |
| Record stream data | `calibration/record_directional_stream.py` | Continuous 2x 20s directional stream capture |
| Analyze stream logs | `calibration/analyze_recorded_streams.py` | Deep kinematic stream analysis & accuracy benchmarking |
| Train v6 axis templates | `calibration/train_axis_signatures.py` | Paste printed dict into `V6_TEMPLATES` in `config.py` |
| Modify signal stages 1-6 | `core/pipeline.py` | `RotationPipeline`, `V4`, `V4_1`, or `RotationPipelineV5` |
| Modify v6 axis signature | `core/axis_signature.py` | `AxisSignatureRecognizer` (zero-filtering accumulator) |
| Modify v7 kinematic swipe| `core/kinematic_swipe.py` | `StreamKinematicSwipeDetector` (velocity-weighted momentum) |
| Add a tracker variant | `trackers/`, copy v6 / v7 | Consume deltas or raw field |
| Plot/view behavior | `update()` in each tracker | FuncAnimation callback; class or module state |

## CODE MAP

Centrality measured via codegraph (callers).

| Symbol | Type | Location | Callers | Role |
|--------|------|----------|---------|------|
| `StreamKinematicSwipeDetector.feed_raw` | method | core/kinematic_swipe.py | tracker_v7, analyze_recorded_streams | Velocity-weighted momentum & dual-plane swipe recognition (v7) |
| `AxisSignatureRecognizer.feed` | method | core/axis_signature.py | tracker_v6, train_axis_signatures | Accumulated rotation-axis signature recognition (v6) |
| `StrokeGestureRecognizer.feed` | method | core/pipeline.py | tracker_v5_1 | Multi-sample stroke accumulator & swipe classifier |
| `RotationPipelineV5.feed` | method | core/pipeline.py | tracker_v5_1 | Stages 1-6 (burst-rectified + 3D-coupled + exact arc) |
| `RotationPipelineV4.feed` | method | core/pipeline.py | tracker_v4_0, tracker_v5 | Stages 1-6 (vector gate + exact arc-angle) |
| `RotationPipeline.feed` | method | core/pipeline.py | tracker_v2, tracker_v3 | Stages 1-6 (median + small-angle cross) |
| `VectorStreamDenoiseFilter` | class | core/filters.py | kinematic_swipe, axis_signature, trackers | Multi-stage 3D-coupled adaptive stream denoiser |
| `ballistic_gain_smooth` | func | core/pipeline.py | tracker_v4_1, tracker_v5_1 | Smooth C-infinity sigmoid velocity ballistics |
| `ballistic_gain` | func | core/pipeline.py | tracker_v2, v3, v4, v5 | Piecewise S-curve rad/s → plot-units/rad |
| `OneEuroFilter` | class | core/filters.py | pipeline ×3 axes | Adaptive smoothing on raw B |
| `collect_data` / `fit_robust_dipole` | funcs | calibration/calibrate_offset.py | `__main__` only | Serial capture / Nelder-Mead sphere fit |
| `update(frame)` | func | each tracker | FuncAnimation | Serial drain → integrate → redraw |

## CONVENTIONS

- Absolute top-level imports (`from config import ...`) → everything runs via `python -m <pkg>.<mod>` **from the project root only**.
- Console protocol: `[+]` ok, `[-]` error, `[*]` action, `[.]` heartbeat. Keep prefixes.
- Serial open pattern everywhere: `try/except serial.SerialException` → print `[-]` → `SystemExit(1)`.
- Mutable-by-design classes (filters, RotationPipeline, AxisSignatureRecognizer, StreamKinematicSwipeDetector) carry a docstring saying why mutation is required.
- Stage-numbered comments (`# ---- Stage N ----`) map to the chain documented in core/pipeline.py docstring + README.

## ANTI-PATTERNS (THIS PROJECT)

- NEVER launch files directly (`python trackers/tracker_v7.py`) — ImportError on `config`/`core`.
- NEVER hand-edit `B_OFFSET` without re-running calibration — it is sphere-fit output, not a tuning knob.
- NEVER duplicate signal-chain logic into trackers — extend `core/pipeline.py`, `core/axis_signature.py`, or `core/kinematic_swipe.py`.
- NEVER poke filter internals (`f._x_lpf.hat_x = None`) — use `.reset()`.
- Do NOT "clean up" `ani = FuncAnimation(...)` as unused var — it keeps the animation alive (`# noqa: F841` intentional).

## UNIQUE STYLES

- Banner section comments (`# ======` blocks) inherited from original scripts — preserve structure.
- Physics constants local to their module are allowed (`DT_SEED/DT_MIN/DT_MAX/NORM_EPS` in pipeline, `LIMIT0` in v3); behavioral knobs go to config.py.

## COMMANDS

```bash
pip install -r requirements.txt          # numpy scipy matplotlib pyserial pandas
python -m unittest discover -s tests -p "test_*.py" # offline test suite (runs without hardware)
python -m calibration.calibrate_offset   # hardware-gated, ~CALIBRATION_SECONDS roll
python -m calibration.collect_calibration_data # 4-phase guided dataset collector
python -m calibration.interactive_calibration  # interactive guided capture, variation analyzer, live prediction
python -m calibration.record_directional_stream # continuous 2-trial directional stream recorder
python -m calibration.analyze_recorded_streams # deep kinematic stream analysis & accuracy benchmarking
python -m calibration.train_axis_signatures    # live / offline rotation-axis template trainer
python -m trackers.tracker_v7            # production tracker v7 (stream kinematic energy & rotation-plane engine, recommended)
python -m trackers.tracker_v6            # production tracker v6 (accumulated rotation axes)
python -m trackers.tracker_v5_1          # production tracker v5.1 (stroke recognition + 3D)
python -m trackers.tracker_v5            # dual-branch v5 tracker
python -m trackers.tracker_v4_1          # leaky 3D viewer v4.1
python -m trackers.tracker_v2            # legacy v2 tracker
python -m trackers.tracker_v1            # baseline EMA tracker
```

No test suite. Offline sanity check possible by feeding synthetic rotating-dipole arrays into `RotationPipeline.feed()` (see git history of this session's verification).

## NOTES

- **Hardware-gated imports**: every entry point opens `SERIAL_PORT` at import time and exits without the device — CI/headless cannot even import tracker modules.
- `SERIAL_PORT = 'COM7'` hardcoded in config.py; Linux would be `/dev/ttyUSB0`.
- tracker_v1 does NOT use RotationPipeline (historical baseline kept intentionally simple).
- `.omo/` is opencode agent session state — never commit, already ignored.
- Repo lives under OneDrive; `__pycache__` churn is expected and ignored.
