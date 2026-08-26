# PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-26
**Commit:** 45b27a8
**Branch:** main

## OVERVIEW

Trackball-style pointer tracker: an ESP32 streams IIS2MDC magnetometer data over serial at 100 Hz; the dipole direction of a rolling magnet is filtered, unwarped, and integrated into live XY (or XYZ debug) trajectories. Python >=3.10, numpy/scipy/matplotlib/pyserial.

## STRUCTURE

```
magnetometer-delta/
├── config.py                      # EVERY tunable: serial, geometry matrices, B_OFFSET, filter/gain knobs
├── calibration/
│   └── calibrate_offset.py        # sphere-fit hard-iron calibration → prints B_OFFSET for config.py
├── core/
│   ├── filters.py                 # LowPassFilter, OneEuroFilter, ExponentialMovingAverage
│   └── pipeline.py                # RotationPipeline = signal stages 1-6 + ballistic_gain()
├── trackers/
│   ├── tracker_v1.py              # baseline: own EMA path, fixed gain + angular deadzone
│   ├── tracker_v2.py              # production: RotationPipeline + ballistics + spatial deadzone + dominant axis
│   └── tracker_v3_debug.py        # debug viewer: XYZ accumulators in 3 projection planes
├── docs/                          # detailed docs: architecture, pipeline math, calibration, config ref
└── requirements.txt / .gitignore / README.md / AGENTS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change any tunable | `config.py` | Single source of truth; modules never hardcode knobs |
| Recalibrate offset | `calibration/calibrate_offset.py` | Paste printed value into `B_OFFSET`; not a hand-tuning knob |
| Modify signal stages 1-6 | `core/pipeline.py::RotationPipeline.feed` | Shared by v2/v3 — v1 bypasses it |
| Add a tracker variant | `trackers/`, copy v2 pattern | Consume `(d_theta, dt)` from `pipeline.feed()` |
| Plot/view behavior | `update()` in each tracker | FuncAnimation callback; module-level global state |

## CODE MAP

Centrality measured via codegraph (callers). **No automated tests exist anywhere.**

| Symbol | Type | Location | Callers | Role |
|--------|------|----------|---------|------|
| `RotationPipeline.feed` | method | core/pipeline.py | tracker_v2, tracker_v3_debug | Stages 1-6: raw mGauss → `(d_theta, dt)` or None |
| `RotationPipeline.reset` | method | core/pipeline.py | both trackers ('C' key) | Full state wipe |
| `ballistic_gain` | func | core/pipeline.py | tracker_v2 ×1, tracker_v3_debug ×1 | S-curve rad/s → plot-units/rad |
| `OneEuroFilter` | class | core/filters.py | pipeline ×3 axes | Adaptive smoothing on raw B |
| `ExponentialMovingAverage` | class | core/filters.py | tracker_v1 only | Plain vector EMA (v1 path) |
| `collect_data` / `fit_robust_dipole` | funcs | calibration/calibrate_offset.py | `__main__` only | Serial capture / Nelder-Mead sphere fit |
| `update(frame)` | func | each tracker | FuncAnimation | Serial drain → integrate → redraw |

## CONVENTIONS

- Absolute top-level imports (`from config import ...`) → everything runs via `python -m <pkg>.<mod>` **from the project root only**.
- Console protocol: `[+]` ok, `[-]` error, `[*]` action, `[.]` heartbeat. Keep prefixes.
- Serial open pattern everywhere: `try/except serial.SerialException` → print `[-]` → `SystemExit(1)`.
- Mutable-by-design classes (filters, RotationPipeline) carry a docstring saying why mutation is required.
- Stage-numbered comments (`# ---- Stage N ----`) map to the 9-stage chain documented in core/pipeline.py docstring + README — keep all three in sync when renumbering.

## ANTI-PATTERNS (THIS PROJECT)

- NEVER launch files directly (`python trackers/tracker_v2.py`) — ImportError on `config`/`core`.
- NEVER hand-edit `B_OFFSET` without re-running calibration — it is sphere-fit output, not a tuning knob.
- NEVER duplicate signal-chain logic into trackers — extend `core/pipeline.py`.
- NEVER poke filter internals (`f._x_lpf.hat_x = None`) — use `.reset()`.
- Do NOT "clean up" `ani = FuncAnimation(...)` as unused var — it keeps the animation alive (`# noqa: F841` intentional).

## UNIQUE STYLES

- Banner section comments (`# ======` blocks) inherited from original scripts — preserve structure.
- Physics constants local to their module are allowed (`DT_SEED/DT_MIN/DT_MAX/NORM_EPS` in pipeline, `LIMIT0` in v3); behavioral knobs go to config.py.

## COMMANDS

```bash
pip install -r requirements.txt          # numpy scipy matplotlib pyserial
python -m calibration.calibrate_offset   # hardware-gated, ~CALIBRATION_SECONDS roll
python -m trackers.tracker_v2            # production tracker (recommended)
python -m trackers.tracker_v1            # baseline EMA tracker
python -m trackers.tracker_v3_debug      # XYZ debug planes
```

No test suite. Offline sanity check possible by feeding synthetic rotating-dipole arrays into `RotationPipeline.feed()` (see git history of this session's verification).

## NOTES

- **Hardware-gated imports**: every entry point opens `SERIAL_PORT` at import time and exits without the device — CI/headless cannot even import tracker modules.
- `SERIAL_PORT = 'COM7'` hardcoded in config.py; Linux would be `/dev/ttyUSB0`.
- tracker_v1 does NOT use RotationPipeline (historical baseline kept intentionally simple).
- `.omo/` is opencode agent session state — never commit, already ignored.
- Repo lives under OneDrive; `__pycache__` churn is expected and ignored.
