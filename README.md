# magnetometer-delta

Live pointer-trajectory tracking from a **rolling magnet**: an IIS2MDC
magnetometer (ESP32 streaming over serial at 100 Hz) watches a magnetized
sphere roll against the PCB. The direction of the measured dipole field acts
like a trackball encoder - rotation deltas are filtered, unwarped, gain-shaped
and integrated into a live XY trajectory (or XYZ debug views).

## Project layout

```
magnetometer-delta/
├── config.py                      # every tunable in one place
├── calibration/
│   └── calibrate_offset.py        # hard-iron offset calibration (sphere fit)
├── core/
│   ├── filters.py                 # LowPassFilter, OneEuroFilter, EMA
│   └── pipeline.py                # RotationPipeline (stages 1-6) + ballistic_gain
├── trackers/
│   ├── tracker_v1.py              # baseline: EMA + fixed gain + deadzone
│   ├── tracker_v2.py              # production: full pipeline + ballistics
│   └── tracker_v3_debug.py        # debug: XYZ rotation-delta planes
└── README.md
```

## Signal chain

Stages 1-6 live in `core/pipeline.py` (`RotationPipeline.feed()`), shared by
v2/v3. Each tracker adds its own stage-7+ shaping:

```
[IIS2MDC Hardware @ 100 Hz / 400 kHz I2C]
    |
    v
1. Spike Rejection       (3-tap per-axis median on raw stream)
2. Adaptive Smoothing    (One Euro filter on raw B)
3. Baseline Correction   (B_clean = B - B_OFFSET)
4. Geometry Unwarping    (m = INV_A @ B_clean)
5. Normalization         (m_hat = m / ||m||)
6. Rotation Delta        (dTheta = m_prev x m_hat, timestamped)
7. Velocity Ballistics   (gain grows with angular speed)
8. Spatial Deadzone      (fractional accumulation - no lost motion)   [v2 only]
9. Dominant Axis         (keep larger of |dX|,|dY| per frame)          [v2 only]
```

## Install

Python >= 3.10, then:

```bash
pip install numpy scipy matplotlib pyserial
```

Set your port (`SERIAL_PORT`) in `config.py`.

**Important:** run everything from the project root so the top-level imports
(`config`, `core`, ...) resolve:

```bash
cd magnetometer-delta
```

## Usage

### 1. Calibrate (once per physical setup)

Tape the PCB face-down, keep metal/watches away, roll continuously for
`CALIBRATION_SECONDS` while it collects:

```bash
python -m calibration.calibrate_offset
```

Paste the printed offset into `B_OFFSET` in `config.py`.

### 2. Track

```bash
python -m trackers.tracker_v1          # baseline EMA tracker
python -m trackers.tracker_v2          # production pipeline (recommended)
python -m trackers.tracker_v3_debug    # debug viewer: XY / YZ / ZX planes
```

Press **C** in any plot window to clear the trace and filter state.

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
