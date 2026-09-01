# Calibration Guide

How to produce a trustworthy `B_OFFSET`, and why the procedure looks the way
it does.

## Why calibrate

The magnetometer measures **everything**: the rolling magnet's dipole plus
hard-iron distortion from nearby ferromagnetic/charged material (the PCB
itself, its battery, your watch) plus Earth's ambient field. In dipole space
(stage 4 of the [pipeline](signal-pipeline.md)) all that static offset shifts
the sphere traced by the rolling field off-center. An off-center sphere makes
rotation deltas asymmetric — phantom drift and dead spots depending on
direction.

`B_OFFSET` is the correction vector subtracted at stage 3. It is *measured*,
never guessed.

## The math

1. With the offset removed, a sphere rolling through every orientation sweeps
   the field direction over the full unit sphere. In raw sensor coordinates
   this sphere is warped by geometry; `INV_A` unwarps it into dipole space.
2. There the samples lie on a sphere of radius `R` centered at residual
   center `c`. We fit both simultaneously with the geometric objective:

   ```
   loss(c, R) = Σ_i ( ||M_i - c|| - R )²
   ```

   minimized by Nelder-Mead (`scipy.optimize.minimize`). Geometric (distance)
   residuals are used rather than algebraic ones because they weight all
   sample directions equally.
3. Initial guess: component-wise min/max midpoint for `c`, `R₀ = 7320 mG`.
4. Back-transform to field units: `B_OFFSET = A_FORWARD @ c`.

The printed radius doubles as a health check — it should land in the same
ballpark as `R₀`. A wildly different radius or a fit that wanders between
runs means bad capture data, not a code bug.

## Procedure

1. **Prepare the environment** — clear the desk area of metal objects,
   watches, phones, speakers. Ferromagnetic clutter within ~30 cm skews the
   fit.
2. **Tape the PCB face-down** so the sphere can roll freely against it
   without you touching the electronics.
3. Connect the device, then run from the project root:

   ```bash
   python -m calibration.calibrate_offset
   ```

4. **Roll continuously for the full 120 s** (`CALIBRATION_SECONDS`). Vary
   speed; try to visit *all* orientations — a full sweep matters far more
   than smoothness. Uneven coverage biases the sphere fit toward the
   directions you favored.
5. When it finishes you get:

   ```
   =============================================
   ROBUST OFFSET (mG): [x.xx, y.yy, z.zz]
   RADIUS: nnnn.nn mG
   =============================================
   Paste into config.py:
   B_OFFSET = np.array([x.xx, y.yy, z.zz])
   ```

6. Paste the `B_OFFSET = ...` line into `config.py`, replacing the old one.

## Validation

- **Radius sanity**: expect thousands of mGauss (same order as `R₀ = 7320`).
- **Repeatability**: run twice; offsets should agree within a few mGauss.
- **Behavioral check**: launch `tracker_v3_debug` and roll slowly — trails in
  the three planes should be roughly symmetric around their origins, and the
  trace should sit still when the ball does.

## When to recalibrate

- Moved to a different desk/building (Earth-field baseline changes).
- Anything metallic/magnetized moved near the rig.
- After any remounting that changes the sensor-to-pivot geometry (that also
  means rederiving `INV_A`/`A_FORWARD` — see [hardware setup](hardware-setup.md)).

---

## Part 2 — Rotation-Axis Signature Calibration (Tracker v6)

While `calibrate_offset.py` finds the static ambient field baseline (`B_OFFSET`), Tracker v6 requires calibrating the **directional rotation-axis templates** (`V6_TEMPLATES`).

### Interactive Guided Workflow (`interactive_calibration.py`)

Run from the project root:

```bash
python -m calibration.interactive_calibration
```

#### Phase 1: Guided Directional Capture
1. The tool continuously streams serial data at 100 Hz in a background thread.
2. It prompts you direction-by-direction (e.g. `UP`, `DOWN`, `LEFT`, `RIGHT`).
3. For each swipe, it detects stillness, arms the trigger, records the burst, and displays the exact duration, sample count, and axis vector.
4. You can immediately **Accept (Enter)**, **Retry (R)**, or **Skip (S)** each swipe.

#### Phase 2: Pattern & Variation Analysis
The engine evaluates intra-cluster consistency and inter-direction separation:
- **Intra-Cluster Angular Spread**: Computes standard deviation of deviation angles from the centroid ($\sigma_\theta$). Target: $< 15^\circ$.
- **Consistency Score**: Mean cosine similarity within the cluster. Target: $> 95\%$.
- **Outlier Pruning**: Automatically rejects noisy return strokes or accidental bumps ($> 35^\circ$ deviation).
- **Pairwise Separation Matrix**: Computes angles and cosine similarities between all target directions. Target: Opposites $\approx 180^\circ$ (sim $\approx -1.0$), Orthogonals $\approx 90^\circ$ (sim $\approx 0.0$).

#### Phase 3: Seamless Live Prediction Validation
1. Instantly loads the calibrated templates into the live recognizer without closing the serial port.
2. As you perform test swipes, it displays live predicted directions, confidence percentages, and top score bars.
3. Prompts you to confirm accuracy to measure real-world performance live.
4. Automatically offers to update `config.py::V6_TEMPLATES` and exports a full timestamped session CSV log.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Program ends silently after serial error line | Port wrong/busy | Check `SERIAL_PORT`; close other serial monitors |
| No output after collection | Fewer than 200 valid samples captured | Verify ~100 Hz stream (serial monitor shows 3 numbers/line); check wiring/firmware |
| Radius differs wildly between runs | Incomplete orientation coverage during roll | Re-roll, deliberately covering all axes |
| High angular dispersion (>25°) during v6 calibration | Fast hand reset / return strokes captured | Use interactive mode (`interactive_calibration`) and press `R` to retry noisy swipes |
| Tracker still drifts after calibration | Offset pasted into wrong place, or environment changed since capture | Re-check paste; recalibrate in final position |
