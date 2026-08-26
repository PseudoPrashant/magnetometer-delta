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

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Program ends silently after serial error line | Port wrong/busy | Check `SERIAL_PORT`; close other serial monitors |
| No output after collection | Fewer than 200 valid samples captured | Verify ~100 Hz stream (serial monitor shows 3 numbers/line); check wiring/firmware |
| Radius differs wildly between runs | Incomplete orientation coverage during roll | Re-roll, deliberately covering all axes |
| Tracker still drifts after calibration | Offset pasted into wrong place, or environment changed since capture | Re-check paste; recalibrate in final position |
