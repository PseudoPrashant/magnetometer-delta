# Signal Pipeline Reference

The full 9-stage chain, stage by stage: what it does, why, and how to tune it.
Stages 1–6 live in `core/pipeline.py::RotationPipeline.feed()`; stages 7–9 are
tracker-side shaping.

```
[IIS2MDC @ 100 Hz]
    |
    v
1. Spike Rejection       3-tap per-axis median on the raw stream
2. Adaptive Smoothing    One Euro filter on raw B
3. Baseline Correction   B_clean = B - B_OFFSET
4. Geometry Unwarping    m = INV_A @ B_clean
5. Normalization         m_hat = m / ||m||
6. Rotation Delta        dTheta = m_prev x m_hat   (timestamped)
    |
    v  --- RotationPipeline returns (d_theta, dt); trackers take over ---
7. Velocity Ballistics   gain grows with angular speed
8. Spatial Deadzone      fractional accumulation - no lost motion     (v2)
9. Dominant Axis         keep larger of |dX|, |dY|                    (v2)
```

---

## Stage 1 — Spike Rejection

```python
med_buf.append(b_raw_mg)                      # deque(maxlen=SPIKE_MEDIAN_TAPS=3)
b_med = np.median(np.asarray(med_buf), axis=0)
```

A sliding 3-tap median per axis. A single corrupt sample (I2C hiccup, EMI
burst) becomes the median of {good, bad, good} = good. Unlike averaging, a
median removes the spike **completely** instead of smearing it across three
samples, and unlike heavier windows it adds essentially zero lag at 100 Hz.

Warm-up: the first two calls operate on 1 and 2 taps respectively — valid,
just less protected.

## Stage 2 — Adaptive Smoothing (One Euro)

Per-axis `OneEuroFilter(min_cutoff=1.2, beta=0.015, d_cutoff=1.0)` applied to
raw B (Casiez et al., CHI 2012):

```
f_c = f_min + beta * |dx_hat|          # adaptive cutoff
alpha = 1 / (1 + tau/dt), tau = 1/(2*pi*f_c)
hat_x <- alpha*x + (1-alpha)*hat_x     # standard LPF with alpha
```

- The speed estimate `dx_hat` passes through its own low-pass (`d_cutoff`)
  so cutoff changes smoothly.
- At rest → cutoff ≈ 1.2 Hz → α ≈ 0.07 per sample at 100 Hz → heavy jitter
  suppression.
- Fast roll → speed term raises the cutoff → α climbs → responsive.
- First sample seeds with nominal `DT_SEED = 0.01 s` (100 Hz).
- `dt` measured between samples via `time.perf_counter()` and clamped to
  `[DT_MIN=0.0005, DT_MAX=0.25]` s — keeps α numerically sane through stalls.
- If the gap exceeds `RESYNC_GAP = 0.5 s`, the delta is dropped this cycle
  and the previous direction reseeds (no giant jump after a stall).

Tuning: lower `OE_MIN_CUTOFF` = calmer at rest but laggier. Raise `OE_BETA`
if fast flicks feel muted.

## Stage 3 — Baseline Correction

```python
b_clean = b_filtered - B_OFFSET
```

`B_OFFSET` bundles everything that shifts the measurement when the magnet is
at rest at the pivot: hard-iron distortion from the PCB/environment plus the
ambient baseline (dominated by Earth's field — hundreds of mGauss). After
subtraction, `B_clean` is (ideally) purely the magnet dipole contribution.

This is why `B_OFFSET` is calibration output, not a knob: see
[Calibration guide](calibration-guide.md).

## Stage 4 — Geometry Unwarping

```python
m = INV_A @ b_clean
```

The sensor sits offset from the roll center (documented mount offset:
(0, 10, 5) mm), so raw axes mix. `INV_A` unwarps sensor coordinates into
*dipole space* where the rolling field traces a clean sphere centered at the
origin. Its inverse `A_FORWARD` maps fitted centers back to field units
during calibration — the pair satisfies `A_FORWARD @ INV_A ≈ I`.

If you remount the sensor or change geometry, both matrices must be rederived
together.

## Stage 5 — Normalization

```python
norm = np.linalg.norm(m)
if norm < NORM_EPS:      # 1e-4
    return None
m_unit = m / norm
```

Only direction survives; magnitude (distance-dependent) is discarded. The
epsilon guard drops degenerate samples where the dipole contribution nearly
cancels.

## Stage 6 — Rotation Delta

```python
d_theta = np.cross(m_prev, m_unit)     # then m_prev = m_unit
return d_theta, dt
```

For consecutive unit vectors, `||d_theta|| = sin(θ) ≈ θ` (radians, small-angle)
and the vector direction encodes the rotation axis by the right-hand rule.
Component mapping downstream:

- `d_theta[1]` (rotation about Y) → X deflection
- `d_theta[0]` (rotation about X) → Y deflection
- `d_theta[2]` (spin about Z / field axis) → visible only in v3 planes;
  discarded by v2's dominant axis.

Angular speed: `omega = ||d_theta|| / dt` (rad/s), fed to ballistics.

Sanity anchor from offline verification: feeding a synthetic 0.05 rad/step
rotation yields mean `||d_theta|| ≈ 0.0497` (the 0.6% deficit is One Euro lag).

## Stage 7 — Velocity Ballistics

```python
gain = GAIN_SLOW                                  if omega <= W_REF_LOW
gain = GAIN_FAST                                  if omega >= W_REF_HIGH
gain = GAIN_SLOW + (GAIN_FAST-GAIN_SLOW) * t^BALLISTICS_GAMMA
      where t = (omega - W_REF_LOW)/(W_REF_HIGH - W_REF_LOW)
```

Defaults: 700 → 4200 plot-units/rad between 0.35 and 5.0 rad/s, gamma 1.6.
Because gamma > 1, the curve is sub-linear — gain hugs `GAIN_SLOW` across the
precision zone, then ramps late:

| omega (rad/s) | gain |
|---|---|
| 0.35 | 700 |
| 1.0 | ~850 |
| 2.0 | ~1370 |
| 3.0 | ~2120 |
| 4.0 | ~3080 |
| 5.0+ | 4200 |

Raise `GAIN_FAST` for snappier flicks; raise `BALLISTICS_GAMMA` to widen the
precision zone further.

## Stage 8 — Spatial Deadzone & Fractional Accumulation (v2)

```python
frac_x += step_x
if abs(frac_x) >= SPATIAL_DEADZONE:      # 5.0 plot-units
    pos_x += frac_x; frac_x = 0.0
```

Steps below threshold are **banked**, not dropped. Slow deliberate rolls
accumulate until they cross 5 units and emit one real step; incoherent noise
oscillates around zero and never crosses coherently. Result: no drift at
rest, no dead zone under slow motion.

Lower `SPATIAL_DEADZONE` = more sensitive, more noise. Raise = calmer, but
very slow rolls emit later.

## Stage 9 — Dominant Axis (v2)

```python
if abs(step_x) >= abs(step_y): step_y = 0.0
else:                           step_x = 0.0
```

Per emitted step, only the stronger axis moves. Trajectories become
staircase-like — deliberate horizontal or vertical strokes instead of
diagonal wobble. Set `DOMINANT_AXIS = False` for continuous diagonal motion.

v3 deliberately omits stages 8–9: all three channels integrate every frame so
the Z component stays visible in its projection planes.

---

## Timing notes

- Timestamps come from `time.perf_counter()` captured right after the median
  tap is appended — consistent sampling-order semantics regardless of serial
  buffering.
- `feed()` returns `None` during warm-up, after degenerate samples, and on
  resync frames. Trackers must treat `None` as "nothing happened", never as
  an error.
