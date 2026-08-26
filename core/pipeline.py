"""Shared rotation-sensing pipeline and velocity-ballistics gain shaping.

Signal chain encapsulated by RotationPipeline.feed() (stages 1-6):

    [IIS2MDC Hardware @ 100 Hz / 400 kHz I2C]
        |
        v
    1. Spike Rejection       (3-tap per-axis median on raw stream)
    2. Adaptive Smoothing    (One Euro filter on raw B)
    3. Baseline Correction   (B_clean = B - B_OFFSET)
    4. Geometry Unwarping    (m = INV_A @ B_clean)
    5. Normalization         (m_hat = m / ||m||)
    6. Rotation Delta        (dTheta = m_prev x m_hat, timestamped)

Trackers consume the emitted deltas and apply their own stage-7+ shaping
(ballistics, deadzones, integration).
"""

import time
from collections import deque

import numpy as np

from config import (
    BALLISTICS_GAMMA,
    B_OFFSET,
    GAIN_FAST,
    GAIN_SLOW,
    INV_A,
    OE_BETA,
    OE_D_CUTOFF,
    OE_MIN_CUTOFF,
    RESYNC_GAP,
    SPIKE_MEDIAN_TAPS,
    W_REF_HIGH,
    W_REF_LOW,
)
from core.filters import OneEuroFilter

DT_SEED: float = 0.01     # nominal 100 Hz interval used to seed the filters
DT_MIN: float = 0.0005    # clamp floor keeps the filters stable
DT_MAX: float = 0.25      # clamp ceiling keeps the filters stable
NORM_EPS: float = 1e-4    # below this dipole norm the sample is discarded


def ballistic_gain(omega: float) -> float:
    """Map angular speed (rad/s) to pointer gain via an S-curve."""
    if omega <= W_REF_LOW:
        return GAIN_SLOW
    if omega >= W_REF_HIGH:
        return GAIN_FAST
    t = (omega - W_REF_LOW) / (W_REF_HIGH - W_REF_LOW)
    return GAIN_SLOW + (GAIN_FAST - GAIN_SLOW) * (t ** BALLISTICS_GAMMA)


class RotationPipeline:
    """Stages 1-6: raw field samples (mGauss) in, rotation deltas out.

    Mutable state machine by design - it owns the median window, the per-axis
    One Euro filters, and the previous unit-dipole needed for the delta.
    """

    def __init__(self) -> None:
        self._med_buf: deque[np.ndarray] = deque(maxlen=SPIKE_MEDIAN_TAPS)
        self._oe_filters = [
            OneEuroFilter(OE_MIN_CUTOFF, OE_BETA, OE_D_CUTOFF) for _ in range(3)
        ]
        self._b_filtered: np.ndarray | None = None
        self._last_t: float | None = None
        self._m_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Clear every pipeline state so tracking restarts cleanly."""
        self._med_buf.clear()
        for f in self._oe_filters:
            f.reset()
        self._b_filtered = None
        self._last_t = None
        self._m_prev = None

    def feed(self, b_raw_mg: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Process one field sample; returns (d_theta, dt) or None while warming up.

        d_theta is the instantaneous rotation delta m_prev x m_hat;
        dt is the clamped seconds elapsed since the previous accepted sample.
        Returns None during filter warm-up, after degenerate samples, or when
        a serial stall (> RESYNC_GAP) forces a resync.
        """
        # ---- Stage 1: Spike Rejection (sliding median) ----
        self._med_buf.append(b_raw_mg)
        b_med = np.median(np.asarray(self._med_buf), axis=0)

        now = time.perf_counter()
        if self._b_filtered is None or self._last_t is None:
            # First sample: seed filters with the nominal 100 Hz interval.
            self._b_filtered = np.array(
                [f.filter(b_med[k], DT_SEED) for k, f in enumerate(self._oe_filters)]
            )
            self._last_t = now
            self._m_prev = None
            return None

        raw_dt = now - self._last_t
        self._last_t = now
        resync = raw_dt > RESYNC_GAP
        dt = min(max(raw_dt, DT_MIN), DT_MAX)

        # ---- Stage 2: Adaptive Smoothing (One Euro on raw B) ----
        self._b_filtered = np.array(
            [f.filter(b_med[k], dt) for k, f in enumerate(self._oe_filters)]
        )

        # ---- Stages 3+4: Baseline Correction & Geometry Unwarping ----
        m = INV_A @ (self._b_filtered - B_OFFSET)

        # ---- Stage 5: Normalization ----
        norm = np.linalg.norm(m)
        if norm < NORM_EPS:
            return None
        m_unit = m / norm

        if self._m_prev is None or resync:
            self._m_prev = m_unit.copy()
            return None

        # ---- Stage 6: Cross-Product Derivative ----
        d_theta = np.cross(self._m_prev, m_unit)
        self._m_prev = m_unit.copy()
        return d_theta, dt
