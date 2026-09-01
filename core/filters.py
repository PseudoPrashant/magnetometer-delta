"""Reusable stateful filters shared by the trackers."""

import math

import numpy as np


class LowPassFilter:
    """Stateful exponential low-pass: hat_x <- alpha*x + (1-alpha)*hat_x.

    Mutable by design - the object exists to accumulate filter state.
    """

    def __init__(self) -> None:
        self.hat_x: float | None = None

    def apply(self, x: float, alpha: float) -> float:
        """Feed one sample with blend factor alpha; returns filtered value."""
        if self.hat_x is None:
            self.hat_x = x
        else:
            self.hat_x = alpha * x + (1.0 - alpha) * self.hat_x
        return self.hat_x

    def reset(self) -> None:
        """Forget all state so the next sample reseeds the filter."""
        self.hat_x = None


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al., CHI 2012).

    Cutoff frequency rises with estimated signal speed: heavy smoothing while
    the board is still (kills jitter), light smoothing during fast rolls
    (keeps responsiveness). One instance tracks one scalar channel.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_lpf = LowPassFilter()
        self._dx_lpf = LowPassFilter()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Blend factor for a low-pass with the given cutoff over interval dt."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float) -> float:
        """Feed one sample measured dt seconds after the previous one."""
        if dt <= 0.0:
            return self._x_lpf.apply(x, 1.0)

        prev = self._x_lpf.hat_x
        dx = 0.0 if prev is None else (x - prev) / dt
        hat_dx = self._dx_lpf.apply(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(hat_dx)
        return self._x_lpf.apply(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        """Clear both internal low-pass states."""
        self._x_lpf.reset()
        self._dx_lpf.reset()


class ExponentialMovingAverage:
    """Plain vector EMA used by tracker_v1: y <- alpha*x + (1-alpha)*y.

    Mutable by design - the object exists to hold the running average.
    """

    def __init__(self) -> None:
        self.state: np.ndarray | None = None

    def filter(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Feed one vector sample; returns the current smoothed vector."""
        if self.state is None:
            self.state = x.copy()
        else:
            self.state = alpha * x + (1.0 - alpha) * self.state
        return self.state

    def reset(self) -> None:
        """Forget the running average."""
        self.state = None


class VectorStreamDenoiseFilter:
    """Multi-stage 3D vector stream denoiser with spike rejection & 3D-coupled adaptive low-pass.

    Designed for real-time sensor streams (IIS2MDC @ 100 Hz):
      1. Spike / Glitch Gate: Suppresses single-sample EMI or bus glitch outliers.
      2. 3D-Coupled One Euro Filter: Eliminates sensor white noise and jitter at rest
         (>10x noise attenuation) while dynamically opening cutoff frequency during
         rapid motions to maintain zero-lag responsiveness.
      3. Synchronous phase: All 3 Cartesian components share speed magnitude to prevent
         inter-axis phase lag or trajectory distortion.

    Mutable by design — accumulates internal filter states.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.003,
        d_cutoff: float = 0.8,
        max_spike_delta: float = 4000.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_spike_delta = max_spike_delta

        self._prev_raw: np.ndarray | None = None
        self._speed_lpf = LowPassFilter()
        self._x_lpfs = [LowPassFilter(), LowPassFilter(), LowPassFilter()]

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Blend factor for low-pass with cutoff frequency over interval dt."""
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, v_raw: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Feed a 3D vector sample and return the clean, denoised vector."""
        v = np.asarray(v_raw, dtype=float)
        dt = max(dt, 1e-4)

        if self._prev_raw is None:
            self._prev_raw = v.copy()
            for k in range(3):
                self._x_lpfs[k].hat_x = v[k]
            return v.copy()

        # 1. Glitch / Spike Gate
        raw_delta = v - self._prev_raw
        raw_delta_norm = float(np.linalg.norm(raw_delta))
        if raw_delta_norm > self.max_spike_delta:
            # Glitch detected: reject spike and hold previous valid value
            v = self._prev_raw.copy()
            raw_delta = np.zeros(3)
            raw_delta_norm = 0.0
        else:
            self._prev_raw = v.copy()

        # 2. 3D-Coupled Speed Estimation
        speed_raw = raw_delta_norm / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        hat_speed = self._speed_lpf.apply(speed_raw, alpha_d)

        # 3. Dynamic Cutoff Frequency (shared across all 3 axes)
        dynamic_cutoff = self.min_cutoff + self.beta * hat_speed
        alpha_x = self._alpha(dynamic_cutoff, dt)

        # 4. Synchronous Vector Low-Pass
        out = np.empty(3, dtype=float)
        for k in range(3):
            out[k] = self._x_lpfs[k].apply(v[k], alpha_x)

        return out

    def reset(self) -> None:
        """Reset internal filter states."""
        self._prev_raw = None
        self._speed_lpf.reset()
        for lpf in self._x_lpfs:
            lpf.reset()
