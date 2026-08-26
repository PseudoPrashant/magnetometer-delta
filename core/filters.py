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
