"""kinematic_swipe.py — Stream Kinematic Energy & Rotation-Plane Swipe Engine (v7).

Algorithm based on empirical continuous stream analysis of IIS2MDC magnetometer data:
  1. Synchronous 3D-Coupled Denoising:
     Attenuates resting noise by >10x while dynamically opening cutoff frequency during fast motion.
  2. Dipole Space Unwarping:
     m = INV_A @ (B_filt - B_OFFSET)
     u = m / ||m||  (unit dipole vector on S^2 sphere)
  3. Physical Stroke Segmentation:
     Tracks angular speed omega = ||du|| / dt.
     Motion start: omega > V7_START_SPEED (1.5 rad/s).
     Motion end: silence >= V7_SILENCE_FRAMES (6 frames / 60 ms).
  4. Velocity-Weighted Momentum Accumulation:
     A_stroke = sum( (u_{k-1} x du_k) * ||du_k|| )
     Weighting by velocity amplifies deliberate forward strokes while suppressing slow return glides.
  5. Dual-Plane & Cosine Similarity Classification:
     - Vertical Plane (UP/DOWN): Rotates around X-axis.
         sgn(A_x) > 0 -> UP
         sgn(A_x) < 0 -> DOWN
     - Horizontal Plane (LEFT/RIGHT): Rotates around Y-axis.
         sgn(A_y) < 0 -> LEFT
         sgn(A_y) > 0 -> RIGHT
  6. Refractory Lockout:
     Enforces lockout (e.g. 180 ms) to eliminate deceleration bounce and return stroke triggers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    B_OFFSET,
    INV_A,
    LSB_TO_MGAUSS,
    V6_DENOISE_BETA,
    V6_DENOISE_D_CUTOFF,
    V6_DENOISE_MIN_CUTOFF,
    V6_MAX_SPIKE_DELTA,
)
from core.filters import VectorStreamDenoiseFilter

# Default empirical reference templates derived from stream recordings
V7_DEFAULT_TEMPLATES: Dict[str, np.ndarray] = {
    "UP": np.array([+0.986, -0.155, -0.055]),
    "DOWN": np.array([-0.987, +0.161, +0.031]),
    "LEFT": np.array([-0.300, -0.954, -0.020]),
    "RIGHT": np.array([-0.010, +0.994, +0.107]),
}
for _k in V7_DEFAULT_TEMPLATES:
    V7_DEFAULT_TEMPLATES[_k] /= np.linalg.norm(V7_DEFAULT_TEMPLATES[_k])


@dataclass(frozen=True)
class KinematicSwipeEvent:
    """Represents a discrete classified swipe gesture."""

    direction: str
    confidence: float
    duration_ms: float
    displacement_deg: float
    peak_speed_rad_s: float
    axis: np.ndarray
    sample_count: int


class StreamKinematicSwipeDetector:
    """Real-time stream kinematic energy & rotation-plane swipe engine."""

    def __init__(
        self,
        templates: Optional[Dict[str, np.ndarray]] = None,
        min_cutoff: float = V6_DENOISE_MIN_CUTOFF,
        beta: float = V6_DENOISE_BETA,
        d_cutoff: float = V6_DENOISE_D_CUTOFF,
        max_spike_delta: float = V6_MAX_SPIKE_DELTA,
        start_speed: float = 1.4,          # rad/s to trigger stroke start
        min_peak_speed: float = 6.0,       # rad/s peak required for valid swipe
        min_displacement_deg: float = 18.0,# degrees total angular arc
        min_samples: int = 10,             # minimum 100 ms duration
        silence_frames: int = 6,           # 60 ms stillness to mark stroke completion
        lockout_frames: int = 16,          # 160 ms refractory lockout
        confidence_threshold: float = 0.40,
    ) -> None:
        self.templates = templates or V7_DEFAULT_TEMPLATES
        self.start_speed = start_speed
        self.min_peak_speed = min_peak_speed
        self.min_displacement_deg = min_displacement_deg
        self.min_samples = min_samples
        self.silence_frames = silence_frames
        self.lockout_frames = lockout_frames
        self.confidence_threshold = confidence_threshold

        self.denoiser = VectorStreamDenoiseFilter(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            max_spike_delta=max_spike_delta,
        )

        self._prev_u: Optional[np.ndarray] = None
        self._in_stroke: bool = False
        self._stroke_u: List[np.ndarray] = []
        self._stroke_du: List[np.ndarray] = []
        self._stroke_speeds: List[float] = []
        self._silence_count: int = 0
        self._lockout_count: int = 0

        # Latest state for HUD/debug monitoring
        self.current_speed_rad_s: float = 0.0
        self.current_u: np.ndarray = np.array([0.0, 0.0, 1.0])
        self.current_b_clean: np.ndarray = np.zeros(3)

    def reset(self) -> None:
        """Reset internal filter and accumulator states."""
        self.denoiser.reset()
        self._prev_u = None
        self._in_stroke = False
        self._stroke_u.clear()
        self._stroke_du.clear()
        self._stroke_speeds.clear()
        self._silence_count = 0
        self._lockout_count = 0
        self.current_speed_rad_s = 0.0

    def feed_raw(
        self,
        b_raw_mg: np.ndarray,
        dt: float = 0.010,
    ) -> Optional[KinematicSwipeEvent]:
        """Feed a raw 3D magnetometer vector in mGauss."""
        # 1. Synchronous Denoising
        b_clean = self.denoiser.filter(b_raw_mg, dt=dt)
        self.current_b_clean = b_clean - B_OFFSET

        # 2. Geometry Unwarping to Dipole Coordinates
        m = np.dot(INV_A, self.current_b_clean)
        norm_m = float(np.linalg.norm(m))
        if norm_m < 1e-4:
            return None

        u = m / norm_m
        self.current_u = u

        if self._prev_u is None:
            self._prev_u = u
            return None

        # 3. Kinematic Differential
        du = u - self._prev_u
        self._prev_u = u
        speed = float(np.linalg.norm(du)) / max(dt, 0.001)
        self.current_speed_rad_s = speed

        # Handle refractory lockout
        if self._lockout_count > 0:
            self._lockout_count -= 1
            return None

        event: Optional[KinematicSwipeEvent] = None

        # 4. Stroke State Machine
        if speed >= self.start_speed:
            if not self._in_stroke:
                self._in_stroke = True
                self._stroke_u = [u]
                self._stroke_du = []
                self._stroke_speeds = [speed]
                self._silence_count = 0
            else:
                self._stroke_u.append(u)
                self._stroke_du.append(du)
                self._stroke_speeds.append(speed)
                self._silence_count = 0
        else:
            if self._in_stroke:
                self._silence_count += 1
                self._stroke_u.append(u)
                self._stroke_du.append(du)
                self._stroke_speeds.append(speed)

                if self._silence_count >= self.silence_frames:
                    # Stroke ended — evaluate kinematics
                    event = self._evaluate_stroke(dt)
                    self._in_stroke = False
                    self._stroke_u.clear()
                    self._stroke_du.clear()
                    self._stroke_speeds.clear()
                    self._silence_count = 0
                    if event is not None:
                        self._lockout_count = self.lockout_frames

        return event

    def _evaluate_stroke(self, dt: float) -> Optional[KinematicSwipeEvent]:
        """Evaluate accumulated stroke kinematics and classify direction."""
        n_samples = len(self._stroke_u)
        if n_samples < self.min_samples:
            return None

        peak_spd = max(self._stroke_speeds) if self._stroke_speeds else 0.0
        if peak_spd < self.min_peak_speed:
            return None

        # Compute total angular displacement
        u_start = self._stroke_u[0]
        u_end = self._stroke_u[-1]
        dot_chord = float(np.clip(np.dot(u_start, u_end), -1.0, 1.0))
        displacement_deg = math.degrees(math.acos(dot_chord))

        if displacement_deg < self.min_displacement_deg:
            return None

        # 5. Velocity-Weighted Rotation Axis Accumulation
        # Non-linear velocity weighting emphasizes intentional fast motions
        u_arr = np.array(self._stroke_u)
        du_arr = np.diff(u_arr, axis=0)
        cross_arr = np.cross(u_arr[:-1], du_arr)
        weights = np.linalg.norm(du_arr, axis=1)[:, None] ** 1.3
        accum_axis = np.sum(cross_arr * weights, axis=0)

        mag = float(np.linalg.norm(accum_axis))
        if mag < 1e-6:
            return None

        unit_axis = accum_axis / mag

        # 6. Direction Classification: Dual-Plane & Cosine Scoring
        # Check template scores
        scores: Dict[str, float] = {}
        for d_name, template in self.templates.items():
            scores[d_name] = float(np.dot(unit_axis, template))

        best_dir = max(scores, key=scores.get)
        confidence = scores[best_dir]

        # Dual-plane backup verification
        ax_x, ax_y = unit_axis[0], unit_axis[1]
        if abs(ax_x) >= 0.75 * abs(ax_y):
            plane_dir = "UP" if ax_x > 0 else "DOWN"
        else:
            plane_dir = "RIGHT" if ax_y > 0 else "LEFT"

        # If cosine match is weak, fallback to plane decision
        if confidence < self.confidence_threshold:
            if abs(ax_x) > 0.35 or abs(ax_y) > 0.35:
                best_dir = plane_dir
                confidence = max(abs(ax_x), abs(ax_y))
            else:
                best_dir = "UNKNOWN"

        duration_ms = n_samples * dt * 1000.0

        return KinematicSwipeEvent(
            direction=best_dir,
            confidence=max(0.0, confidence),
            duration_ms=duration_ms,
            displacement_deg=displacement_deg,
            peak_speed_rad_s=peak_spd,
            axis=unit_axis,
            sample_count=n_samples,
        )
