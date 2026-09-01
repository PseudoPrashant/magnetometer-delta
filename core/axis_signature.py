"""axis_signature.py — Accumulated Rotation-Axis Signatures Engine.

Core implementation of the Accumulated Rotation-Axis Signature algorithm:
  - Extracts the instantaneous vector rotation axis (B_prev x dB) per motion sample.
  - Accumulates the rotation axis across the active swipe window:
        A_accum = sum(B_prev x dB)
  - Normalizes the accumulated vector into a unit signature:
        u_live = normalize(A_accum)
  - Evaluates cosine similarity (dot product) against stored direction templates:
        score[d] = u_live . u_template[d]
  - Classifies swipe direction with confidence thresholding and tiebreaker support.
  - Computes pattern variation and cluster dispersion analytics.

Zero filtering. Zero scalar angle integration. Zero trajectory buffer storage.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    B_OFFSET,
    INV_A,
    V6_AXIS_MIN_MAGNITUDE,
    V6_CONFIDENCE_THRESHOLD,
    V6_DENOISE_BETA,
    V6_DENOISE_D_CUTOFF,
    V6_DENOISE_MIN_CUTOFF,
    V6_DIRECTIONS,
    V6_ENABLE_DENOISING,
    V6_MAX_SAMPLES,
    V6_MAX_SPIKE_DELTA,
    V6_MIN_SAMPLES,
    V6_NOISE_FLOOR,
    V6_SILENCE_TAPS,
    V6_SWIPE_START_THRESH,
    V6_TEMPLATES,
    V6_USE_UNWARPED,
)
from core.filters import VectorStreamDenoiseFilter


@dataclass
class AxisSwipeEvent:
    """Represents a recognized swipe gesture event emitted by AxisSignatureRecognizer."""

    direction: str
    confidence: float
    scores: Dict[str, float]
    axis_unit: np.ndarray
    axis_magnitude: float
    sample_count: int
    duration_sec: float
    dominant_axis: str
    first_sign: int
    timestamp: float = field(default_factory=time.perf_counter)

    def __repr__(self) -> str:
        u_str = f"[{self.axis_unit[0]:.2f}, {self.axis_unit[1]:.2f}, {self.axis_unit[2]:.2f}]"
        return (
            f"AxisSwipeEvent(dir={self.direction:<5s}, conf={self.confidence:.2f}, "
            f"axis={u_str}, n_samples={self.sample_count}, dur={self.duration_sec*1000:.1f}ms)"
        )


@dataclass
class ClusterMetrics:
    """Pattern variation and consistency metrics for a single direction cluster."""

    direction: str
    centroid: np.ndarray
    angular_spread_deg: float
    max_deviation_deg: float
    consistency_score: float
    sample_count: int
    individual_axes: List[np.ndarray] = field(default_factory=list)

    def summary_line(self) -> str:
        c = self.centroid
        c_str = f"[{c[0]:+5.2f}, {c[1]:+5.2f}, {c[2]:+5.2f}]"
        return (
            f"{self.direction:<8s}: centroid={c_str} | "
            f"spread={self.angular_spread_deg:4.1f} deg | max_dev={self.max_deviation_deg:4.1f} deg | "
            f"consistency={self.consistency_score*100:5.1f}% (n={self.sample_count})"
        )


class AxisSignatureRecognizer:
    """State machine for real-time Accumulated Rotation-Axis Signature recognition.

    Features:
      1. Multi-Stage Stream Denoising: Synchronous 3D-coupled adaptive filtering
         removes sensor white noise and glitches while retaining rapid swipe transients.
      2. Cross-Product Axis Accumulation: Calculates rotational axis (B_prev x dB).
      3. Directional Coherence & Debouncing: Verifies physical motion before state trigger.
      4. True Physical Timing: Reflects accurate swipe durations across queued streams.

    Mutable state machine by design — maintains sliding baseline, swipe state,
    filter buffers, and running cross-product vector accumulators.
    """

    def __init__(
        self,
        templates: Optional[Dict[str, np.ndarray]] = None,
        directions: Optional[List[str]] = None,
        noise_floor: float = V6_NOISE_FLOOR,
        swipe_start_thresh: float = V6_SWIPE_START_THRESH,
        silence_taps: int = V6_SILENCE_TAPS,
        min_samples: int = V6_MIN_SAMPLES,
        max_samples: int = V6_MAX_SAMPLES,
        confidence_threshold: float = V6_CONFIDENCE_THRESHOLD,
        axis_min_magnitude: float = V6_AXIS_MIN_MAGNITUDE,
        use_unwarped: bool = V6_USE_UNWARPED,
        enable_denoising: bool = V6_ENABLE_DENOISING,
        early_window_ratio: Optional[float] = None,
    ) -> None:
        self.directions = list(directions or V6_DIRECTIONS)
        self.noise_floor = noise_floor
        self.swipe_start_thresh = swipe_start_thresh
        self.silence_taps = silence_taps
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.confidence_threshold = confidence_threshold
        self.axis_min_magnitude = axis_min_magnitude
        self.use_unwarped = use_unwarped
        self.enable_denoising = enable_denoising
        self.early_window_ratio = early_window_ratio

        # Stream Denoiser
        self._denoiser: Optional[VectorStreamDenoiseFilter] = None
        if self.enable_denoising:
            self._denoiser = VectorStreamDenoiseFilter(
                min_cutoff=V6_DENOISE_MIN_CUTOFF,
                beta=V6_DENOISE_BETA,
                d_cutoff=V6_DENOISE_D_CUTOFF,
                max_spike_delta=V6_MAX_SPIKE_DELTA,
            )

        # Load and normalize reference templates
        self.templates: Dict[str, np.ndarray] = {}
        raw_templates = templates or V6_TEMPLATES
        for name, vec in raw_templates.items():
            v = np.asarray(vec, dtype=float)
            norm = float(np.linalg.norm(v))
            self.templates[name] = v / norm if norm > 1e-9 else v.copy()

        # State machine variables
        self.state: str = "IDLE"  # "IDLE" or "ACTIVE"
        self._baseline: Optional[np.ndarray] = None
        self._prev_vec: Optional[np.ndarray] = None
        self._axis_accum: np.ndarray = np.zeros(3, dtype=float)
        self._sample_count: int = 0
        self._quiet_count: int = 0
        self._swipe_start_time: float = 0.0
        self._first_step_sign: int = 0
        self._first_step_axis: int = 0
        self._last_processed_vec: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal state machine and stream denoiser to IDLE."""
        self.state = "IDLE"
        self._baseline = None
        self._prev_vec = None
        self._axis_accum[:] = 0.0
        self._sample_count = 0
        self._quiet_count = 0
        self._swipe_start_time = 0.0
        self._first_step_sign = 0
        self._first_step_axis = 0
        self._last_processed_vec = None
        if self._denoiser:
            self._denoiser.reset()

    def set_templates(self, templates: Dict[str, np.ndarray]) -> None:
        """Update active reference templates."""
        self.templates.clear()
        for name, vec in templates.items():
            v = np.asarray(vec, dtype=float)
            norm = float(np.linalg.norm(v))
            self.templates[name] = v / norm if norm > 1e-9 else v.copy()

    def transform_raw(self, b_raw_mg: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Apply stream denoising and convert magnetometer sample (mGauss) to analysis space.

        If enable_denoising is True: b_clean_raw = denoiser.filter(b_raw_mg)
        If use_unwarped is True: m = INV_A @ (b_clean_raw - B_OFFSET)
        If use_unwarped is False: B_clean = b_clean_raw - B_OFFSET
        """
        raw = np.asarray(b_raw_mg, dtype=float)
        if self._denoiser is not None:
            raw = self._denoiser.filter(raw, dt=dt)

        b_clean = raw - B_OFFSET
        if self.use_unwarped:
            return np.dot(INV_A, b_clean)
        return b_clean

    def feed(
        self,
        b_raw_mg: np.ndarray,
        dt: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Optional[AxisSwipeEvent]:
        """Feed a single magnetometer sample.

        Returns an AxisSwipeEvent when a valid swipe completes, or None otherwise.
        """
        step_dt = dt if dt is not None and dt > 0 else 0.010
        current_time = now if now is not None else time.perf_counter()
        v = self.transform_raw(b_raw_mg, dt=step_dt)
        self._last_processed_vec = v.copy()

        if self._baseline is None:
            self._baseline = v.copy()
            self._prev_vec = v.copy()
            return None

        # Step delta
        assert self._prev_vec is not None
        dv = v - self._prev_vec
        dv_mag = float(np.linalg.norm(dv))
        dist_from_baseline = float(np.linalg.norm(v - self._baseline))

        event: Optional[AxisSwipeEvent] = None

        if self.state == "IDLE":
            # Adaptive sliding baseline tracking when ball is at rest
            if dv_mag < self.noise_floor and dist_from_baseline < self.swipe_start_thresh:
                self._baseline = 0.96 * self._baseline + 0.04 * v
                self._prev_vec = v.copy()
            elif dist_from_baseline > self.swipe_start_thresh and dv_mag > self.noise_floor:
                # Swipe initiation detected: require both baseline distance & step movement
                self.state = "ACTIVE"
                self._axis_accum[:] = 0.0
                self._sample_count = 0
                self._quiet_count = 0
                self._swipe_start_time = current_time
                self._first_step_sign = 0
                self._first_step_axis = 0

                # Record first motion delta
                axis = np.cross(self._prev_vec, dv)
                self._axis_accum += axis
                self._sample_count = 1
                self._record_first_step(dv)
                self._prev_vec = v.copy()

        elif self.state == "ACTIVE":
            if dv_mag > self.noise_floor:
                # Accumulate rotation axis: axis = B_prev x dB
                axis = np.cross(self._prev_vec, dv)
                self._axis_accum += axis
                self._sample_count += 1
                self._quiet_count = 0

                if self._first_step_sign == 0:
                    self._record_first_step(dv)

                self._prev_vec = v.copy()
            else:
                self._quiet_count += 1

            # Check for swipe completion: stillness detected or max window exceeded
            if self._quiet_count >= self.silence_taps or self._sample_count >= self.max_samples:
                self.state = "IDLE"
                self._baseline = v.copy()
                self._prev_vec = v.copy()

                event = self._evaluate_accumulated_axis(current_time, dt=step_dt)

        return event

    def _record_first_step(self, dv: np.ndarray) -> None:
        """Capture the sign and dominant axis of the initial significant motion."""
        abs_dv = np.abs(dv)
        dom_idx = int(np.argmax(abs_dv))
        self._first_step_axis = dom_idx
        self._first_step_sign = 1 if dv[dom_idx] >= 0 else -1

    def _evaluate_accumulated_axis(
        self,
        current_time: float,
        dt: float = 0.010,
    ) -> Optional[AxisSwipeEvent]:
        """Normalize accumulated axis and score against candidate templates."""
        axis_mag = float(np.linalg.norm(self._axis_accum))

        # True physical duration (handles both real-time streaming and fast FIFO queue drains)
        elapsed_clock = current_time - self._swipe_start_time
        nominal_duration = self._sample_count * dt
        duration = max(elapsed_clock, nominal_duration) if elapsed_clock < 0.04 else elapsed_clock

        # Reject swipes that are too short or whose rotation axis canceled out
        if self._sample_count < self.min_samples or axis_mag < self.axis_min_magnitude:
            return None

        axis_unit = self._axis_accum / axis_mag

        # Compute cosine similarity score against all direction templates
        scores: Dict[str, float] = {}
        for d in self.directions:
            if d in self.templates:
                scores[d] = float(np.dot(axis_unit, self.templates[d]))
            else:
                scores[d] = -1.0

        best_dir = max(scores, key=scores.get) if scores else "UNKNOWN"
        best_score = scores.get(best_dir, 0.0)

        # Determine dominant physical coordinate axis
        axis_names = ["X", "Y", "Z"]
        dominant_axis = axis_names[int(np.argmax(np.abs(axis_unit)))]

        # Confidence gating
        assigned_dir = best_dir if best_score >= self.confidence_threshold else "UNKNOWN"

        return AxisSwipeEvent(
            direction=assigned_dir,
            confidence=best_score,
            scores=scores,
            axis_unit=axis_unit,
            axis_magnitude=axis_mag,
            sample_count=self._sample_count,
            duration_sec=duration,
            dominant_axis=dominant_axis,
            first_sign=self._first_step_sign,
            timestamp=current_time,
        )

    @staticmethod
    def extract_signature(
        vectors: np.ndarray,
        noise_floor: float = V6_NOISE_FLOOR,
    ) -> Tuple[Optional[np.ndarray], float, int]:
        """Compute accumulated rotation axis unit vector from a sequence of points.

        Args:
            vectors: Array of shape (N, 3) representing trajectory points.
            noise_floor: Minimum norm of delta to accumulate.

        Returns:
            (unit_axis, axis_magnitude, sample_count)
        """
        if len(vectors) < 2:
            return None, 0.0, 0

        axis_accum = np.zeros(3, dtype=float)
        v_prev = vectors[0]
        count = 0

        for i in range(1, len(vectors)):
            dv = vectors[i] - v_prev
            if float(np.linalg.norm(dv)) > noise_floor:
                axis_accum += np.cross(v_prev, dv)
                v_prev = vectors[i]
                count += 1

        mag = float(np.linalg.norm(axis_accum))
        if mag < 1e-9:
            return None, 0.0, count
        return axis_accum / mag, mag, count

    @staticmethod
    def train_templates(
        swipes_by_direction: Dict[str, List[np.ndarray]],
        noise_floor: float = V6_NOISE_FLOOR,
        max_outlier_angle_deg: float = 35.0,
    ) -> Dict[str, np.ndarray]:
        """Compute normalized template vectors with automatic outlier pruning."""
        trained: Dict[str, np.ndarray] = {}
        for d, swipe_list in swipes_by_direction.items():
            valid_axes: List[np.ndarray] = []
            for s in swipe_list:
                unit_axis, mag, count = AxisSignatureRecognizer.extract_signature(
                    s, noise_floor=noise_floor
                )
                if unit_axis is not None and count >= 2:
                    valid_axes.append(unit_axis)

            if valid_axes:
                # Initial centroid
                mean_vec = np.mean(valid_axes, axis=0)
                norm = float(np.linalg.norm(mean_vec))
                c0 = mean_vec / norm if norm > 1e-9 else mean_vec

                # Prune outliers exceeding max_outlier_angle_deg
                pruned = []
                for ax in valid_axes:
                    dot_val = float(np.dot(ax, c0))
                    ang = math.degrees(math.acos(min(max(dot_val, -1.0), 1.0)))
                    if ang <= max_outlier_angle_deg:
                        pruned.append(ax)

                final_axes = pruned if len(pruned) >= 2 else valid_axes
                final_mean = np.mean(final_axes, axis=0)
                final_norm = float(np.linalg.norm(final_mean))
                trained[d] = final_mean / final_norm if final_norm > 1e-9 else final_mean
            else:
                trained[d] = np.zeros(3)
        return trained

    @staticmethod
    def compute_cluster_metrics(
        axes: List[np.ndarray],
        direction: str = "DIR",
        prune_outliers: bool = True,
        max_outlier_angle_deg: float = 35.0,
    ) -> ClusterMetrics:
        """Compute angular dispersion, spread in degrees, and consistency score."""
        if not axes:
            return ClusterMetrics(
                direction=direction,
                centroid=np.zeros(3),
                angular_spread_deg=0.0,
                max_deviation_deg=0.0,
                consistency_score=0.0,
                sample_count=0,
                individual_axes=[],
            )

        mean_vec = np.mean(axes, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        c0 = mean_vec / norm if norm > 1e-9 else mean_vec.copy()

        # Outlier filtering
        if prune_outliers and len(axes) >= 3:
            kept_axes = []
            for ax in axes:
                dot_val = float(np.dot(ax, c0))
                ang = math.degrees(math.acos(min(max(dot_val, -1.0), 1.0)))
                if ang <= max_outlier_angle_deg:
                    kept_axes.append(ax)
            effective_axes = kept_axes if len(kept_axes) >= 2 else axes
        else:
            effective_axes = axes

        final_mean = np.mean(effective_axes, axis=0)
        final_norm = float(np.linalg.norm(final_mean))
        centroid = final_mean / final_norm if final_norm > 1e-9 else final_mean.copy()

        deviations_deg: List[float] = []
        cos_sims: List[float] = []
        for ax in effective_axes:
            dot_val = float(np.dot(ax, centroid))
            clamped_dot = min(max(dot_val, -1.0), 1.0)
            angle_deg = math.degrees(math.acos(clamped_dot))
            deviations_deg.append(angle_deg)
            cos_sims.append(clamped_dot)

        angular_spread = float(np.std(deviations_deg)) if len(deviations_deg) > 1 else 0.0
        max_dev = float(np.max(deviations_deg)) if deviations_deg else 0.0
        consistency = float(np.mean(cos_sims)) if cos_sims else 0.0

        return ClusterMetrics(
            direction=direction,
            centroid=centroid,
            angular_spread_deg=angular_spread,
            max_deviation_deg=max_dev,
            consistency_score=consistency,
            sample_count=len(effective_axes),
            individual_axes=list(effective_axes),
        )

    @staticmethod
    def compute_pairwise_separation(
        templates: Dict[str, np.ndarray],
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        """Compute pairwise cosine similarities and separation angles (in degrees).

        Returns:
            (cos_sim_dict, separation_deg_dict) where keys are (dir1, dir2)
        """
        cos_sims: Dict[Tuple[str, str], float] = {}
        angles_deg: Dict[Tuple[str, str], float] = {}

        dirs = list(templates.keys())
        for i, d1 in enumerate(dirs):
            for j, d2 in enumerate(dirs):
                dot_val = float(np.dot(templates[d1], templates[d2]))
                clamped_dot = min(max(dot_val, -1.0), 1.0)
                angle = math.degrees(math.acos(clamped_dot))
                cos_sims[(d1, d2)] = dot_val
                angles_deg[(d1, d2)] = angle

        return cos_sims, angles_deg
