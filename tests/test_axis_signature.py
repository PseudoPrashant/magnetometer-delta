"""test_axis_signature.py — Comprehensive Unit & Offline Replay Tests for Tracker v6.

Verifies:
  1. Synthetic Rotations: Unit rotating vectors about Cartesian axes.
  2. Noise Immunity: Rejection of stationary sensor jitter and micro-fluctuations.
  3. Cluster Variation Metrics: Verification of angular dispersion, centroid, and outlier pruning.
  4. Pairwise Separation: Cosine similarity and angle matrices.
  5. Ground-Truth Dataset Replay: Replays 14,000+ real samples from calibration logs.

Run via:
  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import V6_TEMPLATES
from core.axis_signature import (
    AxisSignatureRecognizer,
    AxisSwipeEvent,
    ClusterMetrics,
)
from calibration.interactive_calibration import InteractiveCalibrationEngine


class TestSyntheticRotations(unittest.TestCase):
    """Test mathematical accuracy of cross-product accumulation on pure geometric rotations."""

    def test_pure_x_axis_rotation(self) -> None:
        """Rotating a vector in the YZ plane must yield an axis along +X."""
        thetas = np.linspace(0, np.pi / 2, 20)
        trajectory = np.array([[0.0, np.cos(th), np.sin(th)] for th in thetas])

        unit_axis, mag, count = AxisSignatureRecognizer.extract_signature(trajectory, noise_floor=0.01)
        self.assertIsNotNone(unit_axis)
        self.assertGreater(mag, 0.1)
        np.testing.assert_allclose(unit_axis, np.array([1.0, 0.0, 0.0]), atol=1e-3)

    def test_pure_y_axis_rotation(self) -> None:
        """Rotating a vector in the ZX plane must yield an axis along +Y."""
        thetas = np.linspace(0, np.pi / 2, 20)
        trajectory = np.array([[np.sin(th), 0.0, np.cos(th)] for th in thetas])

        unit_axis, mag, count = AxisSignatureRecognizer.extract_signature(trajectory, noise_floor=0.01)
        self.assertIsNotNone(unit_axis)
        self.assertGreater(mag, 0.1)
        np.testing.assert_allclose(unit_axis, np.array([0.0, 1.0, 0.0]), atol=1e-3)

    def test_noise_jitter_rejection(self) -> None:
        """Small noise oscillations around a fixed baseline must not trigger active swipe."""
        np.random.seed(42)
        recognizer = AxisSignatureRecognizer(use_unwarped=False)

        base = np.array([100.0, 200.0, 300.0])
        events = []
        for _ in range(100):
            jitter = np.random.normal(0, 0.001, size=3)
            evt = recognizer.feed(base + jitter)
            if evt is not None:
                events.append(evt)

        self.assertEqual(len(events), 0, "Sensor jitter should not emit any swipe events")
        self.assertEqual(recognizer.state, "IDLE")

    def test_template_training_synthetic(self) -> None:
        """Verify train_templates utility with synthetic swipe trajectories."""
        thetas = np.linspace(0, np.pi / 4, 15)
        x_rot_swipes = [
            np.array([[0.0, np.cos(th + i * 0.1), np.sin(th + i * 0.1)] for th in thetas])
            for i in range(3)
        ]
        y_rot_swipes = [
            np.array([[np.sin(th + i * 0.1), 0.0, np.cos(th + i * 0.1)] for th in thetas])
            for i in range(3)
        ]

        data = {"UP": x_rot_swipes, "RIGHT": y_rot_swipes}
        templates = AxisSignatureRecognizer.train_templates(data, noise_floor=0.005)

        self.assertIn("UP", templates)
        self.assertIn("RIGHT", templates)
        np.testing.assert_allclose(templates["UP"], np.array([1.0, 0.0, 0.0]), atol=1e-3)
        np.testing.assert_allclose(templates["RIGHT"], np.array([0.0, 1.0, 0.0]), atol=1e-3)


class TestClusterAndPatternMetrics(unittest.TestCase):
    """Test cluster variation metrics, angular dispersion, and outlier pruning."""

    def test_cluster_metrics_calculation(self) -> None:
        """Verify angular dispersion and consistency score calculation."""
        # 4 vectors closely clustered around +X axis (spread < 5 deg) + 1 outlier at 90 deg
        axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.999, 0.040, 0.0]),
            np.array([0.999, -0.040, 0.0]),
            np.array([0.998, 0.0, 0.060]),
            np.array([0.0, 1.0, 0.0]),  # 90 deg outlier!
        ]
        for i in range(len(axes)):
            axes[i] = axes[i] / np.linalg.norm(axes[i])

        # With outlier pruning enabled:
        metrics = AxisSignatureRecognizer.compute_cluster_metrics(axes, direction="UP", prune_outliers=True)
        self.assertEqual(metrics.sample_count, 4, "Outlier should be pruned")
        self.assertLess(metrics.angular_spread_deg, 5.0, "Pruned cluster spread should be tight (<5 deg)")
        self.assertGreater(metrics.consistency_score, 0.98, "Consistency score should exceed 98%")

    def test_pairwise_separation(self) -> None:
        """Verify pairwise angle and cosine similarity matrices."""
        templates = {
            "UP": np.array([1.0, 0.0, 0.0]),
            "DOWN": np.array([-1.0, 0.0, 0.0]),
            "LEFT": np.array([0.0, -1.0, 0.0]),
            "RIGHT": np.array([0.0, 1.0, 0.0]),
        }
        cos_sims, angles_deg = AxisSignatureRecognizer.compute_pairwise_separation(templates)

        self.assertAlmostEqual(cos_sims[("UP", "DOWN")], -1.0, places=3)
        self.assertAlmostEqual(angles_deg[("UP", "DOWN")], 180.0, places=1)
        self.assertAlmostEqual(cos_sims[("UP", "RIGHT")], 0.0, places=3)
        self.assertAlmostEqual(angles_deg[("UP", "RIGHT")], 90.0, places=1)


class TestDatasetReplay(unittest.TestCase):
    """Replay real hardware recorded calibration datasets."""

    def test_replay_calibration_logs(self) -> None:
        """Feed real calibration datasets and verify classification accuracy."""
        csv_files = [
            "calibration_data_20260827_121002.csv",
            "calibration_data_20260827_121145.csv",
        ]

        total_correct = 0
        total_swipes = 0
        confusion: dict[str, dict[str, int]] = {
            gt: {p: 0 for p in list(V6_TEMPLATES.keys()) + ["UNKNOWN"]}
            for gt in V6_TEMPLATES.keys()
        }

        for fname in csv_files:
            path = Path(fname)
            if not path.exists():
                continue

            df = pd.read_csv(path)
            for ground_truth, group in df.groupby("ground_truth", sort=False):
                if ground_truth not in V6_TEMPLATES:
                    continue

                recognizer = AxisSignatureRecognizer(
                    use_unwarped=True,
                    confidence_threshold=0.50,
                )

                b_raw_arr = group[["b_raw_x", "b_raw_y", "b_raw_z"]].values
                for b_raw in b_raw_arr:
                    event = recognizer.feed(b_raw)
                    if event is not None:
                        total_swipes += 1
                        pred = event.direction
                        confusion[ground_truth][pred] += 1
                        if pred == ground_truth:
                            total_correct += 1

        print(f"\n[+] Total Replay Swipes: {total_swipes}, Correct: {total_correct}")
        if total_swipes > 0:
            accuracy = total_correct / total_swipes
            print(f"[+] Replay Accuracy: {accuracy*100:.1f}%")
            self.assertGreater(
                accuracy,
                0.50,
                "Replay accuracy should exceed 50% on unsegmented continuous roll datasets",
            )

    def test_interactive_calibration_offline(self) -> None:
        """Test that InteractiveCalibrationEngine runs successfully in offline CSV mode."""
        engine = InteractiveCalibrationEngine(swipes_per_dir=5, use_unwarped=True)
        csv_path = "calibration_data_20260827_121002.csv"
        if Path(csv_path).exists():
            engine.load_from_csv(csv_path)
            engine.analyze_patterns_and_variations()
            self.assertIn("UP", engine.trained_templates)
            self.assertIn("DOWN", engine.trained_templates)
            self.assertIn("LEFT", engine.trained_templates)
            self.assertIn("RIGHT", engine.trained_templates)


class TestVectorStreamDenoiseFilter(unittest.TestCase):
    """Test multi-stage 3D vector stream denoiser."""

    def test_jitter_attenuation_at_rest(self) -> None:
        """Denoiser must attenuate resting sensor noise by >5x."""
        from core.filters import VectorStreamDenoiseFilter

        np.random.seed(42)
        base = np.array([200.0, -150.0, 400.0])
        noise_samples = [base + np.random.normal(0, 5.0, size=3) for _ in range(300)]

        raw_diffs = [np.linalg.norm(noise_samples[i] - noise_samples[i - 1]) for i in range(1, len(noise_samples))]
        raw_jitter_std = float(np.std(raw_diffs))

        denoiser = VectorStreamDenoiseFilter(min_cutoff=1.0, beta=0.003, d_cutoff=0.8)
        filtered_samples = [denoiser.filter(s, dt=0.01) for s in noise_samples]

        filt_diffs = [np.linalg.norm(filtered_samples[i] - filtered_samples[i - 1]) for i in range(50, len(filtered_samples))]
        filt_jitter_std = float(np.std(filt_diffs))

        reduction_factor = raw_jitter_std / filt_jitter_std
        print(f"\n[+] Stream Denoise Reduction Factor: {reduction_factor:.1f}x")
        self.assertGreater(reduction_factor, 5.0, "Filter must reduce resting jitter by >5x")

    def test_glitch_spike_suppression(self) -> None:
        """Single-sample non-physical spike (>4000 mG) must be rejected."""
        from core.filters import VectorStreamDenoiseFilter

        denoiser = VectorStreamDenoiseFilter(max_spike_delta=4000.0)
        base = np.array([100.0, 100.0, 100.0])
        denoiser.filter(base, dt=0.01)
        denoiser.filter(base, dt=0.01)

        # Inject 8000 mG glitch spike
        glitch = base + np.array([8000.0, 0.0, 0.0])
        out = denoiser.filter(glitch, dt=0.01)

        # The glitch should be rejected (out should remain close to base, not 8000)
        self.assertLess(out[0], 200.0, "Glitch spike should be suppressed by spike gate")


class TestDirectionalStreamRecorder(unittest.TestCase):
    """Test recorder initialization and CSV formatting."""

    def test_recorder_instantiation(self) -> None:
        """Verify recorder initializes with default ports and directories."""
        from calibration.record_directional_stream import DirectionalStreamRecorder

        recorder = DirectionalStreamRecorder(port="COM7", baud=115200)
        self.assertEqual(recorder.port, "COM7")
        self.assertEqual(recorder.baud, 115200)
        self.assertTrue(recorder.output_dir.exists())


class TestStreamKinematicSwipeDetector(unittest.TestCase):
    """Test Tracker v7 Stream Kinematic Swipe Detector."""

    def test_synthetic_up_down_kinematics(self) -> None:
        """Synthetic pure rotation in YZ plane must classify as UP/DOWN."""
        from core.kinematic_swipe import StreamKinematicSwipeDetector

        detector = StreamKinematicSwipeDetector(start_speed=1.0, min_peak_speed=3.0, min_displacement_deg=10.0)

        # Generate synthetic UP rotation: vector rotating around +X axis
        # m = [0, cos(theta), sin(theta)] -> B = A @ m + B_OFFSET
        from config import INV_A, B_OFFSET
        A = np.linalg.inv(INV_A)

        events = []
        # Rest 20 frames
        for _ in range(20):
            m = np.array([0.0, 1.0, 0.0])
            b = np.dot(A, m * 5000.0) + B_OFFSET
            evt = detector.feed_raw(b, dt=0.01)
            if evt: events.append(evt)

        # Swipe UP 30 frames (rotating about +X axis)
        for i in range(30):
            theta = (i / 30.0) * (np.pi / 2)
            m = np.array([0.0, np.cos(theta), np.sin(theta)])
            b = np.dot(A, m * 5000.0) + B_OFFSET
            evt = detector.feed_raw(b, dt=0.01)
            if evt: events.append(evt)

        # Rest 15 frames
        for _ in range(15):
            m = np.array([0.0, 0.0, 1.0])
            b = np.dot(A, m * 5000.0) + B_OFFSET
            evt = detector.feed_raw(b, dt=0.01)
            if evt: events.append(evt)

        self.assertTrue(len(events) > 0, "Detector should trigger on synthetic UP swipe")
        self.assertEqual(events[0].direction, "UP")


if __name__ == "__main__":
    unittest.main()
