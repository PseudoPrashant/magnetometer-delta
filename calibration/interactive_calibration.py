"""interactive_calibration.py — Guided Per-Direction Swipe Calibration & Pattern Analyzer.

Advanced calibration workflow:
  1. Continuous COM port streaming in background thread.
  2. Guided per-direction swipe capture with instant feedback and Accept/Retry/Discard.
  3. Pattern & variation analysis (intra-cluster angular spread, consistency, separation matrix).
  4. Seamless transition to live prediction validation mode.
  5. Automatic dataset logging and config.py template update.

Run from project root:
  python -m calibration.interactive_calibration
  python -m calibration.interactive_calibration --from-csv calibration_data_20260827_121002.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    BAUD_RATE,
    B_OFFSET,
    INV_A,
    LSB_TO_MGAUSS,
    SERIAL_PORT,
    V6_CONFIDENCE_THRESHOLD,
    V6_DIRECTIONS,
    V6_MAX_SAMPLES,
    V6_MIN_SAMPLES,
    V6_NOISE_FLOOR,
    V6_SILENCE_TAPS,
    V6_SWIPE_START_THRESH,
    V6_TEMPLATES,
    V6_USE_UNWARPED,
)
from core.axis_signature import (
    AxisSignatureRecognizer,
    AxisSwipeEvent,
    ClusterMetrics,
)


class SerialStreamWorker:
    """Background thread that continuously streams and parses raw serial data."""

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE) -> None:
        self.port = port
        self.baud = baud
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.sample_queue: queue.Queue[Tuple[float, np.ndarray]] = queue.Queue(maxsize=5000)
        self.all_samples: List[Tuple[float, np.ndarray]] = []
        self._ser = None

    def start(self) -> bool:
        """Open serial link and launch streaming thread."""
        try:
            import serial

            self._ser = serial.Serial(self.port, self.baud, timeout=0.01)
            time.sleep(1.8)
            self._ser.reset_input_buffer()
            print(f"[+] Connected to {self.port} @ {self.baud} baud.")
        except Exception as e:
            print(f"[-] Serial connection failed on {self.port}: {e}")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self) -> None:
        """Continuous background read loop."""
        while self.running and self._ser is not None:
            try:
                lines = self._ser.read_all().decode("latin-1", errors="ignore").splitlines()
                now = time.perf_counter()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue
                    try:
                        raw_lsb = [float(p) for p in parts]
                    except ValueError:
                        continue

                    b_raw_mg = np.array(raw_lsb, dtype=float) * LSB_TO_MGAUSS
                    self.all_samples.append((now, b_raw_mg))

                    if not self.sample_queue.full():
                        self.sample_queue.put_nowait((now, b_raw_mg))

            except Exception:
                time.sleep(0.01)

            time.sleep(0.005)

    def stop(self) -> None:
        """Stop background worker and close serial connection."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        print("[+] Serial background stream stopped.")


class InteractiveCalibrationEngine:
    """Manages guided directional swipe capture, cluster analysis, and prediction testing."""

    def __init__(
        self,
        directions: Optional[List[str]] = None,
        swipes_per_dir: int = 5,
        use_unwarped: bool = V6_USE_UNWARPED,
    ) -> None:
        self.directions = list(directions or V6_DIRECTIONS)
        self.swipes_per_dir = swipes_per_dir
        self.use_unwarped = use_unwarped

        # Storage: direction -> list of (axis_unit, sample_array, duration, sample_count)
        self.captured_data: Dict[str, List[dict]] = {d: [] for d in self.directions}
        self.cluster_metrics: Dict[str, ClusterMetrics] = {}
        self.trained_templates: Dict[str, np.ndarray] = {}

    def capture_single_swipe_live(
        self,
        streamer: SerialStreamWorker,
        target_dir: str,
        swipe_num: int,
    ) -> Optional[dict]:
        """Isolate and capture one deliberate swipe gesture from the continuous stream."""
        recognizer = AxisSignatureRecognizer(
            use_unwarped=self.use_unwarped,
            confidence_threshold=-1.0,  # accept all valid motion windows
            silence_taps=5,
            min_samples=V6_MIN_SAMPLES,
            max_samples=80,
            enable_denoising=True,
        )

        print(f"\n  -------------------------------------------------------------")
        print(f"  [{target_dir} - Swipe {swipe_num}/{self.swipes_per_dir}]")
        print(f"  Hold ball still, then perform ONE deliberate {target_dir} swipe.")
        print(f"  -------------------------------------------------------------")

        # Drain pending queue to ensure fresh stream
        while not streamer.sample_queue.empty():
            try:
                streamer.sample_queue.get_nowait()
            except queue.Empty:
                break

        motion_detected = False
        captured_samples: List[np.ndarray] = []
        swipe_event: Optional[AxisSwipeEvent] = None
        start_wait = time.perf_counter()

        while True:
            # Check timeout (30 seconds per swipe attempt)
            if time.perf_counter() - start_wait > 30.0:
                print("  [-] Timeout waiting for swipe. Retrying...")
                return None

            try:
                item = streamer.sample_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if isinstance(item, tuple):
                ts, b_raw = item
            else:
                ts, b_raw = time.perf_counter(), item

            captured_samples.append(b_raw)
            event = recognizer.feed(b_raw, dt=0.010, now=ts)

            if recognizer.state == "ACTIVE" and not motion_detected:
                motion_detected = True
                print("    [*] Motion detected! Tracking swipe rotation...", end="", flush=True)

            if event is not None:
                swipe_event = event
                print(" Done!")
                break

        if swipe_event is None or swipe_event.axis_unit is None:
            print("  [-] Swipe was too weak or canceled out. Please retry.")
            return None

        u = swipe_event.axis_unit
        dur_ms = swipe_event.duration_sec * 1000
        print(
            f"  [+] Captured {target_dir} Swipe #{swipe_num}: "
            f"axis=[{u[0]:+5.2f}, {u[1]:+5.2f}, {u[2]:+5.2f}] | "
            f"dur={dur_ms:3.0f}ms | samples={swipe_event.sample_count} | dom={swipe_event.dominant_axis}"
        )

        return {
            "direction": target_dir,
            "axis_unit": u,
            "duration_sec": swipe_event.duration_sec,
            "sample_count": swipe_event.sample_count,
            "dominant_axis": swipe_event.dominant_axis,
            "samples": np.array(captured_samples),
        }

    def run_live_guided_capture(self, streamer: SerialStreamWorker) -> None:
        """Run step-by-step guided capture for every direction."""
        print("\n" + "=" * 70)
        print("PHASE 1: GUIDED DIRECTIONAL SWIPE CAPTURE")
        print(f"Target: {self.swipes_per_dir} clean swipes for each direction: {', '.join(self.directions)}")
        print("=" * 70)

        for dir_idx, d in enumerate(self.directions, 1):
            print(f"\n======================================================================")
            print(f">>> DIRECTION {dir_idx}/{len(self.directions)}: {d}")
            print(f"======================================================================")

            swipes_collected = 0
            while swipes_collected < self.swipes_per_dir:
                swipe_record = self.capture_single_swipe_live(
                    streamer, target_dir=d, swipe_num=swipes_collected + 1
                )

                if swipe_record is None:
                    continue

                # Interactive prompt to accept or retry
                choice = input("    [ENTER: Accept]  [R: Retry this swipe]  [S: Skip]: ").strip().lower()
                if choice == "r":
                    print("    [*] Discarding last swipe. Ready to retry...")
                    continue
                elif choice == "s":
                    print(f"    [*] Skipping remaining {d} swipes.")
                    break
                else:
                    self.captured_data[d].append(swipe_record)
                    swipes_collected += 1
                    print(f"    [+] Saved ({swipes_collected}/{self.swipes_per_dir}) for {d}.")

            print(f"\n[+] Completed collection for direction '{d}'! Brief pause...")
            time.sleep(1.0)

    def load_from_csv(self, csv_path: str) -> None:
        """Extract directional swipes from an existing ground-truth CSV dataset."""
        import pandas as pd

        print(f"[*] Replaying dataset from CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        if "ground_truth" not in df.columns or "b_raw_x" not in df.columns:
            print("[-] Error: CSV must contain 'b_raw_x', 'b_raw_y', 'b_raw_z', and 'ground_truth'.")
            sys.exit(1)

        for ground_truth, group in df.groupby("ground_truth", sort=False):
            if ground_truth not in self.captured_data:
                continue

            recognizer = AxisSignatureRecognizer(
                use_unwarped=self.use_unwarped,
                confidence_threshold=-1.0,
            )

            b_raw_arr = group[["b_raw_x", "b_raw_y", "b_raw_z"]].values
            for b in b_raw_arr:
                evt = recognizer.feed(b)
                if evt is not None:
                    self.captured_data[ground_truth].append({
                        "direction": ground_truth,
                        "axis_unit": evt.axis_unit,
                        "duration_sec": evt.duration_sec,
                        "sample_count": evt.sample_count,
                        "dominant_axis": evt.dominant_axis,
                        "samples": np.zeros((evt.sample_count, 3)),
                    })

    def analyze_patterns_and_variations(self) -> None:
        """Analyze intra-cluster consistency, angular spread, and inter-direction separation."""
        print("\n" + "=" * 70)
        print("PHASE 2: PATTERN & VARIATION ANALYSIS")
        print("=" * 70)

        # 1. Intra-cluster statistics
        print("\n--- 1. Intra-Direction Cluster Consistency ---")
        for d in self.directions:
            records = self.captured_data[d]
            axes = [r["axis_unit"] for r in records]
            metrics = AxisSignatureRecognizer.compute_cluster_metrics(axes, direction=d)
            self.cluster_metrics[d] = metrics
            self.trained_templates[d] = metrics.centroid

            print(metrics.summary_line())

            # Print individual variations if multiple swipes
            if len(axes) > 1:
                for idx, r in enumerate(records, 1):
                    ax = r["axis_unit"]
                    dot_c = float(np.dot(ax, metrics.centroid))
                    angle_deg = math.degrees(math.acos(min(max(dot_c, -1.0), 1.0)))
                    flag = " (! High variance)" if angle_deg > 30.0 else ""
                    print(
                        f"    Swipe #{idx:02d}: axis=[{ax[0]:+5.2f}, {ax[1]:+5.2f}, {ax[2]:+5.2f}] | "
                        f"dev={angle_deg:4.1f} deg from centroid{flag}"
                    )

        # 2. Inter-direction pairwise separation matrix
        print("\n--- 2. Inter-Direction Angular Separation & Similarity Matrix ---")
        cos_sims, angles_deg = AxisSignatureRecognizer.compute_pairwise_separation(self.trained_templates)

        header = f"{'':10s}" + "".join(f"{d:>12s}" for d in self.directions)
        print(f"\n[Pairwise Cosine Similarity] (Target: Opposites ~ -1.0, Orthogonal ~ 0.0)")
        print(header)
        for d1 in self.directions:
            row = f"{d1:<10s}"
            for d2 in self.directions:
                val = cos_sims.get((d1, d2), 0.0)
                row += f"{val:+12.3f}"
            print(row)

        print(f"\n[Pairwise Angular Separation in Degrees] (Target: Opposites ~ 180 deg, Orthogonal ~ 90 deg)")
        print(header)
        for d1 in self.directions:
            row = f"{d1:<10s}"
            for d2 in self.directions:
                ang = angles_deg.get((d1, d2), 0.0)
                row += f"{ang:11.1f} deg"
            print(row)

        print("\n" + "=" * 70)
        self._evaluate_overall_calibration_quality()

    def _evaluate_overall_calibration_quality(self) -> None:
        """Provide a clear rating and diagnosis of the calibration geometry."""
        print("CALIBRATION QUALITY EVALUATION:")
        warnings = []

        for d, m in self.cluster_metrics.items():
            if m.sample_count < 2:
                warnings.append(f"Direction '{d}' has fewer than 2 samples.")
            elif m.angular_spread_deg > 25.0:
                warnings.append(f"Direction '{d}' has high angular dispersion ({m.angular_spread_deg:.1f} deg).")

        # Check opposites separation
        opp_pairs = [("UP", "DOWN"), ("LEFT", "RIGHT")]
        _, angles_deg = AxisSignatureRecognizer.compute_pairwise_separation(self.trained_templates)
        for d1, d2 in opp_pairs:
            if d1 in self.trained_templates and d2 in self.trained_templates:
                ang = angles_deg.get((d1, d2), 0.0)
                if ang < 140.0:
                    warnings.append(f"Opposite pair {d1} <-> {d2} separation is only {ang:.1f} deg (expected > 150 deg).")

        if not warnings:
            print("  [+] Rating: EXCELLENT! High cluster consistency and clear angular separation.")
        else:
            print("  [!] Rating: USABLE WITH WARNINGS:")
            for w in warnings:
                print(f"      - {w}")

    def run_live_prediction_test(self, streamer: SerialStreamWorker) -> None:
        """Interactive test mode where user performs test gestures and verifies predictions."""
        print("\n" + "=" * 70)
        print("PHASE 3: SEAMLESS LIVE PREDICTION VALIDATION")
        print("Perform test swipes in any direction. The engine will classify them live!")
        print("Press Ctrl+C at any time to exit validation.")
        print("=" * 70)

        recognizer = AxisSignatureRecognizer(
            templates=self.trained_templates,
            use_unwarped=self.use_unwarped,
            confidence_threshold=V6_CONFIDENCE_THRESHOLD,
            min_samples=V6_MIN_SAMPLES,
            max_samples=80,
            enable_denoising=True,
        )

        test_count = 0
        correct_count = 0

        # Drain queue
        while not streamer.sample_queue.empty():
            streamer.sample_queue.get_nowait()

        try:
            while True:
                try:
                    item = streamer.sample_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if isinstance(item, tuple):
                    ts, b_raw = item
                else:
                    ts, b_raw = time.perf_counter(), item

                event = recognizer.feed(b_raw, dt=0.010, now=ts)
                if event is not None:
                    test_count += 1
                    u = event.axis_unit
                    conf_pct = event.confidence * 100
                    print(
                        f"\n>>> [LIVE PREDICTION #{test_count}] PREDICTED: {event.direction:<6s} "
                        f"(Confidence: {conf_pct:4.1f}%) | dur={event.duration_sec*1000:3.0f}ms | "
                        f"axis=[{u[0]:+5.2f}, {u[1]:+5.2f}, {u[2]:+5.2f}] <<<"
                    )

                    # Print all score bars
                    score_strs = [f"{d}: {event.scores.get(d, 0.0):+5.2f}" for d in self.directions]
                    print(f"    Scores: {' | '.join(score_strs)}")

                    ans = input("    Was this prediction correct? [Y/n/q]: ").strip().lower()
                    if ans == "q":
                        break
                    elif ans != "n":
                        correct_count += 1
                        print("    [+] Marked as CORRECT.")
                    else:
                        print("    [-] Marked as INCORRECT.")

                    print(f"    Current Accuracy: {correct_count}/{test_count} ({correct_count/test_count*100:.1f}%)")

        except KeyboardInterrupt:
            print("\n[*] Live testing ended.")

        if test_count > 0:
            print(f"\n[+] Final Live Validation Score: {correct_count}/{test_count} ({correct_count/test_count*100:.1f}%)")

    def save_results(self, output_csv_prefix: str = "calibration_session") -> str:
        """Export full dataset to CSV log and format config.py update."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{output_csv_prefix}_{timestamp}.csv"

        # Save template dictionary block
        print("\n" + "=" * 70)
        print("FINAL CALIBRATED ROTATION-AXIS TEMPLATES")
        print("=" * 70)
        print("V6_TEMPLATES: Final[dict[str, np.ndarray]] = {")
        for d in self.directions:
            v = self.trained_templates[d]
            print(f'    "{d}": np.array([{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]),')
        print("}\n")

        # Save session CSV
        total_records = sum(len(records) for records in self.captured_data.values())
        if total_records > 0:
            with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "direction", "swipe_idx", "axis_x", "axis_y", "axis_z",
                    "duration_sec", "sample_count", "dominant_axis"
                ])
                for d, records in self.captured_data.items():
                    for idx, r in enumerate(records, 1):
                        ax = r["axis_unit"]
                        writer.writerow([
                            d, idx, f"{ax[0]:.6f}", f"{ax[1]:.6f}", f"{ax[2]:.6f}",
                            f"{r['duration_sec']:.4f}", r["sample_count"], r["dominant_axis"]
                        ])
            print(f"[+] Calibration swipe records saved to: {csv_filename}")

        return csv_filename

    def update_config_file(self) -> None:
        """Prompt to automatically update config.py with newly calibrated templates."""
        config_path = Path("config.py")
        if not config_path.exists():
            return

        choice = input("\nWould you like to automatically update config.py with these templates? [y/N]: ").strip().lower()
        if choice == "y":
            lines = config_path.read_text(encoding="utf-8").splitlines()
            new_lines = []
            in_templates = False

            template_block = "V6_TEMPLATES: Final[dict[str, np.ndarray]] = {\n"
            for d in self.directions:
                v = self.trained_templates[d]
                template_block += f'    "{d}": np.array([{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}]),\n'
            template_block += "}"

            i = 0
            while i < len(lines):
                if lines[i].startswith("V6_TEMPLATES: Final[dict[str, np.ndarray]] = {"):
                    new_lines.append(template_block)
                    while i < len(lines) and not lines[i].startswith("}"):
                        i += 1
                    i += 1
                else:
                    new_lines.append(lines[i])
                    i += 1

            config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print("[+] config.py::V6_TEMPLATES successfully updated!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Guided Rotation-Axis Calibration & Pattern Analyzer"
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Replay and analyze ground-truth CSV dataset offline",
    )
    parser.add_argument(
        "--swipes",
        type=int,
        default=5,
        help="Target swipes per direction (default: 5)",
    )
    parser.add_argument(
        "--raw-field",
        action="store_true",
        help="Use raw B_clean field space instead of unwarped dipole space m",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=SERIAL_PORT,
        help=f"Serial port (default: {SERIAL_PORT})",
    )
    args = parser.parse_args()

    engine = InteractiveCalibrationEngine(
        directions=V6_DIRECTIONS,
        swipes_per_dir=args.swipes,
        use_unwarped=not args.raw_field,
    )

    streamer: Optional[SerialStreamWorker] = None

    if args.from_csv:
        engine.load_from_csv(args.from_csv)
    else:
        streamer = SerialStreamWorker(port=args.port, baud=BAUD_RATE)
        if not streamer.start():
            sys.exit(1)

        try:
            engine.run_live_guided_capture(streamer)
        except KeyboardInterrupt:
            print("\n[*] Capture interrupted by user.")

    # Phase 2: Pattern analysis
    engine.analyze_patterns_and_variations()

    # Save outputs
    engine.save_results()

    # Phase 3: Live prediction testing (if on live stream)
    if streamer and streamer.running:
        try:
            engine.run_live_prediction_test(streamer)
        finally:
            streamer.stop()

    if not args.from_csv:
        engine.update_config_file()


if __name__ == "__main__":
    main()
