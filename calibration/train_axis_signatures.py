"""train_axis_signatures.py — Accumulated Rotation-Axis Signature Trainer.

Guided training tool to derive 3D unit template rotation axes for Tracker v6.
Can run in three modes:
  1. Interactive Guided Mode (--interactive): Full step-by-step swipe capture,
     pattern variation analytics, and live prediction test loop.
  2. Live Serial Mode: Connects to ESP32 over serial and guides user through
     5–10 swipes per direction with live real-time feedback.
  3. Offline CSV Mode (--from-csv <path>): Extracts, prunes outliers, and
     aggregates signatures from previously collected calibration logs.

Outputs:
  - Formatted Python dictionary ready for config.py::V6_TEMPLATES.
  - Pairwise similarity and angular separation matrices.

Run from project root:
  python -m calibration.train_axis_signatures --interactive
  python -m calibration.train_axis_signatures --from-csv calibration_data_20260827_121002.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from config import (
    BAUD_RATE,
    B_OFFSET,
    INV_A,
    LSB_TO_MGAUSS,
    SERIAL_PORT,
    V6_AXIS_MIN_MAGNITUDE,
    V6_DIRECTIONS,
    V6_MIN_SAMPLES,
    V6_NOISE_FLOOR,
    V6_SILENCE_TAPS,
    V6_SWIPE_START_THRESH,
    V6_USE_UNWARPED,
)
from core.axis_signature import AxisSignatureRecognizer, ClusterMetrics


def print_template_matrix(templates: Dict[str, np.ndarray]) -> None:
    """Print direction unit vectors and pairwise dot-product matrix."""
    dirs = list(templates.keys())
    print("\n" + "=" * 60)
    print("CALIBRATED ROTATION-AXIS TEMPLATES (3D Unit Vectors)")
    print("=" * 60)
    for d in dirs:
        v = templates[d]
        print(f"  {d:<10s} : np.array([{v[0]:+7.4f}, {v[1]:+7.4f}, {v[2]:+7.4f}])")

    print("\n" + "-" * 60)
    print("PAIRWISE COSINE SIMILARITY MATRIX (Target: Orthogonal ~ 0, Opposite ~ -1)")
    print("-" * 60)
    header = f"{'':10s}" + "".join(f"{d:>10s}" for d in dirs)
    print(header)
    for d1 in dirs:
        row = f"{d1:<10s}"
        for d2 in dirs:
            sim = float(np.dot(templates[d1], templates[d2]))
            row += f"{sim:+10.3f}"
        print(row)

    print("\n" + "-" * 60)
    print("PAIRWISE ANGULAR SEPARATION (Target: Orthogonal ~ 90 deg, Opposite ~ 180 deg)")
    print("-" * 60)
    print(header)
    for d1 in dirs:
        row = f"{d1:<10s}"
        for d2 in dirs:
            dot_val = float(np.dot(templates[d1], templates[d2]))
            clamped = min(max(dot_val, -1.0), 1.0)
            ang = math.degrees(math.acos(clamped))
            row += f"{ang:9.1f} deg"
        print(row)
    print("=" * 60)

    print("\n[+] Ready for config.py: Paste the block below into config.py::V6_TEMPLATES\n")
    print("V6_TEMPLATES: Final[dict[str, np.ndarray]] = {")
    for d in dirs:
        v = templates[d]
        print(f'    "{d}": np.array([{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]),')
    print("}\n")


def train_from_csv(csv_path: str, use_unwarped: bool = V6_USE_UNWARPED) -> Dict[str, np.ndarray]:
    """Train templates offline using a ground-truth labeled CSV dataset with robust outlier pruning."""
    import pandas as pd

    print(f"[*] Loading calibration dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if "ground_truth" not in df.columns or "b_raw_x" not in df.columns:
        print("[-] Error: CSV must contain 'b_raw_x', 'b_raw_y', 'b_raw_z', and 'ground_truth' columns.")
        sys.exit(1)

    swipes_by_dir: Dict[str, List[np.ndarray]] = {d: [] for d in df["ground_truth"].unique() if pd.notna(d)}

    for ground_truth, group in df.groupby("ground_truth", sort=False):
        if not ground_truth or ground_truth not in swipes_by_dir:
            continue

        b_raw_arr = group[["b_raw_x", "b_raw_y", "b_raw_z"]].values
        recognizer = AxisSignatureRecognizer(
            use_unwarped=use_unwarped,
            confidence_threshold=-1.0,
        )

        for i in range(len(b_raw_arr)):
            b_raw = b_raw_arr[i]
            event = recognizer.feed(b_raw)
            if event is not None:
                swipes_by_dir[ground_truth].append(event.axis_unit)

    print(f"[+] Extraction complete:")
    templates: Dict[str, np.ndarray] = {}
    for d, axes in swipes_by_dir.items():
        metrics = AxisSignatureRecognizer.compute_cluster_metrics(axes, direction=d, prune_outliers=True)
        templates[d] = metrics.centroid
        print(f"    {metrics.summary_line()}")

    return templates


def train_live_serial(
    directions: List[str],
    swipes_target: int = 6,
    use_unwarped: bool = V6_USE_UNWARPED,
) -> Dict[str, np.ndarray]:
    """Guide user through live physical training over serial connection."""
    import serial

    print(f"[*] Opening serial link on {SERIAL_PORT} @ {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[+] Connected to {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"[-] Serial connection failed: {e}")
        print("[-] Check connection or specify an offline CSV with --from-csv")
        sys.exit(1)

    collected_signatures: Dict[str, List[np.ndarray]] = {d: [] for d in directions}
    recognizer = AxisSignatureRecognizer(
        use_unwarped=use_unwarped,
        confidence_threshold=-1.0,
    )

    print("\n" + "=" * 60)
    print("LIVE GUIDED AXIS SIGNATURE TRAINING")
    print(f"Target: {swipes_target} swipes for each direction: {', '.join(directions)}")
    print("=" * 60)

    try:
        for phase_idx, target_dir in enumerate(directions, 1):
            print(f"\n>>> PHASE {phase_idx}/{len(directions)}: Swipe {target_dir} ({swipes_target} times) <<<")
            print(f"[*] Perform distinct, deliberate {target_dir} swipes with ~0.5s pause between swipes.")
            recognizer.reset()

            swipes_done = 0
            while swipes_done < swipes_target:
                lines = ser.read_all().decode("latin-1", errors="ignore").splitlines()
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

                    b_raw_mg = np.array(raw_lsb) * LSB_TO_MGAUSS
                    event = recognizer.feed(b_raw_mg)
                    if event is not None:
                        swipes_done += 1
                        collected_signatures[target_dir].append(event.axis_unit)
                        u = event.axis_unit
                        print(
                            f"  [{swipes_done}/{swipes_target}] Captured {target_dir} swipe: "
                            f"axis=[{u[0]:+5.2f}, {u[1]:+5.2f}, {u[2]:+5.2f}], "
                            f"samples={event.sample_count}, dur={event.duration_sec*1000:.0f}ms"
                        )
                        if swipes_done >= swipes_target:
                            break

                time.sleep(0.005)

            print(f"[+] Completed {target_dir} phase! Pause 2 seconds before next phase...")
            time.sleep(2)
            ser.reset_input_buffer()

    except KeyboardInterrupt:
        print("\n[*] Training interrupted by user.")
    finally:
        ser.close()
        print("[+] Serial port closed.")

    templates: Dict[str, np.ndarray] = {}
    for d, axes in collected_signatures.items():
        metrics = AxisSignatureRecognizer.compute_cluster_metrics(axes, direction=d, prune_outliers=True)
        templates[d] = metrics.centroid
        print(f"    {metrics.summary_line()}")

    return templates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accumulated Rotation-Axis Signature Calibration & Trainer"
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Path to recorded CSV dataset for offline training",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the full interactive calibration and pattern analyzer engine",
    )
    parser.add_argument(
        "--swipes",
        type=int,
        default=6,
        help="Number of swipes per direction for live training (default: 6)",
    )
    parser.add_argument(
        "--raw-field",
        action="store_true",
        help="Train in raw field space B_clean instead of unwarped dipole space m",
    )
    args = parser.parse_args()

    if args.interactive:
        from calibration.interactive_calibration import main as interactive_main
        interactive_main()
        return

    use_unwarped = not args.raw_field

    if args.from_csv:
        templates = train_from_csv(args.from_csv, use_unwarped=use_unwarped)
    else:
        templates = train_live_serial(
            directions=V6_DIRECTIONS,
            swipes_target=args.swipes,
            use_unwarped=use_unwarped,
        )

    print_template_matrix(templates)


if __name__ == "__main__":
    main()
