"""analyze_recorded_streams.py — Deep Kinematic Analysis of Continuous Stream Recordings.

Analyzes raw recorded stream datasets (`recordings/stream_*.csv`), extracts physical
trajectories, evaluates motion variances, principal rotation planes, and tests
the Stream Kinematic Swipe Detector (Tracker v7).

Run from project root:
  python -m calibration.analyze_recorded_streams
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Top-level import fallback
try:
    from config import INV_A, B_OFFSET
    from core.kinematic_swipe import StreamKinematicSwipeDetector, V7_DEFAULT_TEMPLATES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import INV_A, B_OFFSET
    from core.kinematic_swipe import StreamKinematicSwipeDetector, V7_DEFAULT_TEMPLATES


def analyze_stream_file(filepath: Path) -> dict:
    """Perform comprehensive kinematic and statistical analysis on a single recording."""
    df = pd.read_csv(filepath)
    direction_col = df["direction"].iloc[0] if "direction" in df.columns else filepath.stem.split("_")[1].upper()
    
    still_df = df[df["phase"] == "STILL"] if "phase" in df.columns else df.iloc[:500]
    mov_df = df[df["phase"] == "MOVEMENT"] if "phase" in df.columns else df.iloc[500:]

    # Baseline statistics
    b_still = still_df[["b_raw_x", "b_raw_y", "b_raw_z"]].values
    b_mean_still = np.mean(b_still, axis=0) if len(b_still) > 0 else np.zeros(3)
    b_std_still = np.std(b_still, axis=0) if len(b_still) > 0 else np.zeros(3)

    # Movement statistics
    b_mov = mov_df[["b_raw_x", "b_raw_y", "b_raw_z"]].values
    m_mov = mov_df[["m_x", "m_y", "m_z"]].values if "m_x" in mov_df.columns else np.dot(b_mov - B_OFFSET, INV_A.T)

    cov_b = np.cov(b_mov, rowvar=False)
    eig_vals_b, eig_vecs_b = np.linalg.eigh(cov_b)

    # Unit dipole kinematics
    norm_m = np.linalg.norm(m_mov, axis=1, keepdims=True)
    u_m = m_mov / np.where(norm_m > 1e-6, norm_m, 1.0)
    du = np.diff(u_m, axis=0)
    speed = np.linalg.norm(du, axis=1) / 0.010

    # Total accumulated cross-product axis
    cross_all = np.cross(u_m[:-1], du)
    weights = np.linalg.norm(du, axis=1)[:, None]
    weighted_cross = cross_all * weights
    accum_axis = np.sum(weighted_cross, axis=0)
    mag_axis = float(np.linalg.norm(accum_axis))
    unit_axis = accum_axis / mag_axis if mag_axis > 1e-6 else np.zeros(3)

    # Replay through StreamKinematicSwipeDetector
    detector = StreamKinematicSwipeDetector()
    detected_events = []
    for _, row in mov_df.iterrows():
        b = row[["b_raw_x", "b_raw_y", "b_raw_z"]].values
        evt = detector.feed_raw(b, dt=0.010)
        if evt is not None:
            detected_events.append(evt)

    correct_swipes = sum(1 for e in detected_events if e.direction == direction_col)
    total_swipes = len(detected_events)
    accuracy = (correct_swipes / total_swipes * 100.0) if total_swipes > 0 else 0.0

    return {
        "direction": direction_col,
        "filepath": filepath.name,
        "total_samples": len(df),
        "mov_samples": len(mov_df),
        "still_samples": len(still_df),
        "b_mean_still": b_mean_still,
        "b_std_still": b_std_still,
        "b_var_mov": [np.var(b_mov[:, 0]), np.var(b_mov[:, 1]), np.var(b_mov[:, 2])],
        "normal_plane": eig_vecs_b[:, 0], # eigenvector with lowest variance
        "principal_axis": eig_vecs_b[:, 2], # eigenvector with highest variance
        "accum_unit_axis": unit_axis,
        "speed_mean": float(np.mean(speed)),
        "speed_peak": float(np.max(speed)) if len(speed) > 0 else 0.0,
        "total_swipes": total_swipes,
        "correct_swipes": correct_swipes,
        "accuracy": accuracy,
        "events": detected_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze continuous directional stream recordings")
    parser.add_argument(
        "--dir",
        type=str,
        default="recordings",
        help="Directory containing stream_*.csv recordings",
    )
    args = parser.parse_args()

    recordings_dir = Path(args.dir)
    if not recordings_dir.exists():
        print(f"[-] Recordings directory '{recordings_dir}' not found.")
        sys.exit(1)

    csv_files = sorted(list(recordings_dir.glob("stream_*.csv")))
    if not csv_files:
        print(f"[-] No 'stream_*.csv' files found in '{recordings_dir}'.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"DEEP KINEMATIC ANALYSIS OF {len(csv_files)} RECORDED STREAM DATASETS")
    print("=" * 80)

    all_results = []
    for csv_file in csv_files:
        res = analyze_stream_file(csv_file)
        all_results.append(res)

        print(f"\n[*] File: {res['filepath']} | Direction: {res['direction']}")
        print(f"   • Samples: Total={res['total_samples']:,} | Movement={res['mov_samples']:,} | Still={res['still_samples']:,}")
        print(f"   • STILL Baseline Field : [{res['b_mean_still'][0]:+6.0f}, {res['b_mean_still'][1]:+6.0f}, {res['b_mean_still'][2]:+6.0f}] mG (std={np.mean(res['b_std_still']):.1f} mG)")
        print(f"   • Kinematic Speed      : Mean={res['speed_mean']:4.1f} rad/s | Peak={res['speed_peak']:4.1f} rad/s")
        print(f"   • Accumulated Axis     : [{res['accum_unit_axis'][0]:+5.3f}, {res['accum_unit_axis'][1]:+5.3f}, {res['accum_unit_axis'][2]:+5.3f}]")
        print(f"   • Detected Swipes      : {res['total_swipes']} strokes | Correct: {res['correct_swipes']} ({res['accuracy']:.1f}%)")
        
        if res["events"]:
            print("     Individual Detected Strokes:")
            for idx, e in enumerate(res["events"][:8]):
                status = "OK" if e.direction == res["direction"] else "XX"
                print(f"       [{status}] #{idx+1:02d}: {e.direction:5s} (conf={e.confidence*100:4.1f}%, dur={e.duration_ms:3.0f}ms, arc={e.displacement_deg:4.1f}deg, spd={e.peak_speed_rad_s:4.1f} rad/s)")
            if len(res["events"]) > 8:
                print(f"       ... and {len(res['events'])-8} more strokes")

    # Global summary table
    print("\n" + "=" * 80)
    print("GLOBAL CLASSIFIER ACCURACY & PERFORMANCE SUMMARY")
    print("=" * 80)
    total_detected = sum(r["total_swipes"] for r in all_results)
    total_correct = sum(r["correct_swipes"] for r in all_results)
    overall_acc = (total_correct / total_detected * 100.0) if total_detected > 0 else 0.0

    print(f"  • Total Swipes Detected : {total_detected}")
    print(f"  • Total Swipes Correct  : {total_correct}")
    print(f"  • Overall Replay Accuracy: \033[1;32m{overall_acc:.1f}%\033[0m")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
