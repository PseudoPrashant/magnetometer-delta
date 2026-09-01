"""Analysis script for tracker_v6 accuracy."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, r'C:\Users\prash\OneDrive\Desktop\magnetometer\MKI185V1-ESP32\magnetometer-delta')

from config import V6_TEMPLATES, B_OFFSET, INV_A, LSB_TO_MGAUSS
from core.axis_signature import AxisSignatureRecognizer

csv_files = [
    'calibration_data_20260827_121002.csv',
    'calibration_data_20260827_121145.csv',
]

total_correct = 0
total_swipes = 0
confusion = {}
confidences = {d: [] for d in V6_TEMPLATES}
sample_counts = {d: [] for d in V6_TEMPLATES}
magnitudes = {d: [] for d in V6_TEMPLATES}

for fname in csv_files:
    path = Path(fname)
    if not path.exists():
        print(f'File not found: {fname}')
        continue
    df = pd.read_csv(path)
    for ground_truth, group in df.groupby('ground_truth', sort=False):
        if ground_truth not in V6_TEMPLATES:
            continue
        if ground_truth not in confusion:
            confusion[ground_truth] = {p: 0 for p in list(V6_TEMPLATES.keys()) + ['UNKNOWN']}

        recognizer = AxisSignatureRecognizer(use_unwarped=True, confidence_threshold=0.50)
        b_raw_arr = group[['b_raw_x', 'b_raw_y', 'b_raw_z']].values
        for b_raw in b_raw_arr:
            b_raw_mg = b_raw * LSB_TO_MGAUSS
            event = recognizer.feed(b_raw_mg)
            if event is not None:
                total_swipes += 1
                pred = event.direction
                if pred not in confusion[ground_truth]:
                    confusion[ground_truth][pred] = 0
                confusion[ground_truth][pred] += 1
                if pred == ground_truth:
                    total_correct += 1
                confidences[ground_truth].append(event.confidence)
                sample_counts[ground_truth].append(event.sample_count)
                magnitudes[ground_truth].append(event.axis_magnitude)

print('=' * 70)
print('DETAILED V6 ACCURACY ANALYSIS')
print('=' * 70)
print(f'')
print(f'Total swipes: {total_swipes}, Correct: {total_correct}, Accuracy: {total_correct/total_swipes*100:.1f}%')

print('')
print('--- Confusion Matrix ---')
header = f"{'':10s}" + ''.join(f'{d:>10s}' for d in list(V6_TEMPLATES.keys()) + ['UNKNOWN'])
print(header)
for gt, preds in confusion.items():
    row_str = f'{gt:<10s}' + ''.join(f'{preds.get(p, 0):>10d}' for p in list(V6_TEMPLATES.keys()) + ['UNKNOWN'])
    print(row_str)

print('')
print('--- Per-Direction Confidence Distribution ---')
for d in V6_TEMPLATES:
    confs = confidences[d]
    if confs:
        confs_arr = np.array(confs)
        print(f'')
        print(f'  {d}: {len(confs)} swipes')
        print(f'    Confidence: mean={np.mean(confs_arr):.3f}, min={np.min(confs_arr):.3f}, max={np.max(confs_arr):.3f}')
        if sample_counts[d]:
            sc = np.array(sample_counts[d])
            print(f'    Samples: mean={np.mean(sc):.1f}, min={np.min(sc)}, max={np.max(sc)}')
        if magnitudes[d]:
            mg = np.array(magnitudes[d])
            print(f'    Axis magnitude: mean={np.mean(mg):.6f}, min={np.min(mg):.6f}, max={np.max(mg):.6f}')

print('')
print('--- Template Similarity (should be orthogonal ~0) ---')
for d1 in V6_TEMPLATES:
    for d2 in V6_TEMPLATES:
        if d1 < d2:
            dot_val = float(np.dot(V6_TEMPLATES[d1], V6_TEMPLATES[d2]))
            print(f'  {d1} . {d2} = {dot_val:+.4f}')

# Test with normalized cross product
print('')
print('--- Testing Normalized Cross Product Impact ---')
for fname in csv_files:
    path = Path(fname)
    if not path.exists():
        continue
    df = pd.read_csv(path)

    correct_norm = 0
    total_norm = 0

    for ground_truth, group in df.groupby('ground_truth', sort=False):
        if ground_truth not in V6_TEMPLATES:
            continue
        axis_accum = np.zeros(3)
        prev_vec = None
        base_line = None
        sample_count = 0
        quiet_count = 0
        state = 'IDLE'
        swipe_start_thresh = 0.010
        noise_floor = 0.004
        silence_taps = 5

        b_raw_arr = group[['b_raw_x', 'b_raw_y', 'b_raw_z']].values
        for b_raw in b_raw_arr:
            b_raw_mg = b_raw * LSB_TO_MGAUSS
            v = np.dot(INV_A, b_raw_mg - B_OFFSET)
            if base_line is None:
                base_line = v.copy()
                prev_vec = v.copy()
                continue

            dv = v - prev_vec
            dv_mag = np.linalg.norm(dv)
            dist_from_baseline = np.linalg.norm(v - base_line)

            if state == 'IDLE':
                if dv_mag < noise_floor:
                    base_line = 0.96 * base_line + 0.04 * v
                    prev_vec = v.copy()
                elif dist_from_baseline > swipe_start_thresh or dv_mag > noise_floor * 1.5:
                    state = 'ACTIVE'
                    axis_accum = np.zeros(3)
                    sample_count = 0
                    quiet_count = 0
                    v_prev_norm = prev_vec / (np.linalg.norm(prev_vec) + 1e-12)
                    v_curr_norm = v / (np.linalg.norm(v) + 1e-12)
                    dv_norm = v_curr_norm - v_prev_norm
                    if np.linalg.norm(dv_norm) > noise_floor:
                        axis = np.cross(v_prev_norm, dv_norm)
                        axis_accum += axis
                        sample_count = 1
                    prev_vec = v.copy()
            else:
                if dv_mag > noise_floor:
                    v_prev_norm = prev_vec / (np.linalg.norm(prev_vec) + 1e-12)
                    v_curr_norm = v / (np.linalg.norm(v) + 1e-12)
                    dv_norm = v_curr_norm - v_prev_norm
                    if np.linalg.norm(dv_norm) > noise_floor:
                        axis = np.cross(v_prev_norm, dv_norm)
                        axis_accum += axis
                        sample_count += 1
                        quiet_count = 0
                    prev_vec = v.copy()
                else:
                    quiet_count += 1

                if quiet_count >= silence_taps or sample_count >= 80:
                    state = 'IDLE'
                    base_line = v.copy()
                    prev_vec = v.copy()

                    axis_mag = np.linalg.norm(axis_accum)
                    if sample_count >= 4 and axis_mag > 1e-4:
                        axis_unit = axis_accum / axis_mag
                        scores = {d: float(np.dot(axis_unit, V6_TEMPLATES[d])) for d in V6_TEMPLATES}
                        best_dir = max(scores, key=scores.get)
                        best_score = scores[best_dir]
                        if best_score >= 0.50:
                            total_norm += 1
                            if best_dir == ground_truth:
                                correct_norm += 1

    if total_norm > 0:
        print(f'  {fname}: Normalized cross-product accuracy = {correct_norm/total_norm*100:.1f}% ({correct_norm}/{total_norm})')
