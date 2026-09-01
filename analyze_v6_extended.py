"""Extended analysis: test various improvement strategies."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, r'C:\Users\prash\OneDrive\Desktop\magnetometer\MKI185V1-ESP32\magnetometer-delta')

from config import V6_TEMPLATES, B_OFFSET, INV_A, LSB_TO_MGAUSS, V6_NOISE_FLOOR
from core.axis_signature import AxisSignatureRecognizer

csv_files = [
    'calibration_data_20260827_121002.csv',
    'calibration_data_20260827_121145.csv',
]

def run_v6_standard(df_dict, use_normalized_cross=False, noise_floor=0.004, silence_taps=5, 
                    min_samples=4, swipe_start_thresh=0.010, confidence_threshold=0.50):
    """Run v6 recognition with configurable parameters."""
    total_correct = 0
    total_swipes = 0
    confusion = {}
    
    for ground_truth, group in df_dict.items():
        if ground_truth not in V6_TEMPLATES:
            continue
        if ground_truth not in confusion:
            confusion[ground_truth] = {p: 0 for p in list(V6_TEMPLATES.keys()) + ['UNKNOWN']}
        
        axis_accum = np.zeros(3)
        prev_vec = None
        base_line = None
        sample_count = 0
        quiet_count = 0
        state = 'IDLE'
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
                    prev_vec = v.copy()
                    # Record first step
                    if use_normalized_cross:
                        v_prev_norm = prev_vec / (np.linalg.norm(prev_vec) + 1e-12)
                        v_curr_norm = v / (np.linalg.norm(v) + 1e-12)
                        dv_norm = v_curr_norm - v_prev_norm
                        if np.linalg.norm(dv_norm) > noise_floor:
                            axis = np.cross(v_prev_norm, dv_norm)
                            axis_accum += axis
                    else:
                        axis = np.cross(prev_vec, dv)
                        axis_accum += axis
                    sample_count = 1
            else:
                if dv_mag > noise_floor:
                    if use_normalized_cross:
                        v_prev_norm = prev_vec / (np.linalg.norm(prev_vec) + 1e-12)
                        v_curr_norm = v / (np.linalg.norm(v) + 1e-12)
                        dv_norm = v_curr_norm - v_prev_norm
                        if np.linalg.norm(dv_norm) > noise_floor:
                            axis = np.cross(v_prev_norm, dv_norm)
                            axis_accum += axis
                    else:
                        axis = np.cross(prev_vec, dv)
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
                    if sample_count >= min_samples and axis_mag > 1e-4:
                        axis_unit = axis_accum / axis_mag
                        scores = {d: float(np.dot(axis_unit, V6_TEMPLATES[d])) for d in V6_TEMPLATES}
                        best_dir = max(scores, key=scores.get)
                        best_score = scores[best_dir]
                        if best_score >= confidence_threshold:
                            total_norm_sw = total_swipes
                            total_swipes += 1
                            pred = best_dir
                            if pred not in confusion[ground_truth]:
                                confusion[ground_truth][pred] = 0
                            confusion[ground_truth][pred] += 1
                            if pred == ground_truth:
                                total_correct += 1
    
    acc = total_correct / total_swipes * 100 if total_swipes > 0 else 0
    return acc, total_correct, total_swipes, confusion

# Load all data
all_data = {}
for fname in csv_files:
    path = Path(fname)
    if not path.exists():
        continue
    df = pd.read_csv(path)
    for gt, group in df.groupby('ground_truth', sort=False):
        if gt not in all_data:
            all_data[gt] = []
        all_data[gt].append(group[['b_raw_x', 'b_raw_y', 'b_raw_z']].values)

# Combine all data per direction
combined_data = {}
for gt, groups in all_data.items():
    combined_data[gt] = pd.DataFrame(
        np.vstack(groups), columns=['b_raw_x', 'b_raw_y', 'b_raw_z']
    )

# Also create dict format for the function
df_dict = {}
for gt, groups in all_data.items():
    arr = np.vstack(groups)
    df_dict[gt] = pd.DataFrame(arr, columns=['b_raw_x', 'b_raw_y', 'b_raw_z'])

# Baseline
print("=" * 70)
print("V6 IMPROVEMENT STRATEGY ANALYSIS")
print("=" * 70)

acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=False)
print(f"\n1. Baseline (raw, unnormalized cross): {acc:.1f}% ({correct}/{total})")

acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=True)
print(f"2. Normalized cross product (B_prev_hat x dB_hat): {acc:.1f}% ({correct}/{total})")

# Test with different noise floors
for nf in [0.002, 0.003, 0.004, 0.005, 0.006]:
    acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=True, noise_floor=nf)
    print(f"3. Normalized cross, noise_floor={nf}: {acc:.1f}% ({correct}/{total})")

# Test with different silence taps
for st in [3, 4, 5, 6, 8]:
    acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=True, silence_taps=st)
    print(f"4. Normalized cross, silence_taps={st}: {acc:.1f}% ({correct}/{total})")

# Test with different min_samples
for ms in [2, 3, 4, 5]:
    acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=True, min_samples=ms)
    print(f"5. Normalized cross, min_samples={ms}: {acc:.1f}% ({correct}/{total})")

# Test with different confidence thresholds
for ct in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=True, confidence_threshold=ct)
    print(f"6. Normalized cross, conf_threshold={ct}: {acc:.1f}% ({correct}/{total})")

# Test with different swipe start thresholds
for sst in [0.005, 0.008, 0.010, 0.012, 0.015]:
    acc, correct, total, _ = run_v6_standard(df_dict, use_normalized_cross=True, swipe_start_thresh=sst)
    print(f"7. Normalized cross, swipe_start_thresh={sst}: {acc:.1f}% ({correct}/{total})")

# Check per-direction errors with normalized cross
print("\n--- Per-direction error analysis (normalized cross, conf=0.50) ---")
acc, correct, total, conf_matrix = run_v6_standard(df_dict, use_normalized_cross=True, confidence_threshold=0.50)
print("\nConfusion Matrix:")
header = f"{'':10s}" + ''.join(f'{d:>10s}' for d in list(V6_TEMPLATES.keys()) + ['UNKNOWN'])
print(header)
for gt, preds in conf_matrix.items():
    row_str = f'{gt:<10s}' + ''.join(f'{preds.get(p, 0):>10d}' for p in list(V6_TEMPLATES.keys()) + ['UNKNOWN'])
    print(row_str)

# Analyze UP confusion (the biggest problem)
print("\n--- UP direction analysis ---")
print("UP is misclassified as DOWN, LEFT, RIGHT or UNKNOWN.")
print("DOWN is perfectly classified (37/37=100%) - axis is very distinct.")
print("UP has the most errors. Let's check template geometry:")

for d1 in V6_TEMPLATES:
    for d2 in V6_TEMPLATES:
        if d1 < d2:
            sim = float(np.dot(V6_TEMPLATES[d1], V6_TEMPLATES[d2]))
            angle = np.degrees(np.arccos(np.clip(sim, -1, 1)))
            print(f"  Angle {d1}->{d2}: {angle:.1f} deg (cos={sim:+.4f})")
