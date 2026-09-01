"""Deep analysis: arc-angle weighting and template quality."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, r'C:\Users\prash\OneDrive\Desktop\magnetometer\MKI185V1-ESP32\magnetometer-delta')

from config import V6_TEMPLATES, B_OFFSET, INV_A, LSB_TO_MGAUSS

csv_files = [
    'calibration_data_20260827_121002.csv',
    'calibration_data_20260827_121145.csv',
]

# Load all data per direction
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

df_dict = {}
for gt, groups in all_data.items():
    arr = np.vstack(groups)
    df_dict[gt] = pd.DataFrame(arr, columns=['b_raw_x', 'b_raw_y', 'b_raw_z'])

def run_v6_analysis(df_dict, use_normalized=True, use_arc_angle_weighting=False, 
                     confidence_threshold=0.50, noise_floor=0.004, silence_taps=5, min_samples=4):
    total_correct = 0
    total_swipes = 0
    confusion = {}
    per_dir_correct = {}
    per_dir_total = {}
    
    for ground_truth, group in df_dict.items():
        if ground_truth not in V6_TEMPLATES:
            continue
        if ground_truth not in confusion:
            confusion[ground_truth] = {p: 0 for p in list(V6_TEMPLATES.keys()) + ['UNKNOWN']}
            per_dir_correct[ground_truth] = 0
            per_dir_total[ground_truth] = 0
        
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
                    base_line = 0.97 * base_line + 0.03 * v
                    prev_vec = v.copy()
                elif dist_from_baseline > 0.010 or dv_mag > noise_floor * 1.5:
                    state = 'ACTIVE'
                    axis_accum = np.zeros(3)
                    sample_count = 0
                    quiet_count = 0
                    prev_vec = v.copy()
                    if use_normalized:
                        v_prev_u = prev_vec / (np.linalg.norm(prev_vec) + 1e-12)
                        v_curr_u = v / (np.linalg.norm(v) + 1e-12)
                        dv_u = v_curr_u - v_prev_u
                        dv_u_mag = np.linalg.norm(dv_u)
                        if dv_u_mag > noise_floor:
                            axis = np.cross(v_prev_u, dv_u)
                            if use_arc_angle_weighting:
                                weight = np.arcsin(min(dv_u_mag, 1.0))
                                axis = axis / dv_u_mag * weight  # scale by arc angle
                            axis_accum += axis
                    else:
                        axis = np.cross(prev_vec, dv)
                        axis_accum += axis
                    sample_count = 1
            else:
                if dv_mag > noise_floor:
                    if use_normalized:
                        v_prev_u = prev_vec / (np.linalg.norm(prev_vec) + 1e-12)
                        v_curr_u = v / (np.linalg.norm(v) + 1e-12)
                        dv_u = v_curr_u - v_prev_u
                        dv_u_mag = np.linalg.norm(dv_u)
                        if dv_u_mag > noise_floor:
                            axis = np.cross(v_prev_u, dv_u)
                            if use_arc_angle_weighting:
                                weight = np.arcsin(min(dv_u_mag, 1.0))
                                axis = axis / dv_u_mag * weight
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
                            total_swipes += 1
                            per_dir_total[ground_truth] += 1
                            pred = best_dir
                            if pred not in confusion[ground_truth]:
                                confusion[ground_truth][pred] = 0
                            confusion[ground_truth][pred] += 1
                            if pred == ground_truth:
                                total_correct += 1
                                per_dir_correct[ground_truth] += 1
    
    acc = total_correct / total_swipes * 100 if total_swipes > 0 else 0
    return acc, total_correct, total_swipes, confusion, per_dir_correct, per_dir_total

# Test configurations
print("=" * 70)
print("ARC-ANGLE WEIGHTING & DEEP ANALYSIS")
print("=" * 70)

configs = [
    ("Raw cross (baseline)", False, False),
    ("Normalized cross", True, False),
    ("Normalized + arc-angle weighting", True, True),
]

for label, use_norm, use_arc in configs:
    for ct in [0.50, 0.55, 0.60, 0.65, 0.70]:
        acc, correct, total, conf, pdc, pdt = run_v6_analysis(
            df_dict, use_normalized=use_norm, use_arc_angle_weighting=use_arc, 
            confidence_threshold=ct
        )
        print(f"\n{label}, conf={ct}: {acc:.1f}% ({correct}/{total})")
        print(f"  Per-direction: ", end="")
        for d in V6_TEMPLATES:
            if d in pdt and pdt[d] > 0:
                print(f"{d}={pdc[d]}/{pdt[d]}({pdc[d]/pdt[d]*100:.0f}%) ", end="")
        print()

# Now check what the optimal threshold is for just counting correct at various thresholds
print("\n--- Confusion detail for best config (Normalized + arc, conf=0.6) ---")
acc, correct, total, conf_matrix, pdc, pdt = run_v6_analysis(df_dict, use_normalized=True, use_arc_angle_weighting=True, confidence_threshold=0.6)
print(f"Overall: {acc:.1f}% ({correct}/{total})")
print("\nConfusion Matrix:")
header = f"{'':10s}" + ''.join(f'{d:>10s}' for d in list(V6_TEMPLATES.keys()) + ['UNKNOWN'])
print(header)
for gt, preds in conf_matrix.items():
    row_str = f'{gt:<10s}' + ''.join(f'{preds.get(p, 0):>10d}' for p in list(V6_TEMPLATES.keys()) + ['UNKNOWN'])
    print(row_str)

# The key issue: UP is confused with DOWN, LEFT, RIGHT
# DOWN . UP = -0.9273 (angle 158 deg) -- these should be ~180 deg (antipodal)
# This means the templates are not perfectly antipodal for UP/DOWN pair
# Also DOWN.R = +0.2987 means DOWN and RIGHT share 72.6 deg - not orthogonal

# Check: what if we retrain with strict symmetry?
print("\n--- Template Geometry Issues ---")
print("Current V6_TEMPLATES have non-ideal geometry:")
print(f"  UP . DOWN = {float(np.dot(V6_TEMPLATES['UP'], V6_TEMPLATES['DOWN'])):+.4f} (ideal: -1.0, antipodal)")
print(f"  DOWN . RIGHT = {float(np.dot(V6_TEMPLATES['DOWN'], V6_TEMPLATES['RIGHT'])):+.4f} (ideal: 0.0, orthogonal)")
print(f"  DOWN . LEFT = {float(np.dot(V6_TEMPLATES['DOWN'], V6_TEMPLATES['LEFT'])):+.4f} (ideal: 0.0, orthogonal)")
print(f"  UP . LEFT = {float(np.dot(V6_TEMPLATES['UP'], V6_TEMPLATES['LEFT'])):+.4f} (ideal: 0.0, orthogonal)")

# Try with orthogonalized templates
from core.calibration import orthogonalize_cardinal_templates
ortho_templates = orthogonalize_cardinal_templates(V6_TEMPLATES)

print("\n--- Orthogonalized Templates ---")
for d1 in V6_TEMPLATES:
    for d2 in V6_TEMPLATES:
        if d1 < d2:
            sim_orig = float(np.dot(V6_TEMPLATES[d1], V6_TEMPLATES[d2]))
            sim_ortho = float(np.dot(ortho_templates[d1], ortho_templates[d2]))
            angle_orig = np.degrees(np.clip(np.arccos(sim_orig), 0, np.pi))
            angle_ortho = np.degrees(np.clip(np.arccos(sim_ortho), 0, np.pi))
            print(f"  {d1}->{d2}: {angle_orig:.1f}deg -> {angle_ortho:.1f}deg (sim {sim_orig:+.4f} -> {sim_ortho:+.4f})")

# Test with orthogonal templates
print("\n--- Testing with Orthogonalized Templates ---")
import copy
# Temporarily use orthogonal templates
original_templates = V6_TEMPLATES
import config as config_module
config_module.V6_TEMPLATES = ortho_templates

# Reload the module to pick up new templates
import importlib
import core.axis_signature
importlib.reload(core.axis_signature)
from core.axis_signature import AxisSignatureRecognizer as ASR_new

for ct in [0.50, 0.60, 0.70]:
    total_correct = 0
    total_swipes = 0
    for ground_truth, group in df_dict.items():
        if ground_truth not in V6_TEMPLATES:
            continue
        recognizer = ASR_new(use_unwarped=True, confidence_threshold=ct)
        b_raw_arr = group[['b_raw_x', 'b_raw_y', 'b_raw_z']].values
        for b_raw in b_raw_arr:
            b_raw_mg = b_raw * LSB_TO_MGAUSS
            event = recognizer.feed(b_raw_mg)
            if event is not None:
                total_swipes += 1
                if event.direction == ground_truth:
                    total_correct += 1
    print(f"  Orthogonal templates, conf={ct}: {total_correct/total_swipes*100:.1f}% ({total_correct}/{total_swipes})")

# Restore
config_module.V6_TEMPLATES = original_templates
importlib.reload(core.axis_signature)
