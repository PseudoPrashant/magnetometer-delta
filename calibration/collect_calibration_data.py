"""collect_calibration_data.py - v5 Pipeline Calibration Data Collector.

Standalone tool that instruments every intermediate stage of the RotationPipelineV4
and logs them to CSV for offline analysis.

Guided 4-phase protocol (75 s total):
  Phase 1: Roll magnet UP    for 15 s  →  5 s pause
  Phase 2: Roll magnet DOWN  for 15 s  →  5 s pause
  Phase 3: Roll magnet LEFT  for 15 s  →  5 s pause
  Phase 4: Roll magnet RIGHT for 15 s

Run from the project root:  python -m calibration.collect_calibration_data

CSV columns:
  timestamp_ms, sample_index,
  b_raw_{x,y,z},  b_filt_{x,y,z},  m_{x,y,z},
  dtheta_{x,y,z},  dt,  omega,  phi_deg,
  predicted_direction,  ground_truth
"""

import atexit
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from config import BAUD_RATE, B_OFFSET, INV_A, LSB_TO_MGAUSS, SERIAL_PORT
from core.pipeline import RotationPipelineV4, ballistic_gain, NORM_EPS

# ==========================================
# CONSTANTS
# ==========================================
PHASE_DURATION: float = 15.0        # seconds per direction
PHASES: list[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
DEADZONE_THRESHOLD: float = 0.003   # rad — minimum ‖d_theta‖ for classification
PAUSE_DURATION: float = 5.0         # seconds of rest between each phase
LIMIT0: float = 50.0
TRAIL_MAX: int = 600

CSV_COLUMNS: list[str] = [
    "timestamp_ms", "sample_index",
    "b_raw_x", "b_raw_y", "b_raw_z",
    "b_filt_x", "b_filt_y", "b_filt_z",
    "m_x", "m_y", "m_z",
    "dtheta_x", "dtheta_y", "dtheta_z",
    "dt", "omega", "phi_deg",
    "predicted_direction", "ground_truth",
]

# ==========================================
# SERIAL & PIPELINE
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

pipeline = RotationPipelineV4()

# ==========================================
# DIRECTION CLASSIFIER  (mirrors tracker_v5)
# ==========================================
_Q1 = math.pi / 4          #  45°
_Q2 = 3.0 * math.pi / 4    # 135°
_Q3 = -math.pi / 4         # -45°
_Q4 = -3.0 * math.pi / 4   # -135°


def classify_direction(d_theta: np.ndarray) -> str:
    """Map d_theta to a cardinal swipe label via atan2 heading angle."""
    phi = math.atan2(float(d_theta[0]), float(d_theta[1]))
    if _Q3 <= phi < _Q1:
        return "RIGHT"
    if _Q1 <= phi <= _Q2:
        return "UP"
    if phi > _Q2 or phi < _Q4:
        return "LEFT"
    return "DOWN"

# ==========================================
# COLLECTION STATE
# ==========================================
rows: list[dict] = []
sample_index: int = 0
current_phase_idx: int = 0
collection_start: float = 0.0
collection_active: bool = False
phase_announced: bool = False
csv_saved: bool = False
_prev_seg_idx: int = -1
_prev_pausing: bool | None = None

# Trail accumulators (transposed mapping, same as tracker_v5)
acc_x: float = 0.0
acc_y: float = 0.0
trail: list[tuple[float, float]] = []


def save_csv() -> str:
    """Write all collected rows to a timestamped CSV file."""
    global csv_saved
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"calibration_data_{ts}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    csv_saved = True
    print(f"[+] Saved {len(rows)} rows -> {path}")
    return str(path)


def _save_on_exit() -> None:
    """Safety net: save CSV if the window was closed before normal completion."""
    if rows and not csv_saved:
        save_csv()


atexit.register(_save_on_exit)

# ==========================================
# MATPLOTLIB — LIVE XY TRAIL + HUD
# ==========================================
fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("v5 Calibration Data Collection", fontsize=13, fontweight="bold")
ax.set_title("XY Trail (transposed mapping)", fontsize=10)
ax.set_xlabel("Y deflection (dTheta_y x gain)", fontsize=9)
ax.set_ylabel("X deflection (dTheta_x x gain)", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.5)
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlim(-LIMIT0, LIMIT0)
ax.set_ylim(-LIMIT0, LIMIT0)

trail_line, = ax.plot([], [], "b-", lw=1.2, alpha=0.7)
trail_dot, = ax.plot([], [], "ro", markersize=6)

hud = ax.text(
    0.03, 0.97, "", transform=ax.transAxes,
    fontsize=9, verticalalignment="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
              edgecolor="#cccccc"),
)


def _refresh_trail() -> None:
    """Redraw the XY trail with hysteresis autoscale."""
    if not trail:
        return
    xs, ys = zip(*trail)
    trail_line.set_data(xs, ys)
    trail_dot.set_data([xs[-1]], [ys[-1]])
    target = max(max(abs(v) for pt in trail for v in pt) * 1.2, LIMIT0 * 0.2)
    cur = ax.get_xlim()[1]
    if target > cur or target < 0.5 * cur:
        ax.set_xlim(-target, target)
        ax.set_ylim(-target, target)


# ==========================================
# ANIMATION LOOP
# ==========================================
def update(frame: int):
    global sample_index, current_phase_idx, collection_active, phase_announced
    global acc_x, acc_y, _prev_seg_idx, _prev_pausing

    if not collection_active:
        return

    now = time.time()
    elapsed = now - collection_start
    segment = PHASE_DURATION + PAUSE_DURATION
    total_duration = (len(PHASES) * PHASE_DURATION
                      + (len(PHASES) - 1) * PAUSE_DURATION)

    # ---- Collection complete? ----
    if elapsed >= total_duration:
        collection_active = False
        save_csv()
        print("[*] Collection complete.")
        fig.suptitle("v5 Calibration — COMPLETE", fontsize=13,
                      fontweight="bold", color="green")
        return

    # ---- Determine current segment ----
    seg_idx = min(int(elapsed / segment), len(PHASES) - 1)
    seg_offset = elapsed - seg_idx * segment
    is_pausing = seg_offset >= PHASE_DURATION and seg_idx < len(PHASES) - 1

    # ---- Detect state transitions ----
    if seg_idx != _prev_seg_idx or is_pausing != _prev_pausing:
        phase_announced = False
        _prev_seg_idx = seg_idx
        _prev_pausing = is_pausing

    # ============================================
    # PAUSE PHASE — drain serial, don't collect
    # ============================================
    if is_pausing:
        if not phase_announced:
            nxt = PHASES[min(seg_idx + 1, len(PHASES) - 1)]
            print(f"[*] PAUSE — stop rolling. Next: {nxt} in {PAUSE_DURATION:.0f}s")
            phase_announced = True

        while ser.in_waiting:
            ser.readline()

        pause_remaining = segment - seg_offset
        bars = ""
        for i, name in enumerate(PHASES):
            if i <= seg_idx:
                bars += f"  [{name}] done\n"
            elif i == seg_idx + 1:
                bars += f"  [{name}] next\n"
            else:
                bars += f"  [     ]\n"

        hud.set_text(
            f"PAUSE — stop rolling\n"
            f"Next phase in {pause_remaining:.0f}s\n"
            f"Samples: {sample_index}\n"
            f"{bars}"
        )
        hud.get_bbox_patch().set_facecolor("#fff3cd")
        _refresh_trail()
        return

    # ============================================
    # ROLL PHASE — collect and log data
    # ============================================
    current_phase_idx = seg_idx
    if not phase_announced:
        print(f"[*] Phase {current_phase_idx + 1}/4: "
              f"{PHASES[current_phase_idx]}"
              f" — roll the magnet "
              f"{PHASES[current_phase_idx].lower()}ward"
              f" ({PHASE_DURATION:.0f}s)")
        phase_announced = True

    ground_truth = PHASES[current_phase_idx]
    roll_elapsed = seg_offset
    roll_remaining = PHASE_DURATION - roll_elapsed

    # ---- Drain serial buffer ----
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            b_raw = np.array([float(p) for p in parts]) * LSB_TO_MGAUSS
        except ValueError:
            continue
        if not np.all(np.isfinite(b_raw)):
            continue

        sample_index += 1
        timestamp_ms: float = elapsed * 1000.0

        # ---- Feed through pipeline ----
        fed = pipeline.feed(b_raw)

        # ---- Extract intermediate pipeline state ----
        b_filt = (pipeline._b_filtered.copy()
                  if pipeline._b_filtered is not None
                  else np.full(3, float("nan")))

        m = INV_A @ (b_filt - B_OFFSET)
        norm = float(np.linalg.norm(m))
        m_unit = m / norm if norm >= NORM_EPS else np.full(3, float("nan"))

        # ---- Branch A: direction classification ----
        if fed is not None:
            d_theta, dt = fed
            omega = float(np.linalg.norm(d_theta) / dt)
            phi_deg = math.degrees(
                math.atan2(float(d_theta[0]), float(d_theta[1]))
            )
            norm_dt = float(np.linalg.norm(d_theta))
            predicted = (classify_direction(d_theta)
                         if norm_dt > DEADZONE_THRESHOLD else "")

            # ---- Branch B: transposed accumulation for trail ----
            gain = ballistic_gain(omega)
            acc_x += d_theta[1] * gain
            acc_y += d_theta[0] * gain
            trail.append((acc_x, acc_y))
            if len(trail) > TRAIL_MAX:
                trail.pop(0)
        else:
            d_theta = np.full(3, float("nan"))
            dt = omega = phi_deg = float("nan")
            predicted = ""

        # ---- Build CSV row ----
        def _f(v: float) -> str:
            return f"{v:.6f}" if math.isfinite(v) else ""

        def _fv(a: np.ndarray) -> tuple[str, str, str]:
            return tuple(_f(float(v)) for v in a)  # type: ignore[return-value]

        bx, by, bz = _fv(b_filt)
        mx, my, mz = _fv(m_unit)
        dx, dy, dz = _fv(d_theta)

        rows.append({
            "timestamp_ms": f"{timestamp_ms:.1f}",
            "sample_index": sample_index,
            "b_raw_x": f"{b_raw[0]:.2f}",
            "b_raw_y": f"{b_raw[1]:.2f}",
            "b_raw_z": f"{b_raw[2]:.2f}",
            "b_filt_x": bx, "b_filt_y": by, "b_filt_z": bz,
            "m_x": mx, "m_y": my, "m_z": mz,
            "dtheta_x": dx, "dtheta_y": dy, "dtheta_z": dz,
            "dt": _f(dt),
            "omega": _f(omega),
            "phi_deg": (f"{phi_deg:.2f}" if math.isfinite(phi_deg)
                        else ""),
            "predicted_direction": predicted,
            "ground_truth": ground_truth,
        })

    # ---- Update display ----
    _refresh_trail()

    bars = ""
    for i, name in enumerate(PHASES):
        if i < current_phase_idx:
            bars += f"  [{name}] done\n"
        elif i == current_phase_idx:
            pct = int(roll_elapsed / PHASE_DURATION * 100)
            filled = pct // 5
            bars += (f"  [{name}] "
                     f"{'#' * filled}{'.' * (20 - filled)} "
                     f"{pct:3d}%\n")
        else:
            bars += f"  [     ]\n"

    hud.set_text(
        f"Phase {current_phase_idx + 1}/4: "
        f"{PHASES[current_phase_idx]}\n"
        f"Roll: {roll_remaining:.0f}s"
        f" | Total: {max(0.0, total_duration - elapsed):.0f}s\n"
        f"Samples: {sample_index}\n"
        f"{bars}"
    )
    hud.get_bbox_patch().set_facecolor("white")

    # Heartbeat to console
    if frame % 50 == 0:
        print(f"[.] phase={PHASES[current_phase_idx]}  "
              f"roll={roll_remaining:.0f}s  samples={sample_index}")


ani = FuncAnimation(fig, update, interval=20, blit=False,  # noqa: F841
                    cache_frame_data=False)


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    print("=" * 56)
    print("  v5 PIPELINE CALIBRATION DATA COLLECTOR")
    print("=" * 56)
    print()
    print("Collects every intermediate pipeline stage to CSV.")
    print()
    print("Protocol (75 s total):")
    for i, name in enumerate(PHASES, 1):
        pause = "  → 5 s pause" if i < len(PHASES) else ""
        print(f"  Phase {i}: Roll magnet {name.lower()}ward"
              f" for {PHASE_DURATION:.0f}s{pause}")
    print()
    print("Keep metal / watches away from the sensor.")
    print("Press ENTER to start collection...")
    input()

    global collection_start, collection_active, phase_announced, current_phase_idx
    print("[*] Starting in 3 s — get ready to roll UP...")
    for i in range(3, 0, -1):
        print(f"    {i}")
        time.sleep(1)

    collection_start = time.time()
    collection_active = True
    phase_announced = False
    current_phase_idx = 0

    plt.show()
    ser.close()


if __name__ == "__main__":
    main()
