"""tracker_v2.py - Live Magnetic Sphere Rolling Trajectory (v2 pipeline).

Improvements over tracker_v1:

    [IIS2MDC Hardware @ 100 Hz / 400 kHz I2C]
        |
        v
    1-6. Shared signal chain (see core/pipeline.py)
    7. Velocity Ballistics   (gain grows with angular speed)
    8. Spatial Deadzone      (fractional accumulation - no lost motion)
    9. Dominant Axis         (per frame: keep larger of |dX|,|dY|, discard rest)

Run from the project root:  python -m trackers.tracker_v2
Press 'C' in the plot window to clear the trace.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from config import (
    BAUD_RATE,
    DOMINANT_AXIS,
    LSB_TO_MGAUSS,
    MAX_HISTORY,
    SERIAL_PORT,
    SPATIAL_DEADZONE,
    VIEW_LIMIT,
)
from core.pipeline import RotationPipeline, ballistic_gain

# ==========================================
# GLOBAL STATE & INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Press 'C' in the window to clear trace.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

pipeline = RotationPipeline()

# Fractional accumulation residues (Stage 8)
frac_x = 0.0
frac_y = 0.0

# Trajectory tracking state
pos_x = 0.0
pos_y = 0.0
traj_x = [0.0]
traj_y = [0.0]


def reset_state() -> None:
    """Clear trajectory AND all pipeline state so tracking restarts cleanly."""
    global frac_x, frac_y, pos_x, pos_y, traj_x, traj_y
    pipeline.reset()
    frac_x = frac_y = 0.0
    pos_x = pos_y = 0.0
    traj_x = [0.0]
    traj_y = [0.0]
    print("[*] Trace cleared.")


# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
line_traj, = ax.plot([], [], 'b-', lw=1.5, alpha=0.7, label='Trajectory')
current_dot, = ax.plot([], [], 'ro', markersize=8, label='Current Pos')

ax.set_title("Live Magnetic Sphere Rolling Trajectory (v2)", fontsize=14, fontweight='bold')
ax.set_xlabel("X Deflection (Integrated dTheta_Y)", fontsize=11)
ax.set_ylabel("Y Deflection (Integrated dTheta_X)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(0, color='gray', lw=0.8)
ax.legend(loc='upper right')

ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)

fig.canvas.mpl_connect('key_press_event', lambda event: reset_state()
                       if event.key in ('c', 'C') else None)


# ==========================================
# ANIMATION LOOP
# ==========================================
def update(frame: int):
    """Drain the serial buffer, integrate motion, refresh the plot."""
    global frac_x, frac_y, pos_x, pos_y, traj_x, traj_y

    # Drain serial buffer to process all incoming packets
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 3:
            continue

        try:
            b_raw = np.array([float(p) for p in parts]) * LSB_TO_MGAUSS
        except ValueError:
            continue
        if not np.all(np.isfinite(b_raw)):
            continue

        fed = pipeline.feed(b_raw)   # stages 1-6
        if fed is None:
            continue
        d_theta, dt = fed

        # ---- Stage 7: Dynamic Velocity Ballistics ----
        gain = ballistic_gain(np.linalg.norm(d_theta) / dt)

        # ---- Stage 8: Spatial Deadzone & Fractional Accumulation ----
        # Sub-threshold motion is banked in residues instead of discarded,
        # so slow deliberate rolls eventually move the pointer while
        # incoherent sensor noise never crosses the threshold.
        step_x = d_theta[1] * gain
        step_y = d_theta[0] * gain

        if DOMINANT_AXIS:
            if abs(step_x) >= abs(step_y):
                step_y = 0.0
            else:
                step_x = 0.0

        frac_x += step_x
        frac_y += step_y

        moved = False
        if abs(frac_x) >= SPATIAL_DEADZONE:
            pos_x += frac_x
            frac_x = 0.0
            moved = True
        if abs(frac_y) >= SPATIAL_DEADZONE:
            pos_y += frac_y
            frac_y = 0.0
            moved = True

        if moved:
            traj_x.append(pos_x)
            traj_y.append(pos_y)
            if len(traj_x) > MAX_HISTORY:
                traj_x.pop(0)
                traj_y.pop(0)

    # Update plot lines
    if traj_x:
        line_traj.set_data(traj_x, traj_y)
        current_dot.set_data([pos_x], [pos_y])

        # Dynamic autoscaling if trajectory goes out of bounds
        max_bound = max(abs(pos_x), abs(pos_y), VIEW_LIMIT) * 1.2
        if max_bound > ax.get_xlim()[1]:
            ax.set_xlim(-max_bound, max_bound)
            ax.set_ylim(-max_bound, max_bound)

    return line_traj, current_dot


ani = FuncAnimation(fig, update, interval=20, blit=False)  # noqa: F841 - keeps animation alive
plt.show()

ser.close()
