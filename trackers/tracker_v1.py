"""tracker_v1.py - Live Magnetic Sphere Rolling Trajectory (v1 baseline).

Simplest working tracker:

    EMA smoothing -> baseline subtraction -> geometry unwarp -> normalize
    -> cross-product delta -> fixed-gain integration with an angular deadzone.

Run from the project root:  python -m trackers.tracker_v1
Press 'C' in the plot window to clear the trace.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from config import (
    BAUD_RATE,
    B_OFFSET,
    EMA_ALPHA,
    INV_A,
    LSB_TO_MGAUSS,
    MAX_HISTORY,
    POINTER_GAIN,
    ROTATION_DEADZONE,
    SERIAL_PORT,
    VIEW_LIMIT,
)
from core.filters import ExponentialMovingAverage

# ==========================================
# GLOBAL STATE & INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Close the window or press 'C' to clear trace.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

ema = ExponentialMovingAverage()

# Trajectory tracking state
pos_x = 0.0
pos_y = 0.0
traj_x = [0.0]
traj_y = [0.0]

m_prev: np.ndarray | None = None


def reset_state() -> None:
    """Clear trajectory AND filter state so tracking restarts cleanly."""
    global pos_x, pos_y, traj_x, traj_y, m_prev
    ema.reset()
    pos_x, pos_y = 0.0, 0.0
    traj_x = [0.0]
    traj_y = [0.0]
    m_prev = None
    print("[*] Trace cleared.")


# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
line_traj, = ax.plot([], [], 'b-', lw=1.5, alpha=0.7, label='Trajectory')
current_dot, = ax.plot([], [], 'ro', markersize=8, label='Current Pos')

ax.set_title("Live Magnetic Sphere Rolling Trajectory", fontsize=14, fontweight='bold')
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
    global pos_x, pos_y, traj_x, traj_y, m_prev

    # Drain serial buffer to process all incoming packets
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 3:
            continue

        try:
            # 1. Convert raw LSB to mGauss
            b_raw = np.array([float(p) for p in parts]) * LSB_TO_MGAUSS
        except ValueError:
            continue

        # 2. Low-Pass EMA Filter
        b_filtered = ema.filter(b_raw, EMA_ALPHA)

        # 3+4. Ambient Subtraction & Inverse Geometry Transform
        m = INV_A @ (b_filtered - B_OFFSET)

        # 5. Normalize to Unit Vector
        norm = np.linalg.norm(m)
        if norm < 1e-4:
            continue
        m_unit = m / norm

        if m_prev is None:
            m_prev = m_unit.copy()
            continue

        # 6. Instantaneous Cross-Product: dTheta = m_prev x m_now
        d_theta = np.cross(m_prev, m_unit)
        m_prev = m_unit.copy()
        dtheta_x, dtheta_y, _ = d_theta

        # 7. Deadzone & Trajectory Integration
        step_x = dtheta_y * POINTER_GAIN if abs(dtheta_y) > ROTATION_DEADZONE else 0.0
        step_y = dtheta_x * POINTER_GAIN if abs(dtheta_x) > ROTATION_DEADZONE else 0.0

        if step_x != 0.0 or step_y != 0.0:
            pos_x += step_x
            pos_y += step_y
            traj_x.append(pos_x)
            traj_y.append(pos_y)

            # Limit buffer history
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
