"""tracker_v4.py - High-Performance Magnetic Trackball Pointer (v4 pipeline).

Key innovations over tracker_v2:
    1. Whole-Vector Glitch Gate: Zero axis phase skew, zero group latency.
    2. Exact Geodesic Arc-Angle: Eliminates chord deficit during fast flicks (>50 rad/s).
    3. Interactive Diagnostics HUD: Real-time angular speed, gain, and FPS metrics.
    4. Runtime Hotkeys: 'C' to clear, 'D' to toggle dominant axis on-the-fly.

Run from the project root:  python -m trackers.tracker_v4
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
from core.pipeline import RotationPipelineV4, ballistic_gain

# ==========================================
# GLOBAL STATE & INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Press 'C' to clear, 'D' to toggle dominant axis.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

pipeline = RotationPipelineV4()

# Runtime configuration toggles
dominant_axis_enabled: bool = DOMINANT_AXIS

# Fractional accumulation residues (Stage 8)
frac_x = 0.0
frac_y = 0.0

# Trajectory tracking state
pos_x = 0.0
pos_y = 0.0
traj_x = [0.0]
traj_y = [0.0]

# Diagnostics state
last_omega = 0.0
last_gain = 0.0
samples_processed = 0
glitches_dropped = 0


def reset_state() -> None:
    """Clear trajectory AND all pipeline state so tracking restarts cleanly."""
    global frac_x, frac_y, pos_x, pos_y, traj_x, traj_y, last_omega, last_gain
    pipeline.reset()
    frac_x = frac_y = 0.0
    pos_x = pos_y = 0.0
    traj_x = [0.0]
    traj_y = [0.0]
    last_omega = 0.0
    last_gain = 0.0
    print("[*] Trace cleared.")


def toggle_dominant_axis() -> None:
    """Toggle orthogonal axis snapping mode on/off."""
    global dominant_axis_enabled
    dominant_axis_enabled = not dominant_axis_enabled
    state_str = "ON" if dominant_axis_enabled else "OFF"
    print(f"[*] Dominant Axis Snapping: {state_str}")


# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(8.5, 8.5))
line_traj, = ax.plot([], [], 'b-', lw=1.6, alpha=0.75, label='Trajectory')
current_dot, = ax.plot([], [], 'ro', markersize=8, label='Current Pos')

ax.set_title("Live Magnetic Sphere Rolling Trajectory (v4)", fontsize=13, fontweight='bold')
ax.set_xlabel("X Deflection (Integrated dTheta_Y)", fontsize=10)
ax.set_ylabel("Y Deflection (Integrated dTheta_X)", fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(0, color='gray', lw=0.8)
ax.legend(loc='upper right', framealpha=0.8)

ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)

# Diagnostic HUD overlay
hud_text = ax.text(
    0.02, 0.98, '', transform=ax.transAxes,
    fontsize=9, verticalalignment='top',
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='#cccccc')
)


def on_key(event) -> None:
    """Handle keyboard hotkeys."""
    if event.key in ('c', 'C'):
        reset_state()
    elif event.key in ('d', 'D'):
        toggle_dominant_axis()


fig.canvas.mpl_connect('key_press_event', on_key)


# ==========================================
# ANIMATION LOOP
# ==========================================
def update(frame: int):
    """Drain serial buffer, process through v4 pipeline, and refresh canvas."""
    global frac_x, frac_y, pos_x, pos_y, traj_x, traj_y
    global last_omega, last_gain, samples_processed, glitches_dropped

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

        samples_processed += 1

        fed = pipeline.feed(b_raw)   # stages 1-6 (v4 vector gate + exact arc angle)
        if fed is None:
            continue
        d_theta, dt = fed

        # Compute angular velocity & gain
        omega = float(np.linalg.norm(d_theta) / dt)
        gain = ballistic_gain(omega)
        last_omega = omega
        last_gain = gain

        # ---- Stage 7: Dynamic Velocity Ballistics ----
        # Component mapping: d_theta[1] (Y-rot) -> X, d_theta[0] (X-rot) -> Y
        step_x = d_theta[1] * gain
        step_y = d_theta[0] * gain

        # ---- Stage 8: Dominant Axis Selection ----
        if dominant_axis_enabled:
            if abs(step_x) >= abs(step_y):
                step_y = 0.0
            else:
                step_x = 0.0

        # ---- Stage 9: Spatial Deadzone & Fractional Accumulation ----
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

    # Update plot lines and HUD
    if traj_x:
        line_traj.set_data(traj_x, traj_y)
        current_dot.set_data([pos_x], [pos_y])

        # Dynamic autoscaling if trajectory escapes viewport
        max_bound = max(abs(pos_x), abs(pos_y), VIEW_LIMIT) * 1.2
        if max_bound > ax.get_xlim()[1]:
            ax.set_xlim(-max_bound, max_bound)
            ax.set_ylim(-max_bound, max_bound)

    # Update HUD status
    dom_str = "ON" if dominant_axis_enabled else "OFF"
    hud_text.set_text(
        f"Pos: ({pos_x:+6.1f}, {pos_y:+6.1f})\n"
        f"Speed: {last_omega:5.2f} rad/s\n"
        f"Gain:  {last_gain:5.0f} pt/rad\n"
        f"Snap:  {dom_str} (Press 'D')\n"
        f"Total: {samples_processed} samples"
    )

    return line_traj, current_dot, hud_text


ani = FuncAnimation(fig, update, interval=20, blit=False)  # noqa: F841 - keeps animation alive
plt.show()

ser.close()
