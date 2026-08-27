"""tracker_v4.py - 3-Plane Rotation Space Viewer (v4 pipeline + HUD).

3 projection planes (XY, YZ, ZX) with independent hysteresis autoscaling.
All three dTheta channels integrate every frame (no dominant-axis rejection).

Uses RotationPipelineV4: whole-vector glitch gate + exact arc-angle kinematics.

Run from the project root:  python -m trackers.tracker_v4
Press 'C' in the window to clear trails and filter state.
"""

import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from config import (
    BAUD_RATE,
    LIMIT_FLOOR,
    LSB_TO_MGAUSS,
    SERIAL_PORT,
    TRAIL_LEN,
)
from core.pipeline import RotationPipelineV4, ballistic_gain

LIMIT0 = 50.0

# ==========================================
# GLOBAL STATE & INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Press 'C' to clear trails.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

pipeline = RotationPipelineV4()

acc_x = 0.0
acc_y = 0.0
acc_z = 0.0

trail_xy: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)
trail_yz: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)
trail_zx: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)

last_omega = 0.0
last_gain = 0.0
samples_seen = 0
deltas_seen = 0


def reset_state() -> None:
    global acc_x, acc_y, acc_z, samples_seen, deltas_seen, last_omega, last_gain
    pipeline.reset()
    acc_x = acc_y = acc_z = 0.0
    trail_xy.clear()
    trail_yz.clear()
    trail_zx.clear()
    samples_seen = deltas_seen = 0
    last_omega = 0.0
    last_gain = 0.0
    print("[*] Trails cleared.")


# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, (ax_xy, ax_yz, ax_zx) = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Rotation-Delta Space (v4 Exact Arc + Vector Gate)", fontsize=13, fontweight='bold')

planes: list[tuple[plt.Axes, plt.Line2D, plt.Line2D, deque]] = []
for ax, xlab, ylab, color, trail, title in (
    (ax_xy, "X deflection (dTheta_x x gain)", "Y deflection (dTheta_y x gain)", 'b', trail_xy, "XY Plane (Yaw / Pitch)"),
    (ax_yz, "Y deflection (dTheta_y x gain)", "Z deflection (dTheta_z x gain)", 'g', trail_yz, "YZ Plane (Pitch / Roll)"),
    (ax_zx, "Z deflection (dTheta_z x gain)", "X deflection (dTheta_x x gain)", 'm', trail_zx, "ZX Plane (Roll / Yaw)"),
):
    line, = ax.plot([], [], f'{color}-', lw=1.2, alpha=0.7)
    dot, = ax.plot([], [], 'ro', markersize=6)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlim(-LIMIT0, LIMIT0)
    ax.set_ylim(-LIMIT0, LIMIT0)
    planes.append((ax, line, dot, trail))

hud_text = ax_xy.text(
    0.03, 0.96, '', transform=ax_xy.transAxes,
    fontsize=8.5, verticalalignment='top',
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc')
)

fig.canvas.mpl_connect('key_press_event', lambda event: reset_state()
                       if event.key in ('c', 'C') else None)


def refresh_plane(ax: plt.Axes, line: plt.Line2D, dot: plt.Line2D,
                  trail: deque) -> None:
    if not trail:
        return
    xs, ys = zip(*trail)
    line.set_data(xs, ys)
    dot.set_data([xs[-1]], [ys[-1]])
    target = max(max(abs(v) for pt in trail for v in pt) * 1.2, LIMIT_FLOOR)
    cur = ax.get_xlim()[1]
    if target > cur or target < 0.5 * cur:
        ax.set_xlim(-target, target)
        ax.set_ylim(-target, target)


# ==========================================
# ANIMATION LOOP
# ==========================================
def update(frame: int):
    global acc_x, acc_y, acc_z, samples_seen, deltas_seen, last_omega, last_gain

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
        samples_seen += 1

        fed = pipeline.feed(b_raw)
        if fed is None:
            continue
        d_theta, dt = fed

        omega = float(np.linalg.norm(d_theta) / dt)
        gain = ballistic_gain(omega)
        last_omega = omega
        last_gain = gain

        acc_x += d_theta[0] * gain
        acc_y += d_theta[1] * gain
        acc_z += d_theta[2] * gain

        deltas_seen += 1
        trail_xy.append((acc_x, acc_y))
        trail_yz.append((acc_y, acc_z))
        trail_zx.append((acc_z, acc_x))

    for ax, line, dot, trail in planes:
        refresh_plane(ax, line, dot, trail)

    hud_text.set_text(
        f"Speed: {last_omega:5.2f} rad/s\n"
        f"Gain:  {last_gain:5.0f} pt/rad\n"
        f"Pos: ({acc_x:+.0f}, {acc_y:+.0f}, {acc_z:+.0f})\n"
        f"Pkt: {deltas_seen}/{samples_seen}"
    )

    if frame % 25 == 0 and (samples_seen or deltas_seen == 0):
        print(f"[.] samples={samples_seen}  deltas={deltas_seen}  "
              f"pos=({acc_x:+.0f}, {acc_y:+.0f}, {acc_z:+.0f})")

    return tuple(artist for _, line, dot, _ in planes for artist in (line, dot)) + (hud_text,)


ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)  # noqa: F841
plt.show()

ser.close()
