"""tracker_v5.py - 3-Plane Rotation Viewer with Swipe Detection (v5 pipeline).

Dual-branch architecture on top of RotationPipelineV4:

  Branch A (2D Mapping & HUD):
    Deadzone gate → atan2 direction angle → 4-quadrant classifier → HUD flash.

  Branch B (3D Integration & Trails):
    Transposed coordinate mapping (dx←d_theta_y, dy←d_theta_x, dz←d_theta_z)
    → ballistic gain → accumulate 3D position → ring-buffer trails.

  3 projection planes (XY, YZ, ZX) with independent hysteresis autoscaling.

Run from the project root:  python -m trackers.tracker_v5
Press 'C' in the window to clear trails and filter state.
"""

import math
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
# v5 BRANCH-A CONSTANTS
# ==========================================
DEADZONE_THRESHOLD: float = 0.003   # rad - minimum ‖d_theta‖ to pass the gate
SWIPE_FLASH_DURATION: float = 0.6   # seconds the HUD banner stays lit

# Quadrant boundaries in radians (half-open intervals)
_Q1 = math.pi / 4        #  45°
_Q2 = 3.0 * math.pi / 4  # 135°
_Q3 = -math.pi / 4       # -45°
_Q4 = -3.0 * math.pi / 4 # -135°


def classify_direction(d_theta: np.ndarray) -> str:
    """Map d_theta to a cardinal swipe label via atan2 heading angle.

    φ = atan2(d_theta[0], d_theta[1])
      RIGHT  : [-45°,  +45°]
      UP     : ( +45°, +135°]
      LEFT   : > +135°  or  < -135°
      DOWN   : [-135°,  -45°)
    """
    phi = math.atan2(float(d_theta[0]), float(d_theta[1]))
    if _Q3 <= phi < _Q1:
        return "RIGHT"
    if _Q1 <= phi <= _Q2:
        return "UP"
    if phi > _Q2 or phi < _Q4:
        return "LEFT"
    # _Q4 <= phi < _Q3
    return "DOWN"


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

# Branch-A HUD state
swipe_label: str = ""
swipe_flash_until: float = 0.0       # perf_counter deadline for the flash


def reset_state() -> None:
    global acc_x, acc_y, acc_z, samples_seen, deltas_seen
    global last_omega, last_gain, swipe_label, swipe_flash_until
    pipeline.reset()
    acc_x = acc_y = acc_z = 0.0
    trail_xy.clear()
    trail_yz.clear()
    trail_zx.clear()
    samples_seen = deltas_seen = 0
    last_omega = 0.0
    last_gain = 0.0
    swipe_label = ""
    swipe_flash_until = 0.0
    print("[*] Trails cleared.")


# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, (ax_xy, ax_yz, ax_zx) = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle(
    "Rotation-Delta Space (v5: Exact Arc + Vector Gate + Swipe Detection)",
    fontsize=13, fontweight='bold',
)

planes: list[tuple[plt.Axes, plt.Line2D, plt.Line2D, deque]] = []
for ax, xlab, ylab, color, trail, title in (
    (ax_xy, "Y deflection (dTheta_y x gain)", "X deflection (dTheta_x x gain)",
     'b', trail_xy, "XY Plane (Yaw / Pitch)"),
    (ax_yz, "X deflection (dTheta_x x gain)", "Z deflection (dTheta_z x gain)",
     'g', trail_yz, "YZ Plane (Pitch / Roll)"),
    (ax_zx, "Z deflection (dTheta_z x gain)", "Y deflection (dTheta_y x gain)",
     'm', trail_zx, "ZX Plane (Roll / Yaw)"),
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
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85,
              edgecolor='#cccccc')
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
    global acc_x, acc_y, acc_z, samples_seen, deltas_seen
    global last_omega, last_gain, swipe_label, swipe_flash_until

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

        # ---- Branch A: 2D Swipe Detection & HUD Flash ----
        norm_dtheta = float(np.linalg.norm(d_theta))
        if norm_dtheta > DEADZONE_THRESHOLD:
            swipe_label = classify_direction(d_theta)
            swipe_flash_until = time.perf_counter() + SWIPE_FLASH_DURATION

        # ---- Branch B: 3D Integration (transposed mapping) ----
        # dx ← d_theta[1], dy ← d_theta[0], dz ← d_theta[2]
        acc_x += d_theta[1] * gain
        acc_y += d_theta[0] * gain
        acc_z += d_theta[2] * gain

        deltas_seen += 1
        trail_xy.append((acc_x, acc_y))
        trail_yz.append((acc_y, acc_z))
        trail_zx.append((acc_z, acc_x))

    for ax, line, dot, trail in planes:
        refresh_plane(ax, line, dot, trail)

    # ---- HUD text (stats + swipe flash) ----
    flash_active = time.perf_counter() < swipe_flash_until
    swipe_line = f"Swipe: {swipe_label}" if flash_active else ""

    hud_text.set_text(
        f"Speed: {last_omega:5.2f} rad/s\n"
        f"Gain:  {last_gain:5.0f} pt/rad\n"
        f"Pos: ({acc_x:+.0f}, {acc_y:+.0f}, {acc_z:+.0f})\n"
        f"Pkt: {deltas_seen}/{samples_seen}"
        + (f"\n{swipe_line}" if swipe_line else "")
    )

    # Flash colour: green background while active, white otherwise
    hud_text.get_bbox_patch().set_facecolor(
        '#d4f5d4' if flash_active else 'white'
    )

    if frame % 25 == 0 and (samples_seen or deltas_seen == 0):
        print(f"[.] samples={samples_seen}  deltas={deltas_seen}  "
              f"pos=({acc_x:+.0f}, {acc_y:+.0f}, {acc_z:+.0f})")

    return tuple(artist for _, line, dot, _ in planes
                 for artist in (line, dot)) + (hud_text,)


ani = FuncAnimation(fig, update, interval=20, blit=False,  # noqa: F841
                    cache_frame_data=False)
plt.show()

ser.close()
