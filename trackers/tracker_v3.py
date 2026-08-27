"""tracker_v3_debug.py - Rotation-Delta Space Viewer (3 projection planes).

Signal chain identical to tracker_v2 through stage 6, then stage 7 ballistics.
Each scaled component integrates into its own accumulator channel and the
resulting trajectory plots live in three projection planes:

    left   :  acc_x vs acc_y      (from dTheta_x, dTheta_y)
    middle :  acc_y vs acc_z      (from dTheta_y, dTheta_z)
    right  :  acc_z vs acc_x      (from dTheta_z, dTheta_x)

Removed vs tracker_v2: spatial-deadzone banking and dominant-axis selection -
all three channels integrate every frame, so dTheta_z shows up in two planes
instead of being discarded.

Run from the project root:  python -m trackers.tracker_v3_debug
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
from core.pipeline import RotationPipeline, ballistic_gain

LIMIT0 = 50.0  # plot-units - initial half-width per plane

# ==========================================
# GLOBAL STATE & INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Press 'C' in the window to clear trails.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

pipeline = RotationPipeline()

# Integrated channels: each dTheta component x ballistic gain
acc_x = 0.0
acc_y = 0.0
acc_z = 0.0

# Trajectory trails: (acc_x, acc_y), (acc_y, acc_z), (acc_z, acc_x)
trail_xy: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)
trail_yz: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)
trail_zx: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)

# Heartbeat counters (console visibility that data is actually flowing)
samples_seen = 0
deltas_seen = 0


def reset_state() -> None:
    """Clear trails AND all pipeline state so tracking restarts cleanly."""
    global acc_x, acc_y, acc_z, samples_seen, deltas_seen
    pipeline.reset()
    acc_x = acc_y = acc_z = 0.0
    trail_xy.clear()
    trail_yz.clear()
    trail_zx.clear()
    samples_seen = deltas_seen = 0
    print("[*] Trails cleared.")


# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, (ax_xy, ax_yz, ax_zx) = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Rotation-Delta Space (raw dTheta, no dominance)", fontsize=14, fontweight='bold')

planes: list[tuple[plt.Axes, plt.Line2D, plt.Line2D, deque]] = []
for ax, xlab, ylab, color, trail in (
    (ax_xy, "X deflection (dTheta_x x gain)", "Y deflection (dTheta_y x gain)", 'b', trail_xy),
    (ax_yz, "Y deflection (dTheta_y x gain)", "Z deflection (dTheta_z x gain)", 'g', trail_yz),
    (ax_zx, "Z deflection (dTheta_z x gain)", "X deflection (dTheta_x x gain)", 'm', trail_zx),
):
    line, = ax.plot([], [], f'{color}-', lw=1, alpha=0.6)
    dot, = ax.plot([], [], 'ro', markersize=6)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlim(-LIMIT0, LIMIT0)
    ax.set_ylim(-LIMIT0, LIMIT0)
    planes.append((ax, line, dot, trail))

fig.canvas.mpl_connect('key_press_event', lambda event: reset_state()
                       if event.key in ('c', 'C') else None)


def refresh_plane(ax: plt.Axes, line: plt.Line2D, dot: plt.Line2D,
                  trail: deque) -> None:
    """Push one trail into its axes and auto-fit the (symmetric) view.

    Grows the moment data would escape the frame; shrinks only when the
    needed range drops below half the current one (hysteresis keeps the
    tick labels from flickering while motion hovers near the boundary).
    """
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
    """Drain the serial buffer, integrate all channels, refresh the planes."""
    global acc_x, acc_y, acc_z, samples_seen, deltas_seen

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
        samples_seen += 1

        fed = pipeline.feed(b_raw)   # stages 1-6
        if fed is None:
            continue
        d_theta, dt = fed

        # ---- Stage 7: Velocity Ballistics + per-channel integration ----
        # No dominance: every channel integrates every frame, so the z
        # channel appears in two of the three planes.
        gain = ballistic_gain(np.linalg.norm(d_theta) / dt)

        acc_x += d_theta[0] * gain
        acc_y += d_theta[1] * gain
        acc_z += d_theta[2] * gain

        deltas_seen += 1
        trail_xy.append((acc_x, acc_y))
        trail_yz.append((acc_y, acc_z))
        trail_zx.append((acc_z, acc_x))

    for ax, line, dot, trail in planes:
        refresh_plane(ax, line, dot, trail)

    if frame % 25 == 0 and (samples_seen or deltas_seen == 0):
        print(f"[.] samples={samples_seen}  deltas={deltas_seen}  "
              f"pos=({acc_x:+.0f}, {acc_y:+.0f}, {acc_z:+.0f})")

    return tuple(artist for _, line, dot, _ in planes for artist in (line, dot))


ani = FuncAnimation(fig, update, interval=20, blit=False)  # noqa: F841 - keeps animation alive
plt.show()

ser.close()
