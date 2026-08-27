"""tracker_v4_1.py - Improved 3-Plane Rotation Viewer (v5 pipeline).

Improvements over tracker_v4:
    1. Leaky integrator — eliminates stationary drift.
    2. Per-frame sample cap — prevents frame-time spikes.
    3. 3D-coupled One Euro — zero residual axis skew (in V5 pipeline).
    4. Warm-up telemetry — HUD shows pipeline state during startup.
    5. Adaptive glitch gate — self-tuning threshold (in V5 pipeline).
    6. CSV session logging — timestamped raw data for offline replay.
    7. O(1) deque trails — capped at TRAIL_LEN.
    8. Smooth sigmoid gain — C-infinity gain curve, no derivative jumps.
    9. Class-based state — no bare globals, single reset point.
   10. d_theta noise floor — sub-noise rotation skipped before integration.

Run from the project root:  python -m trackers.tracker_v4_1
Press 'C' to clear trails and filter state.
"""

import csv
import datetime
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation

from config import (
    BAUD_RATE,
    DTHETA_NOISE_FLOOR,
    LEAK_ALPHA,
    LIMIT_FLOOR,
    LSB_TO_MGAUSS,
    MAX_SAMPLES_PER_FRAME,
    SERIAL_PORT,
    TRAIL_LEN,
)
from core.pipeline import RotationPipelineV5, ballistic_gain_smooth

LIMIT0 = 50.0


class TrackerState:
    def __init__(self) -> None:
        self.pipeline = RotationPipelineV5()
        self.acc = np.zeros(3)
        self.trails: dict[str, deque[tuple[float, float]]] = {
            k: deque(maxlen=TRAIL_LEN) for k in ('xy', 'yz', 'zx')
        }
        self.last_omega = 0.0
        self.last_gain = 0.0
        self.samples_seen = 0
        self.deltas_seen = 0
        self.warming_up_count = 0
        self._log_file = None
        self._log_writer = None
        self._start_logging()

    def _start_logging(self) -> None:
        name = f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        self._log_file = open(name, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(['timestamp', 'b_x', 'b_y', 'b_z'])

    def log_sample(self, b_raw: np.ndarray) -> None:
        if self._log_writer is not None:
            self._log_writer.writerow([time.perf_counter(), b_raw[0], b_raw[1], b_raw[2]])

    def reset(self) -> None:
        self.pipeline.reset()
        self.acc[:] = 0.0
        for trail in self.trails.values():
            trail.clear()
        self.last_omega = 0.0
        self.last_gain = 0.0
        self.samples_seen = 0
        self.deltas_seen = 0
        self.warming_up_count = 0
        print("[*] Trails cleared.")

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()

    def feed_sample(self, b_raw: np.ndarray) -> None:
        self.log_sample(b_raw)
        self.samples_seen += 1

        fed = self.pipeline.feed(b_raw)
        if fed is None:
            if not self.pipeline._warmup_done:
                self.warming_up_count += 1
            return

        d_theta, dt = fed
        omega = float(np.linalg.norm(d_theta) / dt)
        gain = ballistic_gain_smooth(omega)

        d_theta_mag = float(np.linalg.norm(d_theta))
        if d_theta_mag < DTHETA_NOISE_FLOOR:
            return

        self.last_omega = omega
        self.last_gain = gain
        self.acc *= LEAK_ALPHA
        self.acc += d_theta * gain

        self.deltas_seen += 1
        self.trails['xy'].append((float(self.acc[0]), float(self.acc[1])))
        self.trails['yz'].append((float(self.acc[1]), float(self.acc[2])))
        self.trails['zx'].append((float(self.acc[2]), float(self.acc[0])))


try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Press 'C' to clear trails.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

state = TrackerState()

fig, (ax_xy, ax_yz, ax_zx) = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Rotation-Delta Space (v5: Adaptive Gate + Coupled Filter + Smooth Gain)",
             fontsize=13, fontweight='bold')

planes: list[tuple[plt.Axes, plt.Line2D, plt.Line2D, deque]] = []
for ax, xlab, ylab, color, trail_key, title in (
    (ax_xy, "X deflection (dTheta_x x gain)", "Y deflection (dTheta_y x gain)",
     'b', 'xy', "XY Plane (Yaw / Pitch)"),
    (ax_yz, "Y deflection (dTheta_y x gain)", "Z deflection (dTheta_z x gain)",
     'g', 'yz', "YZ Plane (Pitch / Roll)"),
    (ax_zx, "Z deflection (dTheta_z x gain)", "X deflection (dTheta_x x gain)",
     'm', 'zx', "ZX Plane (Roll / Yaw)"),
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
    planes.append((ax, line, dot, trail_key))

hud_text = ax_xy.text(
    0.03, 0.96, '', transform=ax_xy.transAxes,
    fontsize=8.5, verticalalignment='top',
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc')
)

fig.canvas.mpl_connect('key_press_event',
                       lambda event: state.reset() if event.key in ('c', 'C') else None)


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


def update(frame: int):
    count = 0
    while ser.in_waiting and count < MAX_SAMPLES_PER_FRAME:
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

        state.feed_sample(b_raw)
        count += 1

    for ax, line, dot, trail_key in planes:
        refresh_plane(ax, line, dot, state.trails[trail_key])

    warmup_str = f"Warmup: {state.warming_up_count}\n" if not state.pipeline._warmup_done else ""
    hud_text.set_text(
        f"{warmup_str}"
        f"Speed: {state.last_omega:5.2f} rad/s\n"
        f"Gain:  {state.last_gain:5.0f} pt/rad\n"
        f"Pos: ({state.acc[0]:+.0f}, {state.acc[1]:+.0f}, {state.acc[2]:+.0f})\n"
        f"Pkt: {state.deltas_seen}/{state.samples_seen}"
    )

    if frame % 25 == 0 and (state.samples_seen or state.deltas_seen == 0):
        print(f"[.] samples={state.samples_seen}  deltas={state.deltas_seen}  "
              f"pos=({state.acc[0]:+.0f}, {state.acc[1]:+.0f}, {state.acc[2]:+.0f})")

    return tuple(artist for _, line, dot, _ in planes for artist in (line, dot)) + (hud_text,)


ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)  # noqa: F841
plt.show()

state.close()
ser.close()
