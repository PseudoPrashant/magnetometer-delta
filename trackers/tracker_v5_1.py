"""tracker_v5_1.py - 3-Plane Rotation Space Viewer with Stroke Gesture Recognition.

Dual-branch architecture on top of RotationPipelineV5:

  Branch A (Stroke-Level Gesture & Swipe Recognition):
    - Multi-frame gesture accumulator: accumulates total stroke vector ΔΘ_stroke = Σ d_theta.
    - Energy gating: requires minimum displacement (SWIPE_MIN_DISPLACEMENT) and peak speed (SWIPE_MIN_OMEGA_PEAK).
    - Biomechanical tilt trim (SWIPE_TILT_OFFSET_DEG) to eliminate diagonal finger curl.
    - Post-swipe cooldown lockout (SWIPE_COOLDOWN_SEC) to reject return strokes and finger releases.
    - Real-time HUD banner with direction, displacement magnitude, and peak velocity.

  Branch B (3D Trajectory Integration & Multi-Plane Viewer):
    - 3D-coupled One Euro filter (zero inter-axis phase lag during fast flicks).
    - Burst-timing rectification (normalizes dt during serial buffer batch reads).
    - Sub-noise floor gating (DTHETA_NOISE_FLOOR) and leaky retention damping (LEAK_ALPHA) for zero drift.
    - Smooth C-infinity sigmoid velocity ballistics (ballistic_gain_smooth).
    - Transposed coordinate mapping (dx←d_theta_y, dy←d_theta_x, dz←d_theta_z).
    - 3 projection planes (XY, YZ, ZX) with independent hysteresis autoscaling.

Run from the project root:  python -m trackers.tracker_v5_1
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
    DTHETA_NOISE_FLOOR,
    LEAK_ALPHA,
    LIMIT_FLOOR,
    LSB_TO_MGAUSS,
    SERIAL_PORT,
    SWIPE_FLASH_DURATION,
    TRAIL_LEN,
)
from core.pipeline import (
    RotationPipelineV5,
    StrokeGestureRecognizer,
    ballistic_gain_smooth,
)

LIMIT0 = 50.0


# ==========================================
# TRACKER STATE MACHINE
# ==========================================
class TrackerState:
    """Encapsulates all filter, gesture, and trajectory state."""

    def __init__(self) -> None:
        self.pipeline = RotationPipelineV5()
        self.recognizer = StrokeGestureRecognizer()

        self.acc_x: float = 0.0
        self.acc_y: float = 0.0
        self.acc_z: float = 0.0

        self.trails: dict[str, deque[tuple[float, float]]] = {
            'xy': deque(maxlen=TRAIL_LEN),
            'yz': deque(maxlen=TRAIL_LEN),
            'zx': deque(maxlen=TRAIL_LEN),
        }

        self.last_omega: float = 0.0
        self.last_gain: float = 0.0
        self.samples_seen: int = 0
        self.deltas_seen: int = 0
        self.strokes_detected: int = 0

        # Branch A HUD state
        self.last_swipe_label: str = ""
        self.last_swipe_disp: float = 0.0
        self.last_swipe_peak_w: float = 0.0
        self.swipe_flash_until: float = 0.0

    def reset(self) -> None:
        """Clear all internal state so tracking restarts cleanly."""
        self.pipeline.reset()
        self.recognizer.reset()
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.acc_z = 0.0
        for trail in self.trails.values():
            trail.clear()
        self.samples_seen = 0
        self.deltas_seen = 0
        self.strokes_detected = 0
        self.last_omega = 0.0
        self.last_gain = 0.0
        self.last_swipe_label = ""
        self.last_swipe_disp = 0.0
        self.last_swipe_peak_w = 0.0
        self.swipe_flash_until = 0.0
        print("[*] Trails and gesture state cleared.")

    def feed_sample(self, b_raw: np.ndarray) -> None:
        """Process one raw field sample through both pipeline branches."""
        self.samples_seen += 1

        fed = self.pipeline.feed(b_raw)
        if fed is None:
            return

        d_theta, dt = fed
        now = time.perf_counter()

        omega = float(np.linalg.norm(d_theta) / dt) if dt > 0 else 0.0
        gain = ballistic_gain_smooth(omega)
        self.last_omega = omega
        self.last_gain = gain

        # ---- Branch A: Stroke-Level Gesture Recognizer ----
        swipe_evt = self.recognizer.feed(d_theta, dt, now)
        if swipe_evt is not None:
            direction, disp, peak_w = swipe_evt
            self.last_swipe_label = direction
            self.last_swipe_disp = disp
            self.last_swipe_peak_w = peak_w
            self.swipe_flash_until = now + SWIPE_FLASH_DURATION
            self.strokes_detected += 1
            print(
                f"[+] SWIPE: {direction:5s} | disp={disp:.3f} rad ({np.degrees(disp):.1f}°) | peak_w={peak_w:.1f} rad/s"
            )

        # ---- Branch B: 3D Trajectory Integration (Leaky + Noise Floor) ----
        d_theta_mag = float(np.linalg.norm(d_theta))
        if d_theta_mag >= DTHETA_NOISE_FLOOR:
            # Apply leaky damping to eliminate stationary drift
            self.acc_x *= LEAK_ALPHA
            self.acc_y *= LEAK_ALPHA
            self.acc_z *= LEAK_ALPHA

            # Transposed mapping: dx ← d_theta[1] (Y-rot), dy ← d_theta[0] (X-rot), dz ← d_theta[2] (Z-rot)
            self.acc_x += d_theta[1] * gain
            self.acc_y += d_theta[0] * gain
            self.acc_z += d_theta[2] * gain

            self.deltas_seen += 1
            self.trails['xy'].append((self.acc_x, self.acc_y))
            self.trails['yz'].append((self.acc_y, self.acc_z))
            self.trails['zx'].append((self.acc_z, self.acc_x))


# ==========================================
# SERIAL INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"[+] Connected to {SERIAL_PORT}. Press 'C' to clear trails.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    raise SystemExit(1)

state = TrackerState()

# ==========================================
# MATPLOTLIB SETUP
# ==========================================
fig, (ax_xy, ax_yz, ax_zx) = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle(
    "Rotation-Delta Space (v5.1: Stroke Recognizer + 3D-Coupled Filter + Leaky Integration)",
    fontsize=13,
    fontweight='bold',
)

planes: list[tuple[plt.Axes, plt.Line2D, plt.Line2D, str]] = []
for ax, xlab, ylab, color, trail_key, title in (
    (
        ax_xy,
        "Y deflection (dTheta_y x gain)",
        "X deflection (dTheta_x x gain)",
        'b',
        'xy',
        "XY Plane (Yaw / Pitch)",
    ),
    (
        ax_yz,
        "X deflection (dTheta_x x gain)",
        "Z deflection (dTheta_z x gain)",
        'g',
        'yz',
        "YZ Plane (Pitch / Roll)",
    ),
    (
        ax_zx,
        "Z deflection (dTheta_z x gain)",
        "Y deflection (dTheta_y x gain)",
        'm',
        'zx',
        "ZX Plane (Roll / Yaw)",
    ),
):
    (line,) = ax.plot([], [], f'{color}-', lw=1.2, alpha=0.7)
    (dot,) = ax.plot([], [], 'ro', markersize=6)
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
    0.03,
    0.96,
    '',
    transform=ax_xy.transAxes,
    fontsize=8.5,
    verticalalignment='top',
    family='monospace',
    bbox=dict(
        boxstyle='round,pad=0.4',
        facecolor='white',
        alpha=0.88,
        edgecolor='#cccccc',
    ),
)

fig.canvas.mpl_connect(
    'key_press_event',
    lambda event: state.reset() if event.key in ('c', 'C') else None,
)


def refresh_plane(
    ax: plt.Axes, line: plt.Line2D, dot: plt.Line2D, trail: deque
) -> None:
    """Redraw trail lines with hysteresis autoscaling."""
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

        state.feed_sample(b_raw)

    for ax, line, dot, trail_key in planes:
        refresh_plane(ax, line, dot, state.trails[trail_key])

    # ---- HUD Text & Gesture Flash ----
    now = time.perf_counter()
    flash_active = now < state.swipe_flash_until

    if flash_active:
        swipe_str = f"SWIPE: >>> {state.last_swipe_label} <<< ({np.degrees(state.last_swipe_disp):.0f}°, {state.last_swipe_peak_w:.1f} rad/s)\n"
    elif state.recognizer.state == "ACTIVE":
        swipe_str = "SWIPE: [Tracking stroke...]\n"
    else:
        swipe_str = (
            f"Last:  {state.last_swipe_label or 'None'} (Swipes: {state.strokes_detected})\n"
        )

    hud_text.set_text(
        f"{swipe_str}"
        f"Speed: {state.last_omega:5.2f} rad/s\n"
        f"Gain:  {state.last_gain:5.0f} pt/rad\n"
        f"Pos:   ({state.acc_x:+.0f}, {state.acc_y:+.0f}, {state.acc_z:+.0f})\n"
        f"Pkt:   {state.deltas_seen}/{state.samples_seen}"
    )

    # Colorize HUD box: vibrant light green while flash is active
    hud_text.get_bbox_patch().set_facecolor(
        '#d4f5d4' if flash_active else 'white'
    )

    if frame % 25 == 0 and (state.samples_seen or state.deltas_seen == 0):
        print(
            f"[.] samples={state.samples_seen} deltas={state.deltas_seen} swipes={state.strokes_detected} "
            f"pos=({state.acc_x:+.0f}, {state.acc_y:+.0f}, {state.acc_z:+.0f})"
        )

    return tuple(
        artist for _, line, dot, _ in planes for artist in (line, dot)
    ) + (hud_text,)


ani = FuncAnimation(  # noqa: F841
    fig, update, interval=20, blit=False, cache_frame_data=False
)
plt.show()

ser.close()
