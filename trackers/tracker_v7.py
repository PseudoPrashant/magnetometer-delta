"""tracker_v7.py — Stream Kinematic Energy & Rotation-Plane Swipe Tracker.

Empirically derived from continuous 100 Hz stream recordings:
  • Synchronous 3D-coupled denoiser eliminating resting sensor noise.
  • Dipole-space geometry unwarping: m = INV_A @ (B - B_OFFSET).
  • Stateful angular velocity and physical stroke windowing.
  • Velocity-weighted momentum accumulation: A = sum((u x du) * ||du||^1.3).
  • Dual-plane + template cosine similarity matching.
  • Dynamic refractory lockout against deceleration bounce.

Visualization Layout:
  1. 3D Dipole Vector Sphere: Live unit dipole direction, historical trail, and template axes.
  2. Kinematic Tachometer: Live angular velocity (rad/s) and cosine similarity bars.
  3. Real-Time HUD & History: Active swipe highlight banner and gesture ledger.

Run from project root:
  python -m trackers.tracker_v7
Press 'C' to clear history/state, 'T' to print templates, 'Q' to quit.
"""

from __future__ import annotations

import collections
import sys
import time
from typing import Deque, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import (
    BAUD_RATE,
    B_OFFSET,
    INV_A,
    LSB_TO_MGAUSS,
    SERIAL_PORT,
    V6_FLASH_DURATION,
)
from core.kinematic_swipe import (
    KinematicSwipeEvent,
    StreamKinematicSwipeDetector,
    V7_DEFAULT_TEMPLATES,
)

DIRECTIONS: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
DIR_COLORS: Dict[str, str] = {
    "UP": "#00E5FF",      # Cyan
    "DOWN": "#FF5252",    # Coral Red
    "LEFT": "#FFD600",    # Yellow
    "RIGHT": "#00E676",   # Green
    "UNKNOWN": "#9E9E9E", # Grey
}


class TrackerV7State:
    """Encapsulates detector instance, HUD state, and recent history."""

    def __init__(self) -> None:
        self.detector = StreamKinematicSwipeDetector()
        self.samples_seen: int = 0
        self.swipes_detected: int = 0

        # Current live state
        self.current_u: np.ndarray = np.array([0.0, 0.0, 1.0])
        self.u_trail: Deque[np.ndarray] = collections.deque(maxlen=40)
        self.current_speed: float = 0.0
        self.live_scores: Dict[str, float] = {d: 0.0 for d in DIRECTIONS}

        # HUD & Flash state
        self.last_event: Optional[KinematicSwipeEvent] = None
        self.flash_until: float = 0.0
        self.history: Deque[KinematicSwipeEvent] = collections.deque(maxlen=6)

    def reset(self) -> None:
        """Clear detector state machine and history."""
        self.detector.reset()
        self.samples_seen = 0
        self.swipes_detected = 0
        self.u_trail.clear()
        self.last_event = None
        self.flash_until = 0.0
        self.history.clear()
        self.live_scores = {d: 0.0 for d in DIRECTIONS}
        print("[*] Tracker v7 state, history, and detector reset.")

    def feed_raw(self, b_raw_mg: np.ndarray, dt: float = 0.010) -> Optional[KinematicSwipeEvent]:
        """Feed a single raw field sample."""
        self.samples_seen += 1
        event = self.detector.feed_raw(b_raw_mg, dt=dt)

        self.current_u = self.detector.current_u
        self.u_trail.append(self.current_u.copy())
        self.current_speed = self.detector.current_speed_rad_s

        # Update live template scores
        for d in DIRECTIONS:
            tmpl = self.detector.templates.get(d)
            if tmpl is not None:
                self.live_scores[d] = float(np.dot(self.current_u, tmpl))

        if event is not None:
            self.swipes_detected += 1
            self.last_event = event
            self.flash_until = time.perf_counter() + V6_FLASH_DURATION
            self.history.appendleft(event)
            print(
                f"[+] >>> [SWIPE DETECTED #{self.swipes_detected:02d}] "
                f"DIR: {event.direction:5s} | Conf: {event.confidence*100:4.1f}% | "
                f"Dur: {event.duration_ms:3.0f}ms | Arc: {event.displacement_deg:4.1f}deg | "
                f"Speed: {event.peak_speed_rad_s:4.1f} rad/s"
            )

        return event


# ==========================================
# SERIAL SETUP
# ==========================================
print(f"[*] Opening serial port {SERIAL_PORT} @ {BAUD_RATE} baud...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.005)
    time.sleep(1.8)
    ser.reset_input_buffer()
    print(f"[+] Connected successfully to {SERIAL_PORT}.")
except serial.SerialException as e:
    print(f"[-] Serial error: {e}")
    print("[-] Please ensure ESP32 is plugged in and other monitors are closed.")
    raise SystemExit(1)


# ==========================================
# MATPLOTLIB DASHBOARD SETUP
# ==========================================
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 8.5), facecolor="#121212")
fig.canvas.manager.set_window_title("Tracker v7 — Stream Kinematic Energy & Rotation-Plane Engine")

# Grid layout: 3D sphere on left, tachometer/bars top-right, HUD bottom-right
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.30)
ax_3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="#181818")
ax_bars = fig.add_subplot(gs[0, 1], facecolor="#181818")
ax_hud = fig.add_subplot(gs[1, 1], facecolor="#181818")

# --- Configure 3D Sphere Axes ---
ax_3d.set_xlim([-1.1, 1.1])
ax_3d.set_ylim([-1.1, 1.1])
ax_3d.set_zlim([-1.1, 1.1])
ax_3d.set_xlabel("X (UP/DOWN)", color="#CCCCCC", labelpad=8)
ax_3d.set_ylabel("Y (LEFT/RIGHT)", color="#CCCCCC", labelpad=8)
ax_3d.set_zlabel("Z (Vertical)", color="#CCCCCC", labelpad=8)
ax_3d.set_title("Live Dipole Orientation & Rotation Axes", color="#FFFFFF", fontsize=13, pad=12)

# Draw unit sphere wireframe
u_theta = np.linspace(0, 2 * np.pi, 28)
v_phi = np.linspace(0, np.pi, 14)
xs = np.outer(np.cos(u_theta), np.sin(v_phi))
ys = np.outer(np.sin(u_theta), np.sin(v_phi))
zs = np.outer(np.ones_like(u_theta), np.cos(v_phi))
ax_3d.plot_wireframe(xs, ys, zs, color="#2C3440", alpha=0.35, linewidth=0.6)

# Reference Template Rays
for d, tmpl in V7_DEFAULT_TEMPLATES.items():
    c = DIR_COLORS.get(d, "#FFFFFF")
    ax_3d.quiver(0, 0, 0, tmpl[0], tmpl[1], tmpl[2], color=c, alpha=0.75, linewidth=2.0, arrow_length_ratio=0.15)
    ax_3d.text(tmpl[0] * 1.15, tmpl[1] * 1.15, tmpl[2] * 1.15, d, color=c, fontsize=10, fontweight="bold")

# Dynamic elements on 3D plot
live_trail_line, = ax_3d.plot([], [], [], color="#00E5FF", linewidth=1.5, alpha=0.6)
live_point = ax_3d.scatter([0], [0], [1], color="#FFFFFF", s=80, edgecolors="#00E5FF", linewidth=1.5)

# --- Configure Bars Axes ---
ax_bars.set_xlim([-1.0, 1.0])
ax_bars.set_ylim([-0.6, len(DIRECTIONS) - 0.4])
ax_bars.set_yticks(range(len(DIRECTIONS)))
ax_bars.set_yticklabels(DIRECTIONS, color="#FFFFFF", fontsize=11, fontweight="bold")
ax_bars.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--")
ax_bars.set_xlabel("Dipole Projection / Cosine Score", color="#CCCCCC", fontsize=10)
ax_bars.set_title("Kinematic Speed & Direction Projections", color="#FFFFFF", fontsize=12, pad=10)
bar_rects = ax_bars.barh(range(len(DIRECTIONS)), [0.0] * len(DIRECTIONS), color=[DIR_COLORS[d] for d in DIRECTIONS], height=0.55)

# --- Configure HUD & Ledger Axes ---
ax_hud.axis("off")

state = TrackerV7State()


# ==========================================
# KEYBOARD EVENT HANDLERS
# ==========================================
def on_key(event) -> None:
    if event.key in ("q", "Q"):
        plt.close(fig)
        sys.exit(0)
    elif event.key in ("c", "C"):
        state.reset()
    elif event.key in ("t", "T"):
        print("\n" + "=" * 60)
        print("V7 REFERENCE ROTATION AXES:")
        for d, t in V7_DEFAULT_TEMPLATES.items():
            print(f"  {d:5s} : [{t[0]:+6.3f}, {t[1]:+6.3f}, {t[2]:+6.3f}]")
        print("=" * 60 + "\n")


fig.canvas.mpl_connect("key_press_event", on_key)


# ==========================================
# ANIMATION LOOP
# ==========================================
def update(_frame) -> None:
    # 1. Drain serial packets
    if ser.is_open:
        try:
            lines = ser.read_all().decode("latin-1", errors="ignore").splitlines()
        except Exception:
            lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            try:
                raw_lsb = [float(p) for p in parts]
            except ValueError:
                continue

            b_raw_mg = np.array(raw_lsb, dtype=float) * LSB_TO_MGAUSS
            state.feed_raw(b_raw_mg, dt=0.010)

    # 2. Update 3D Sphere Visuals
    if state.u_trail:
        trail_arr = np.array(state.u_trail)
        live_trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        live_trail_line.set_3d_properties(trail_arr[:, 2])

    u = state.current_u
    live_point._offsets3d = ([u[0]], [u[1]], [u[2]])

    # 3. Update Bar Plots
    for idx, d in enumerate(DIRECTIONS):
        val = state.live_scores.get(d, 0.0)
        bar_rects[idx].set_width(val)

    # 4. Render HUD & Ledger
    ax_hud.clear()
    ax_hud.axis("off")

    now = time.perf_counter()
    flashing = now < state.flash_until and state.last_event is not None

    if flashing and state.last_event:
        evt = state.last_event
        c = DIR_COLORS.get(evt.direction, "#FFFFFF")
        banner_text = f">>> SWIPE: {evt.direction} ({evt.confidence*100:.1f}%) <<<"
        sub_text = f"Duration: {evt.duration_ms:.0f} ms  |  Arc: {evt.displacement_deg:.1f}°  |  Peak: {evt.peak_speed_rad_s:.1f} rad/s"
        ax_hud.text(0.5, 0.88, banner_text, color=c, fontsize=18, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#242424", edgecolor=c, linewidth=2.0))
        ax_hud.text(0.5, 0.68, sub_text, color="#DDDDDD", fontsize=10, ha="center", va="center")
    else:
        status_color = "#00E676" if state.current_speed > 1.2 else "#666666"
        status_label = "ACTIVE MOTION" if state.current_speed > 1.2 else "IDLE / READY"
        ax_hud.text(0.5, 0.88, f"Status: {status_label}", color=status_color, fontsize=14, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1F1F1F", edgecolor="#333333", linewidth=1.0))
        ax_hud.text(0.5, 0.68, f"Live Speed: {state.current_speed:4.1f} rad/s  |  Samples: {state.samples_seen:,}  |  Total Swipes: {state.swipes_detected}",
                    color="#888888", fontsize=10, ha="center", va="center")

    # Historical Event Ledger
    ax_hud.text(0.02, 0.48, "Recent Gesture Ledger:", color="#AAAAAA", fontsize=10, fontweight="bold")
    y_pos = 0.36
    if not state.history:
        ax_hud.text(0.05, y_pos, "No swipes detected yet — swipe trackball in any cardinal direction.", color="#555555", fontsize=9)
    else:
        for idx, h_evt in enumerate(list(state.history)[:4]):
            h_col = DIR_COLORS.get(h_evt.direction, "#FFFFFF")
            line = f"#{state.swipes_detected - idx:02d}  {h_evt.direction:5s}  |  {h_evt.confidence*100:4.1f}%  |  {h_evt.duration_ms:3.0f}ms  |  arc={h_evt.displacement_deg:4.1f}°  |  {h_evt.peak_speed_rad_s:4.1f} rad/s"
            ax_hud.text(0.05, y_pos, line, color=h_col, fontsize=9.5, fontfamily="monospace")
            y_pos -= 0.09

    ax_hud.text(0.5, 0.02, "[C: Clear State]    [T: Print Templates]    [Q: Quit]", color="#555555", fontsize=9, ha="center")


ani = FuncAnimation(fig, update, interval=25, cache_frame_data=False)  # noqa: F841

try:
    plt.show()
finally:
    if ser.is_open:
        ser.close()
        print("[+] Serial port closed.")
