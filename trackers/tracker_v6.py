"""tracker_v6.py — Accumulated Rotation-Axis Signatures Swipe Recognizer.

Zero-filtering, vector-axis matching architecture on top of AxisSignatureRecognizer:
  - Extracts the instantaneous vector rotation axis (B_prev x dB) per sample.
  - Accumulates rotation axis across the swipe window: A_accum = sum(B_prev x dB).
  - Normalizes the accumulated axis: u_live = normalize(A_accum).
  - Dot-product scoring against 3D direction templates: score[d] = dot(u_live, u_template[d]).
  - Zero scalar angle integration, zero filtering lag, zero trajectory buffers.

Visualization Layout:
  1. 3D Vector Sphere: Live dipole direction, template axes, and accumulated swipe axis.
  2. Cosine Similarity Bars: Real-time dot-product scores for UP, DOWN, LEFT, RIGHT.
  3. Real-Time HUD & History: Active swipe banner, metrics, and recent gesture logs.

Run from project root:
  python -m trackers.tracker_v6
Press 'C' to clear history/state, 'T' to print templates, 'Q' to quit.
"""

from __future__ import annotations

import collections
import sys
import time
from typing import Deque, List, Optional, Tuple

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
    V6_CONFIDENCE_THRESHOLD,
    V6_DIRECTIONS,
    V6_FLASH_DURATION,
    V6_TEMPLATES,
    V6_USE_UNWARPED,
)
from core.axis_signature import AxisSignatureRecognizer, AxisSwipeEvent


# ==========================================
# TRACKER STATE MACHINE
# ==========================================
class TrackerV6State:
    """Encapsulates recognizer instance, HUD state, and recent history."""

    def __init__(self) -> None:
        self.recognizer = AxisSignatureRecognizer()
        self.samples_seen: int = 0
        self.swipes_detected: int = 0

        # Current live state
        self.current_m: np.ndarray = np.array([0.0, 0.0, 1.0])
        self.live_axis_unit: Optional[np.ndarray] = None
        self.live_scores: dict[str, float] = {d: 0.0 for d in V6_DIRECTIONS}

        # HUD & Flash state
        self.last_event: Optional[AxisSwipeEvent] = None
        self.flash_until: float = 0.0
        self.history: Deque[AxisSwipeEvent] = collections.deque(maxlen=6)

    def reset(self) -> None:
        """Clear recognizer state machine and history."""
        self.recognizer.reset()
        self.samples_seen = 0
        self.swipes_detected = 0
        self.live_axis_unit = None
        self.last_event = None
        self.flash_until = 0.0
        self.history.clear()
        self.live_scores = {d: 0.0 for d in V6_DIRECTIONS}
        print("[*] Tracker state, history, and recognizer reset.")

    def feed_raw(self, b_raw_mg: np.ndarray) -> Optional[AxisSwipeEvent]:
        """Feed a single raw field sample."""
        self.samples_seen += 1
        now = time.perf_counter()

        event = self.recognizer.feed(b_raw_mg, now=now)

        # Update current dipole orientation vector from recognizer's cleaned sample
        if self.recognizer._last_processed_vec is not None:
            v = self.recognizer._last_processed_vec
            v_norm = float(np.linalg.norm(v))
            if v_norm > 1e-6:
                self.current_m = v / v_norm

        # Update in-progress live axis if currently active
        if self.recognizer.state == "ACTIVE":
            accum_norm = float(np.linalg.norm(self.recognizer._axis_accum))
            if accum_norm > 1e-9:
                self.live_axis_unit = self.recognizer._axis_accum / accum_norm
                for d in V6_DIRECTIONS:
                    if d in self.recognizer.templates:
                        self.live_scores[d] = float(
                            np.dot(self.live_axis_unit, self.recognizer.templates[d])
                        )
        else:
            self.live_axis_unit = None

        if event is not None:
            self.last_event = event
            self.flash_until = now + V6_FLASH_DURATION
            self.swipes_detected += 1
            self.history.appendleft(event)
            self.live_scores = dict(event.scores)
            self.live_axis_unit = event.axis_unit.copy()
            u = event.axis_unit
            print(
                f"[+] SWIPE: {event.direction:<5s} | conf={event.confidence:.2f} | "
                f"axis=[{u[0]:+5.2f}, {u[1]:+5.2f}, {u[2]:+5.2f}] | "
                f"samples={event.sample_count} | dur={event.duration_sec*1000:.0f}ms"
            )

        return event


# ==========================================
# SERIAL INITIALIZATION
# ==========================================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"[+] Connected to {SERIAL_PORT}. Initializing Tracker v6...")
except serial.SerialException as e:
    print(f"[-] Serial connection failed on {SERIAL_PORT}: {e}")
    raise SystemExit(1)

state = TrackerV6State()

# ==========================================
# MATPLOTLIB UI SETUP
# ==========================================
fig = plt.figure(figsize=(15, 6), facecolor="#141419")
fig.canvas.manager.set_window_title("Tracker v6 — Accumulated Rotation-Axis Signatures")

# Grid Layout: Left (3D Axis Sphere), Center (Scores Bar Chart), Right (HUD & History)
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.1], wspace=0.25)

# --- 1. 3D Axis Sphere Panel ---
ax_3d = fig.add_subplot(gs[0, 0], projection="3d", facecolor="#141419")
ax_3d.set_title("3D Rotation-Axis Space", color="#ffffff", fontsize=11, fontweight="bold", pad=12)

# Draw wireframe sphere
u_ang = np.linspace(0, 2 * np.pi, 24)
v_ang = np.linspace(0, np.pi, 16)
xs = 0.98 * np.outer(np.cos(u_ang), np.sin(v_ang))
ys = 0.98 * np.outer(np.sin(u_ang), np.sin(v_ang))
zs = 0.98 * np.outer(np.ones(np.size(u_ang)), np.cos(v_ang))
ax_3d.plot_wireframe(xs, ys, zs, color="#2c2d38", alpha=0.35, linewidth=0.6)

# Coordinate origin axes
ax_3d.plot([-1.2, 1.2], [0, 0], [0, 0], color="#555555", linestyle="--", linewidth=0.8)
ax_3d.plot([0, 0], [-1.2, 1.2], [0, 0], color="#555555", linestyle="--", linewidth=0.8)
ax_3d.plot([0, 0], [0, 0], [-1.2, 1.2], color="#555555", linestyle="--", linewidth=0.8)

# Template vectors (Static colored rays)
TEMPLATE_COLORS = {"UP": "#2ecc71", "DOWN": "#e74c3c", "LEFT": "#3498db", "RIGHT": "#f39c12"}
for name, vec in state.recognizer.templates.items():
    col = TEMPLATE_COLORS.get(name, "#9b59b6")
    ax_3d.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=col, linewidth=2.0, arrow_length_ratio=0.15)
    ax_3d.text(vec[0] * 1.15, vec[1] * 1.15, vec[2] * 1.15, name, color=col, fontsize=9, fontweight="bold")

# Dynamic rays: Current dipole orientation (cyan) & Live rotation axis (gold)
(dipole_line,) = ax_3d.plot([0, 0], [0, 0], [0, 1], color="#00ffff", linewidth=2.5, label="Dipole m")
(axis_line,) = ax_3d.plot([0, 0], [0, 0], [0, 0], color="#f1c40f", linewidth=3.5, label="Rotation Axis")

ax_3d.set_xlim([-1.2, 1.2])
ax_3d.set_ylim([-1.2, 1.2])
ax_3d.set_zlim([-1.2, 1.2])
ax_3d.tick_params(colors="#888888", labelsize=7)
ax_3d.grid(False)

# --- 2. Live Scores Panel ---
ax_bar = fig.add_subplot(gs[0, 1], facecolor="#1a1a24")
ax_bar.set_title("Cosine Similarity Scores", color="#ffffff", fontsize=11, fontweight="bold")
y_pos = np.arange(len(V6_DIRECTIONS))
bars = ax_bar.barh(y_pos, [0.0] * len(V6_DIRECTIONS), color="#34495e", height=0.55, edgecolor="#ffffff22")
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(V6_DIRECTIONS, color="#ffffff", fontsize=9, fontweight="bold")
ax_bar.set_xlim([-1.0, 1.0])
ax_bar.axvline(0, color="#666666", linewidth=0.8)
ax_bar.axvline(V6_CONFIDENCE_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.2, label="Confidence Thresh")
ax_bar.tick_params(colors="#888888", labelsize=8)
ax_bar.grid(True, linestyle=":", alpha=0.3, color="#555555")

# --- 3. Real-Time HUD & History Panel ---
ax_hud = fig.add_subplot(gs[0, 2], facecolor="#1a1a24")
ax_hud.set_title("Gesture Recognition & Metrics", color="#ffffff", fontsize=11, fontweight="bold")
ax_hud.axis("off")

hud_banner = ax_hud.text(
    0.5,
    0.85,
    "IDLE",
    color="#888888",
    fontsize=20,
    fontweight="bold",
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#22222e", edgecolor="#444455", linewidth=1.5),
)

hud_metrics = ax_hud.text(
    0.05,
    0.65,
    "Initializing...",
    color="#cccccc",
    fontsize=8.5,
    family="monospace",
    va="top",
)

hud_history = ax_hud.text(
    0.05,
    0.38,
    "Recent Swipes:\n(No gestures yet)",
    color="#aaaaaa",
    fontsize=8.0,
    family="monospace",
    va="top",
)

hud_controls = ax_hud.text(
    0.5,
    0.04,
    "[C] Clear State   |   [T] Print Templates   |   [Q] Quit",
    color="#666677",
    fontsize=8.0,
    ha="center",
    va="bottom",
)


# ==========================================
# KEYBOARD EVENT HANDLERS
# ==========================================
def on_key_press(event) -> None:
    """Handle interactive keyboard shortcuts."""
    if event.key in ("c", "C"):
        state.reset()
    elif event.key in ("t", "T"):
        print("\n" + "=" * 55)
        print("CURRENT ROTATION-AXIS TEMPLATES")
        print("=" * 55)
        for d, v in state.recognizer.templates.items():
            print(f"  {d:<8s}: [{v[0]:+6.3f}, {v[1]:+6.3f}, {v[2]:+6.3f}]")
        print("=" * 55 + "\n")
    elif event.key in ("q", "Q"):
        plt.close(fig)
        sys.exit(0)


fig.canvas.mpl_connect("key_press_event", on_key_press)


# ==========================================
# ANIMATION & SERIAL DRAIN LOOP
# ==========================================
def update(frame: int):
    """Drain serial queue and update Matplotlib visualizer."""
    now = time.perf_counter()

    # Drain serial buffer
    while ser.in_waiting:
        line = ser.readline().decode("latin-1", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            raw_lsb = [float(p) for p in parts]
        except ValueError:
            continue

        b_raw_mg = np.array(raw_lsb) * LSB_TO_MGAUSS
        state.feed_raw(b_raw_mg)

    # 1. Update 3D Axis Sphere
    m = state.current_m
    dipole_line.set_data_3d([0, m[0]], [0, m[1]], [0, m[2]])

    if state.live_axis_unit is not None:
        ax_u = state.live_axis_unit
        axis_line.set_data_3d([0, ax_u[0]], [0, ax_u[1]], [0, ax_u[2]])
        axis_line.set_alpha(1.0)
    else:
        axis_line.set_data_3d([0, 0], [0, 0], [0, 0])
        axis_line.set_alpha(0.0)

    # 2. Update Scores Bar Chart
    is_active = state.recognizer.state == "ACTIVE"
    for i, d in enumerate(V6_DIRECTIONS):
        val = state.live_scores.get(d, 0.0)
        bars[i].set_width(val)
        if val >= V6_CONFIDENCE_THRESHOLD:
            bars[i].set_color(TEMPLATE_COLORS.get(d, "#2ecc71"))
        elif val > 0.3:
            bars[i].set_color("#f39c12")
        else:
            bars[i].set_color("#34495e")

    # 3. Update HUD Banner & Metrics
    if now < state.flash_until and state.last_event is not None:
        evt = state.last_event
        col = TEMPLATE_COLORS.get(evt.direction, "#ffffff")
        hud_banner.set_text(f"SWIPE {evt.direction}")
        hud_banner.set_color(col)
        hud_banner.set_bbox(dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", edgecolor=col, linewidth=2.0))
    elif is_active:
        hud_banner.set_text("RECORDING SWIPE...")
        hud_banner.set_color("#f1c40f")
        hud_banner.set_bbox(dict(boxstyle="round,pad=0.5", facecolor="#2c2d18", edgecolor="#f1c40f", linewidth=1.5))
    else:
        hud_banner.set_text("IDLE")
        hud_banner.set_color("#777788")
        hud_banner.set_bbox(dict(boxstyle="round,pad=0.5", facecolor="#1e1e28", edgecolor="#333344", linewidth=1.2))

    # Metrics text
    last_conf = f"{state.last_event.confidence:.2f}" if state.last_event else "N/A"
    last_dur = f"{state.last_event.duration_sec*1000:.0f} ms" if state.last_event else "N/A"
    last_samples = f"{state.last_event.sample_count}" if state.last_event else "N/A"
    metrics_str = (
        f"Samples Seen    : {state.samples_seen}\n"
        f"Swipes Detected : {state.swipes_detected}\n"
        f"Last Confidence : {last_conf}\n"
        f"Last Duration   : {last_dur} ({last_samples} samples)\n"
        f"Coord Unwarping : {'INV_A Dipole (m)' if V6_USE_UNWARPED else 'B_clean (mGauss)'}"
    )
    hud_metrics.set_text(metrics_str)

    # History list
    if state.history:
        hist_lines = ["Recent Swipes:"]
        for evt in list(state.history)[:5]:
            hist_lines.append(
                f"  • {evt.direction:<5s} | conf={evt.confidence:.2f} | dur={evt.duration_sec*1000:.0f}ms | dom={evt.dominant_axis}"
            )
        hud_history.set_text("\n".join(hist_lines))
    else:
        hud_history.set_text("Recent Swipes:\n  (Perform a swipe to record)")

    return dipole_line, axis_line, *bars


ani = FuncAnimation(fig, update, interval=15, blit=False)  # noqa: F841


def main() -> None:
    """Entry point for tracker v6."""
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("[+] Serial port closed. Tracker v6 exited.")


if __name__ == "__main__":
    main()
