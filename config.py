"""Central configuration for the magnetometer-delta project.

Every tunable lives here so the calibration tool and all trackers share one
source of truth. Edit SERIAL_PORT / B_OFFSET for your hardware, then re-tune
the filter and gain blocks to taste.
"""

from typing import Final

import numpy as np

# ==========================================
# SERIAL LINK
# ==========================================
SERIAL_PORT: Final[str] = 'COM7'   # 'COM3' on Windows, '/dev/ttyUSB0' on Linux/Mac
BAUD_RATE: Final[int] = 115200

# ==========================================
# SENSOR & GEOMETRY
# ==========================================
LSB_TO_MGAUSS: Final[float] = 1.5  # IIS2MDC sensitivity (mGauss per LSB)

# Calibrated hard-iron + ambient baseline (mGauss).
# Output of calibration/calibrate_offset.py - re-run it to refresh this value.
B_OFFSET: Final = np.array([148.47, 662.57, 102.31])

# Inverse geometry unwarping matrix for sensor offset (0, 10, 5) mm.
INV_A: Final = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.2,  0.6],
    [ 0.0,  0.6, -0.7],
])

# Forward geometry matrix - calibration maps the fitted dipole-space sphere
# center back through A_FORWARD into raw-field offset units.
A_FORWARD: Final = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.4,  1.2],
    [ 0.0,  1.2, -0.4],
])

# ==========================================
# CALIBRATION
# ==========================================
CALIBRATION_SECONDS: Final[int] = 120  # duration of the roll-and-collect phase

# ==========================================
# FILTERING
# ==========================================
EMA_ALPHA: Final[float] = 0.70         # tracker_v1 plain EMA smoothing (0..1)
SPIKE_MEDIAN_TAPS: Final[int] = 3      # v2/v3: sliding median window size (odd)
SPIKE_MAX_DELTA_B: Final[float] = 8000.0  # v4: max permissible raw delta (mG/sample) for vector gate
OE_MIN_CUTOFF: Final[float] = 1.2      # Hz - lower = calmer at rest but laggier
OE_BETA: Final[float] = 0.015          # speed coefficient - raise if fast rolls lag
OE_D_CUTOFF: Final[float] = 1.0        # Hz - cutoff of the internal speed estimator

# ==========================================
# VELOCITY BALLISTICS (v2+)
# ==========================================
GAIN_SLOW: Final[float] = 700.0        # plot-units/rad near zero angular speed
GAIN_FAST: Final[float] = 4200.0       # plot-units/rad at high angular speed
W_REF_LOW: Final[float] = 0.35         # rad/s - below this: pure GAIN_SLOW
W_REF_HIGH: Final[float] = 5.0         # rad/s - above this: pure GAIN_FAST
BALLISTICS_GAMMA: Final[float] = 1.6   # >1 keeps the precision zone wide before ramping

# ==========================================
# TRACKER BEHAVIOUR
# ==========================================
ROTATION_DEADZONE: Final[float] = 0.0035  # v1 angular deadzone threshold (rad)
POINTER_GAIN: Final[float] = 1500.0       # v1 trajectory scaling factor
SPATIAL_DEADZONE: Final[float] = 5.0      # v2 plot-units of coherent motion before a step
DOMINANT_AXIS: Final[bool] = True         # v2 keep only larger |dX|/|dY| per frame

RESYNC_GAP: Final[float] = 0.5    # s - serial stall longer than this resyncs tracking
MAX_HISTORY: Final[int] = 2000    # v1/v2 max trajectory points kept in trace
VIEW_LIMIT: Final[float] = 500.0  # v1/v2 initial half-width of plot view
TRAIL_LEN: Final[int] = 600       # v3 frames of history shown per plane
LIMIT_FLOOR: Final[float] = 10.0  # v3 closest auto-zoom allowed (plot-units)

# ==========================================
# v4_1 / v5_1 PIPELINE TUNING
# ==========================================
SPIKE_GATE_MULTIPLIER: Final[float] = 3.0   # adaptive gate: reject > N x recent avg delta
DTHETA_NOISE_FLOOR: Final[float] = 0.0001  # rad - below this d_theta is sensor noise
LEAK_ALPHA: Final[float] = 0.9999          # leaky integrator retention per frame
OE_V4_1_D_CUTOFF: Final[float] = 3.0        # Hz - v4_1/v5 speed estimator cutoff (faster ramp-up)

# ==========================================
# v5_1 GESTURE & STROKE RECOGNIZER
# ==========================================
SWIPE_MIN_DISPLACEMENT: Final[float] = 0.08   # rad (~4.5 deg roll) minimum net stroke motion
SWIPE_MIN_OMEGA_PEAK: Final[float] = 1.0      # rad/s - minimum peak angular speed during stroke
SWIPE_SILENCE_TAPS: Final[int] = 4            # consecutive quiet frames (~40ms) to resolve stroke
SWIPE_COOLDOWN_SEC: Final[float] = 0.18       # s - cooldown after swipe to ignore return strokes
SWIPE_TILT_OFFSET_DEG: Final[float] = -15.0   # deg - hand biomechanical tilt compensation
SWIPE_FLASH_DURATION: Final[float] = 0.60     # s - HUD flash duration for recognized gesture

