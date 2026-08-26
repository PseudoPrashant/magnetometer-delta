"""Robust hard-iron offset calibration for the magnetometer-delta rig.

Workflow:
    1. Tape the PCB face-down, keep metal/watches away.
    2. Roll continuously for CALIBRATION_SECONDS while samples stream in.
    3. The collected field traces a sphere in dipole space (INV_A-unwarped);
       a geometric sphere fit locates its center.
    4. The center maps back through A_FORWARD into raw-field units; paste the
       printed offset into B_OFFSET in config.py.

Run from the project root:  python -m calibration.calibrate_offset
"""

import time

import numpy as np
from scipy.optimize import minimize

from config import (
    A_FORWARD,
    BAUD_RATE,
    CALIBRATION_SECONDS,
    INV_A,
    LSB_TO_MGAUSS,
    SERIAL_PORT,
)


def collect_data() -> np.ndarray | None:
    """Stream CALIBRATION_SECONDS of raw serial samples, scaled to mGauss."""
    try:
        import serial

        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:  # port missing/busy, driver errors
        print(f"[-] Serial error: {e}")
        return None

    print(f"[*] Tape PCB down. Keep metal/watches away. Roll continuously for {CALIBRATION_SECONDS}s...")
    start_time = time.time()
    samples: list[list[float]] = []
    while time.time() - start_time < CALIBRATION_SECONDS:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 3:
            try:
                samples.append([float(p) * LSB_TO_MGAUSS for p in parts])
            except ValueError:
                continue  # malformed line - skip
    ser.close()
    return np.array(samples)


def fit_robust_dipole(data: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a sphere to dipole-space samples; returns (b_offset, radius)."""
    # 1. Transform to dipole space
    m_space = (INV_A @ data.T).T

    # 2. Geometric sphere objective: sum of (||M - c|| - R)^2
    def loss(params: np.ndarray) -> float:
        c = params[:3]
        r = params[3]
        dist = np.linalg.norm(m_space - c, axis=1)
        return float(np.sum((dist - r) ** 2))

    # Initial guess using min-max midpoint
    c0 = (m_space.max(axis=0) + m_space.min(axis=0)) / 2.0
    res = minimize(loss, np.append(c0, 7320.0), method='Nelder-Mead')

    b_offset = A_FORWARD @ res.x[:3]
    return b_offset, float(res.x[3])


if __name__ == '__main__':
    data = collect_data()
    if data is not None and len(data) > 200:
        offset, radius = fit_robust_dipole(data)
        print("\n" + "=" * 45)
        print(f"ROBUST OFFSET (mG): [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}]")
        print(f"RADIUS: {radius:.2f} mG")
        print("=" * 45)
        print("\nPaste into config.py:")
        print(f"B_OFFSET = np.array([{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}])")
