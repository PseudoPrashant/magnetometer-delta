"""record_directional_stream.py — Continuous Directional Magnetometer Stream Recorder.

Records high-rate continuous serial streaming data from the IIS2MDC magnetometer at COM7.

Recording Protocol per Direction (2 trials per direction):
  • Trial 1 (20 s):
      - Phase 1A: 5 seconds Still Ball Baseline  (phase="STILL")
      - Phase 1B: 15 seconds Active Movement    (phase="MOVEMENT")
  • Pause: 2 seconds rest break before Trial 2
  • Trial 2 (20 s):
      - Phase 2A: 5 seconds Still Ball Baseline  (phase="STILL")
      - Phase 2B: 15 seconds Active Movement    (phase="MOVEMENT")

Outputs:
  Structured timestamped CSV in `recordings/stream_{direction}_{timestamp}.csv`
  with millisecond timestamps, raw LSB, raw mGauss, B_clean, and unwarped dipole m.

Run from project root:
  python -m calibration.record_directional_stream
  python -m calibration.record_directional_stream --port COM7
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Top-level imports with fallback if run directly
try:
    from config import BAUD_RATE, B_OFFSET, INV_A, LSB_TO_MGAUSS, SERIAL_PORT
except ImportError:
    # Allow direct execution: python calibration/record_directional_stream.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import BAUD_RATE, B_OFFSET, INV_A, LSB_TO_MGAUSS, SERIAL_PORT


STILL_DURATION_S: float = 5.0      # 5 seconds still ball baseline
MOVEMENT_DURATION_S: float = 15.0  # 15 seconds active directional movement
PAUSE_DURATION_S: float = 2.0      # 2 seconds pause between trials
TRIALS_PER_DIRECTION: int = 2      # 2 trials per direction


class DirectionalStreamRecorder:
    """Manages continuous serial streaming and structured protocol recording."""

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE) -> None:
        self.port = port
        self.baud = baud
        self.ser = None
        self.output_dir = Path("recordings")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> bool:
        """Establish serial connection with the sensor."""
        import serial

        print(f"\n[*] Connecting to magnetometer on {self.port} @ {self.baud} baud...")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
            time.sleep(1.8)
            self.ser.reset_input_buffer()
            print(f"[+] Connected successfully to {self.port}.")
            return True
        except serial.SerialException as e:
            print(f"[-] Failed to connect to {self.port}: {e}")
            print("[-] Please check that the ESP32 is plugged in and other serial monitors are closed.")
            return False

    def close(self) -> None:
        """Safely close serial port."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
            print("[+] Serial port closed.")

    def drain_buffer(self) -> None:
        """Flush old samples from serial buffer."""
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()

    def record_phase(
        self,
        direction: str,
        trial_num: int,
        phase_name: str,
        duration_s: float,
        session_start_perf: float,
        global_sample_idx: int,
    ) -> Tuple[List[dict], int]:
        """Record a single timed phase (STILL or MOVEMENT) continuously at 100 Hz."""
        phase_rows: List[dict] = []
        phase_start = time.perf_counter()
        last_print = 0.0
        prev_sample_time = phase_start

        phase_label = "KEEP BALL STILL" if phase_name == "STILL" else f"SWIPE {direction} NOW!"
        color_prefix = "\033[93m" if phase_name == "STILL" else "\033[92m"
        color_reset = "\033[0m"

        print(f"\n  {color_prefix}>>> [TRIAL {trial_num}/{TRIALS_PER_DIRECTION}] [{phase_name}] {phase_label} ({duration_s:.0f}s) <<<{color_reset}")

        while True:
            now = time.perf_counter()
            elapsed_phase = now - phase_start
            remaining = duration_s - elapsed_phase

            if elapsed_phase >= duration_s:
                break

            # Read all available bytes from serial
            if self.ser and self.ser.is_open:
                try:
                    lines = self.ser.read_all().decode("latin-1", errors="ignore").splitlines()
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

                    sample_time = time.perf_counter()
                    timestamp_ms = (sample_time - session_start_perf) * 1000.0
                    dt_ms = (sample_time - prev_sample_time) * 1000.0
                    prev_sample_time = sample_time

                    b_raw = np.array(raw_lsb, dtype=float) * LSB_TO_MGAUSS
                    b_clean = b_raw - B_OFFSET
                    m = np.dot(INV_A, b_clean)

                    global_sample_idx += 1
                    row = {
                        "timestamp_ms": f"{timestamp_ms:.2f}",
                        "dt_ms": f"{dt_ms:.2f}",
                        "sample_idx": global_sample_idx,
                        "direction": direction,
                        "trial": trial_num,
                        "phase": phase_name,
                        "phase_elapsed_s": f"{elapsed_phase:.3f}",
                        "b_raw_x": f"{b_raw[0]:.2f}",
                        "b_raw_y": f"{b_raw[1]:.2f}",
                        "b_raw_z": f"{b_raw[2]:.2f}",
                        "lsb_x": int(raw_lsb[0]),
                        "lsb_y": int(raw_lsb[1]),
                        "lsb_z": int(raw_lsb[2]),
                        "b_clean_x": f"{b_clean[0]:.2f}",
                        "b_clean_y": f"{b_clean[1]:.2f}",
                        "b_clean_z": f"{b_clean[2]:.2f}",
                        "m_x": f"{m[0]:.4f}",
                        "m_y": f"{m[1]:.4f}",
                        "m_z": f"{m[2]:.4f}",
                    }
                    phase_rows.append(row)

            # Live terminal progress update (~10 Hz refresh)
            if now - last_print >= 0.10:
                last_print = now
                progress_ratio = min(max(elapsed_phase / duration_s, 0.0), 1.0)
                bar_len = 25
                filled = int(progress_ratio * bar_len)
                bar = "=" * filled + "-" * (bar_len - filled)
                current_rate = len(phase_rows) / max(elapsed_phase, 0.001)
                
                # Terminal status line
                sys.stdout.write(
                    f"\r  [{bar}] {remaining:4.1f}s left | samples: {len(phase_rows):4d} | rate: {current_rate:4.1f} Hz "
                )
                sys.stdout.flush()

            time.sleep(0.005)

        sys.stdout.write(f"\r  [{'=' * 25}]  0.0s left | samples: {len(phase_rows):4d} | Phase Complete!      \n")
        sys.stdout.flush()

        return phase_rows, global_sample_idx

    def record_pause(self, duration_s: float = PAUSE_DURATION_S) -> None:
        """Perform a quiet pause between trials with a visual countdown."""
        print(f"\n  \033[90m>>> [PAUSE] {duration_s:.1f}s break before next trial... Prepare ball at center <<<\033[0m")
        start_pause = time.perf_counter()
        while time.perf_counter() - start_pause < duration_s:
            rem = duration_s - (time.perf_counter() - start_pause)
            sys.stdout.write(f"\r  [Pause]: {rem:3.1f}s remaining... ")
            sys.stdout.flush()
            time.sleep(0.05)
        sys.stdout.write("\r  [Pause]: Ready!                              \n")
        sys.stdout.flush()
        self.drain_buffer()

    def record_direction_session(self, direction: str) -> Optional[Path]:
        """Run the full 2-trial recording protocol for a single direction."""
        direction = direction.strip().upper()
        if not direction:
            direction = "CUSTOM"

        print("\n" + "=" * 70)
        print(f"RECORDING PROTOCOL FOR DIRECTION: {direction}")
        print(f"  • Trial 1: {STILL_DURATION_S:.0f}s STILL + {MOVEMENT_DURATION_S:.0f}s SWIPING '{direction}'")
        print(f"  • Break  : {PAUSE_DURATION_S:.0f}s PAUSE")
        print(f"  • Trial 2: {STILL_DURATION_S:.0f}s STILL + {MOVEMENT_DURATION_S:.0f}s SWIPING '{direction}'")
        print("=" * 70)

        input("\nPress [ENTER] to begin recording...")

        all_session_rows: List[dict] = []
        session_start_perf = time.perf_counter()
        global_sample_idx = 0

        self.drain_buffer()

        try:
            for trial_idx in range(1, TRIALS_PER_DIRECTION + 1):
                # Phase A: Still ball baseline (5s)
                still_rows, global_sample_idx = self.record_phase(
                    direction=direction,
                    trial_num=trial_idx,
                    phase_name="STILL",
                    duration_s=STILL_DURATION_S,
                    session_start_perf=session_start_perf,
                    global_sample_idx=global_sample_idx,
                )
                all_session_rows.extend(still_rows)

                # Phase B: Active directional movement (15s)
                move_rows, global_sample_idx = self.record_phase(
                    direction=direction,
                    trial_num=trial_idx,
                    phase_name="MOVEMENT",
                    duration_s=MOVEMENT_DURATION_S,
                    session_start_perf=session_start_perf,
                    global_sample_idx=global_sample_idx,
                )
                all_session_rows.extend(move_rows)

                # Pause between trials (if not last trial)
                if trial_idx < TRIALS_PER_DIRECTION:
                    self.record_pause(duration_s=PAUSE_DURATION_S)

        except KeyboardInterrupt:
            print("\n\n[!] Recording interrupted by user. Saving captured data...")

        if not all_session_rows:
            print("[-] No data captured during session.")
            return None

        # Save to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stream_{direction.lower()}_{timestamp_str}.csv"
        filepath = self.output_dir / filename

        columns = list(all_session_rows[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_session_rows)

        total_time_s = time.perf_counter() - session_start_perf
        avg_hz = len(all_session_rows) / max(total_time_s, 0.001)

        print("\n" + "=" * 70)
        print("RECORDING COMPLETE & SAVED")
        print("=" * 70)
        print(f"  • Direction       : {direction}")
        print(f"  • Total Samples   : {len(all_session_rows):,}")
        print(f"  • Total Duration  : {total_time_s:.2f} seconds")
        print(f"  • Effective Rate  : {avg_hz:.1f} Hz")
        print(f"  • Output File     : {filepath.resolve()}")
        print("=" * 70)

        return filepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous Directional Magnetometer Stream Recorder (2x 20s trials per direction)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=SERIAL_PORT,
        help=f"Serial port to connect (default: {SERIAL_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Baud rate (default: {BAUD_RATE})",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Direction name to record immediately without interactive prompt",
    )
    args = parser.parse_args()

    recorder = DirectionalStreamRecorder(port=args.port, baud=args.baud)

    if not recorder.connect():
        sys.exit(1)

    try:
        if args.direction:
            recorder.record_direction_session(args.direction)
        else:
            # Interactive loop: prompts for direction and allows recording multiple directions
            while True:
                print("\n-------------------------------------------------------------")
                user_dir = input("Enter swipe direction to record (e.g. UP, DOWN, LEFT, RIGHT, or 'q' to quit): ").strip()
                if not user_dir or user_dir.lower() in ("q", "quit", "exit"):
                    break

                recorder.record_direction_session(user_dir)

                again = input("\nWould you like to record another direction? [y/N]: ").strip().lower()
                if again != "y":
                    break

    finally:
        recorder.close()

    print("\n[+] Recording session finished. Goodbye!")


if __name__ == "__main__":
    main()
