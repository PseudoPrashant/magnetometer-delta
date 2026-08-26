# Hardware Setup

The Python side expects an ESP32 streaming IIS2MDC magnetometer data over USB
serial. This page documents the contract the host relies on and the physical
arrangement the geometry constants encode.

## Serial protocol contract

The host parser (`trackers/*.py`, `calibration/calibrate_offset.py`) assumes:

- **Line format:** three comma-separated numeric fields + newline:
  ```
  -1234,567,89\n
  ```
- **Units:** raw LSB counts (host multiplies by `LSB_TO_MGAUSS = 1.5`).
- **Rate:** ~100 Hz (filters seed with a nominal 10 ms interval; much slower
  rates still work but reduce effective smoothing quality).
- **Baud:** `115200` (`config.BAUD_RATE`).
- **Robustness:** blank lines, malformed lines, and non-finite values are
  skipped silently — the host never crashes on garbage input, it just drops
  samples.

Anything meeting that contract works — the host does not care whether the
source is an ESP32, a logic analyzer replay, or another board.

## Sensor & mounting

- **Sensor:** ST IIS2MDC magnetometer on I2C (firmware reportedly runs the
  bus at 400 kHz, sampling continuously).
- **Mounting offset:** the documented mechanical arrangement places the
  sensor at **(0, 10, 5) mm** relative to the sphere's roll pivot. This exact
  offset is what `INV_A` unwraps (and `A_FORWARD` inverts). If you move the
  sensor, those matrices no longer describe your rig — trajectory quality
  will degrade until they are rederived for the new geometry.
- **Rolling element:** any small magnetized sphere whose dipole direction
  rotates as it rolls. Stronger magnets give a larger fitted radius and
  better signal margin above ambient noise.

## Environment

- The measured field is dipole + hard iron + **Earth's field** (~250–650 mG
  total depending on location). That ambient baseline is part of why
  `B_OFFSET` must be calibrated per location — see
  [calibration guide](calibration-guide.md).
- Keep ferromagnetic objects (steel desks, stands, watches, speakers, phones)
  away during both calibration and use. Moving such an object after
  calibration invalidates the offset.
- Calibration runs **PCB face-down, untouched**, so the sphere rolls freely.

## Ports

| OS | Typical name | Notes |
|---|---|---|
| Windows | `COM7` (current default) | Device Manager → Ports (COM & LPT); pick the "USB Serial Device" entry |
| Linux | `/dev/ttyUSB0` | Add yourself to `dialout` group or run with appropriate udev rules |
| macOS | `/dev/tty.usbserial-*` | Listed under `/dev/` after plugging in |

Edit `SERIAL_PORT` in `config.py` to match. Only one process can hold the
port — close Arduino IDE serial monitors and other terminals before starting
a tracker.

## Firmware expectations (for firmware-side work)

If you modify or replace the ESP32 sketch, preserve these properties:

1. Continuous sampling at ~100 Hz with fixed line framing (newline-
   terminated CSV).
2. Stable LSB scaling consistent with `LSB_TO_MGAUSS = 1.5` (IIS2MDC default
   ±4 gauss range ⇒ 1.5 mG/LSB).
3. No extra text on the data stream — debug prints corrupt parsing. Use a
   second interface for logs, or gate debug output behind a build flag.
4. Timestamps are taken host-side; firmware need not send time information.

## See also

- [Architecture](architecture.md) — how the host consumes the stream
- [Calibration guide](calibration-guide.md) — producing `B_OFFSET` for your physical setup
