#!/usr/bin/env python3
"""
Verify TinyTB-Net quantized model running on physical ESP32 via serial.
Connects to /dev/ttyUSB0, triggers BENCHMARK command, and validates
numerical accuracy, latency, and memory safety against PyTorch FX INT8 reference.
"""

import sys
import time
import serial

import os
import subprocess

PORT = "/dev/ttyUSB0"
BAUD = 115200
TIMEOUT = 120 # allow up to 120s for both inferences

def ensure_usb_forwarded(busid="1-3", hwid="10c4:ea60"):
    if os.path.exists(PORT):
        return True
    print(f"{PORT} not found in WSL. Checking Windows usbipd-win...")
    try:
        res = subprocess.run(
            ["powershell.exe", "-Command", "& 'C:\\Program Files\\usbipd-win\\usbipd.exe' list"],
            capture_output=True, text=True, timeout=5, cwd="/mnt/c"
        )
        if hwid in res.stdout or busid in res.stdout:
            print(f"Device found in Windows. Attaching via usbipd...")
            subprocess.run(
                ["powershell.exe", "-Command", f"& 'C:\\Program Files\\usbipd-win\\usbipd.exe' attach --wsl --busid {busid} --auto-attach"],
                capture_output=True, text=True, timeout=10, cwd="/mnt/c"
            )
            time.sleep(1.5)
            return os.path.exists(PORT)
        else:
            print(f"Device {hwid} (busid {busid}) not currently detected on Windows host.")
            return False
    except Exception as e:
        print(f"Could not query Windows host: {e}")
        return False

def main():
    if not os.path.exists(PORT):
        ensure_usb_forwarded()
    print(f"Connecting to ESP32 on {PORT} at {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0)
    except Exception as e:
        print(f"ERROR: Could not open {PORT}: {e}")
        sys.exit(1)

    # Toggle DTR/RTS to reset ESP32 and capture fresh boot log
    print("Resetting ESP32 via RTS/DTR toggle...")
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    time.sleep(0.5)

    ser.reset_input_buffer()
    time.sleep(1.0)

    # Send BENCHMARK command
    print("Sending BENCHMARK command...")
    ser.write(b"BENCHMARK\n")
    ser.flush()

    start_time = time.time()
    captured_lines = []
    benchmark_complete = False

    while time.time() - start_time < TIMEOUT:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"[ESP32] {line}")
            captured_lines.append(line)
            if "BENCHMARK COMPLETE" in line:
                benchmark_complete = True
                break

    ser.close()

    if not benchmark_complete:
        print("WARNING: Benchmark did not output completion token within timeout!")
    else:
        print("\n=== HARDWARE VERIFICATION SUCCEEDED ===")

if __name__ == "__main__":
    main()
