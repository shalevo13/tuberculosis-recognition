#!/usr/bin/env python3
"""
ESP32 TinyTB-Net Live Inference & Verification Runner.

Features:
- PING / Health check over serial UART
- Full benchmark execution of on-flash reference samples
- Live streaming of raw or preprocessed 224x224 CXR images via UART
- Clinical triage classification (high-sensitivity screening threshold 0.35 vs standard 0.50)
- Auto-attach USB device from Windows host via usbipd-win if not yet forwarded
"""

import sys
import os
import time
import argparse
import subprocess
from PIL import Image
import numpy as np

try:
    import serial
except ImportError:
    print("pyserial is required: pip install pyserial")
    sys.exit(1)

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
TIMEOUT_PER_INFERENCE = 60  # seconds

def ensure_usb_forwarded(busid="1-3", hwid="10c4:ea60"):
    """Check if /dev/ttyUSB0 exists; if not, attempt attach via usbipd-win."""
    if os.path.exists(DEFAULT_PORT):
        return True
    
    print(f"{DEFAULT_PORT} not found in WSL. Checking Windows usbipd-win...")
    try:
        # Check if usbipd is available on Windows
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
            return os.path.exists(DEFAULT_PORT)
        else:
            print(f"Device {hwid} (busid {busid}) not currently detected on Windows host.")
            return False
    except Exception as e:
        print(f"Could not query Windows host via powershell: {e}")
        return False

def open_serial(port=DEFAULT_PORT, baud=DEFAULT_BAUD, reset=True):
    ser = serial.Serial(port, baud, timeout=1.0)
    if reset:
        # Toggle DTR/RTS to perform hardware reset
        ser.dtr = False
        ser.rts = True
        time.sleep(0.1)
        ser.rts = False
        time.sleep(0.5)
        ser.reset_input_buffer()
        time.sleep(0.5)
    return ser

def run_ping(ser):
    ser.reset_input_buffer()
    ser.write(b"PING\n")
    ser.flush()
    t0 = time.time()
    while time.time() - t0 < 3.0:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line == "PONG":
            return True
    return False

def run_benchmark(ser):
    print("\n--- Triggering On-Device Benchmark ---")
    ser.reset_input_buffer()
    ser.write(b"BENCHMARK\n")
    ser.flush()

    t0 = time.time()
    results = []
    while time.time() - t0 < 120.0:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"  [ESP32] {line}")
            if line.startswith("RESULT:"):
                results.append(line)
            if "BENCHMARK COMPLETE" in line:
                break
    return results

def stream_image(ser, img_path, raw=True):
    print(f"\n--- Loading and preprocessing image: {img_path} ---")
    img = Image.open(img_path).convert("L")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_data = np.array(img, dtype=np.uint8).tobytes()
    assert len(img_data) == 50176, f"Expected 50176 bytes, got {len(img_data)}"

    cmd = f"INFER_RAW {len(img_data)}\n" if raw else f"INFER_QUANT {len(img_data)}\n"
    ser.reset_input_buffer()
    ser.write(cmd.encode("ascii"))
    ser.flush()

    # Wait for READY
    ready = False
    t0 = time.time()
    while time.time() - t0 < 5.0:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if "READY" in line:
            ready = True
            break
        elif "ERROR" in line:
            print(f"ESP32 error: {line}")
            return None

    if not ready:
        print("Timeout waiting for READY from ESP32")
        return None

    print(f"Streaming {len(img_data)} bytes to ESP32 at {ser.baudrate} baud...")
    chunk_size = 512
    for i in range(0, len(img_data), chunk_size):
        ser.write(img_data[i:i+chunk_size])
        time.sleep(0.005)
    ser.flush()

    print("Data sent. Waiting for inference results (~19s)...")
    t0 = time.time()
    result_line = None
    while time.time() - t0 < TIMEOUT_PER_INFERENCE:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"  [ESP32] {line}")
            if line.startswith("RESULT:"):
                result_line = line
            if "----------------------------------------" in line and result_line:
                break
    return result_line

def main():
    parser = argparse.ArgumentParser(description="ESP32 TinyTB-Net Serial Runner")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port (/dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (115200)")
    parser.add_argument("--ping", action="store_true", help="Send PING to ESP32")
    parser.add_argument("--benchmark", action="store_true", help="Run on-flash benchmark")
    parser.add_argument("--image", help="Stream raw CXR image and perform inference")
    parser.add_argument("--no-reset", action="store_true", help="Do not toggle RTS/DTR on connect")
    args = parser.parse_args()

    if not os.path.exists(args.port):
        ensure_usb_forwarded()

    if not os.path.exists(args.port):
        print(f"ERROR: Serial port {args.port} is not available.")
        print("Please verify that the ESP32 is plugged in and forwarded via usbipd-win.")
        sys.exit(1)

    print(f"Connecting to ESP32 on {args.port} @ {args.baud} baud...")
    ser = open_serial(args.port, args.baud, reset=not args.no_reset)

    try:
        if args.ping:
            ok = run_ping(ser)
            print("PING -> PONG: SUCCESS" if ok else "PING: TIMEOUT")

        if args.image:
            res = stream_image(ser, args.image, raw=True)
            if res:
                print(f"\nFinal Result: {res}")
            else:
                print("\nInference failed.")

        if args.benchmark or (not args.ping and not args.image):
            results = run_benchmark(ser)
            print(f"\nBenchmark finished. Captured {len(results)} inference results.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
