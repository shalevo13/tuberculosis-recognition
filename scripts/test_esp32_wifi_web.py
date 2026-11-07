#!/usr/bin/env python3
"""
Test script for ESP32 TinyTB-Net Wi-Fi & Web Interface.
Reads serial boot logs to retrieve the assigned IP address, then exercises HTTP endpoints:
- GET /health
- GET / (web dashboard)
- POST /predict (TB positive)
- POST /predict (TB negative / Normal)
"""

import sys
import time
import json
import urllib.request
import urllib.error
import serial
from PIL import Image
import numpy as np

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

def get_ip_from_serial(timeout=20):
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1.0)
    # Don't toggle DTR/RTS aggressively if already booted, but let's read lines
    start_time = time.time()
    ip_address = None
    print("Listening for Wi-Fi boot logs...")
    
    # Send an INFO command just in case it's already in the loop
    ser.write(b"INFO\n")
    ser.flush()
    
    while time.time() - start_time < timeout:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"[ESP32] {line}")
            if "[WiFi] IP Address:" in line:
                ip_address = line.split("[WiFi] IP Address:")[1].strip()
                break
            elif "IP Address:" in line:
                parts = line.split("IP Address:")
                candidate = parts[1].strip().split()[0]
                if candidate.count(".") == 3:
                    ip_address = candidate
                    break
            elif "[WiFi] SoftAP IP:" in line:
                ip_address = line.split("[WiFi] SoftAP IP:")[1].strip()
                break
    
    ser.close()
    return ip_address

def test_endpoints(ip):
    base_url = f"http://{ip}"
    print(f"\n==========================================")
    print(f"Testing ESP32 Web Server at {base_url}")
    print(f"==========================================")
    
    # 1. Health check
    print(f"\n1. GET {base_url}/health")
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            body = response.read().decode('utf-8')
            print(f"   Status: {status}")
            print(f"   Response: {body}")
    except Exception as e:
        print(f"   Failed: {e}")
        return False
        
    # 2. Main Web Dashboard
    print(f"\n2. GET {base_url}/")
    try:
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            content_encoding = response.headers.get("Content-Encoding", "none")
            content_length = response.headers.get("Content-Length", "unknown")
            body = response.read()
            if content_encoding == "gzip":
                import gzip
                body = gzip.decompress(body)
            print(f"   Status: {status}")
            print(f"   Encoding: {content_encoding}, Compressed: {content_length} bytes, Decompressed: {len(body)} bytes")
            assert "TinyTB-Net" in body.decode('utf-8', errors='ignore')
            print("   UI Dashboard loaded successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"   Failed: {e}")
        return False

    # Helper for image inference POST
    def run_image_post(img_path, label):
        print(f"\n3. POST {base_url}/predict ({label}: {img_path})")
        img = Image.open(img_path).convert("L").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        raw_bytes = arr.tobytes()
        assert len(raw_bytes) == 50176
        
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=raw_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(raw_bytes))
            },
            method="POST"
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as response:
            elapsed = time.time() - t0
            status = response.status
            body = response.read().decode('utf-8')
            print(f"   HTTP Status: {status} in {elapsed:.2f}s")
            print(f"   Inference JSON: {body}")
            res_json = json.loads(body)
            return res_json

    tb_res = run_image_post("data/images/CHNCXR_0404_1.png", "TB Positive")
    norm_res = run_image_post("data/images/CHNCXR_0001_0.png", "Normal / TB Negative")
    
    print("\n==========================================")
    print("VERIFICATION SUMMARY:")
    print(f"TB Case: Probability = {tb_res.get('probability')*100:.2f}%, Screening = {tb_res.get('screening', tb_res.get('screening_flag'))}, Standard = {tb_res.get('standard', tb_res.get('standard_flag'))}")
    print(f"Normal Case: Probability = {norm_res.get('probability')*100:.2f}%, Screening = {norm_res.get('screening', norm_res.get('screening_flag'))}, Standard = {norm_res.get('standard', norm_res.get('standard_flag'))}")
    print(f"ESP32 On-Chip Latency: TB={tb_res.get('latency_ms')}ms, Normal={norm_res.get('latency_ms')}ms")
    print("==========================================")
    return True

if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else None
    if not ip:
        ip = get_ip_from_serial(timeout=20)
    
    if not ip:
        print("ERROR: Could not retrieve IP address from ESP32.")
        sys.exit(1)
        
    print(f"\nESP32 IP identified: {ip}")
    success = test_endpoints(ip)
    if not success:
        sys.exit(1)
