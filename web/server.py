"""
Lightweight Local HTTP Server for TinyTB-Net Web Diagnostic Interface.
Serves web/index.html and handles POST /predict for testing before ESP32 flashing.
"""

import io
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import TinyTBNet
from src.train import BEST_MODEL_PATH

MODEL = None
DEVICE = torch.device("cpu")


def get_model():
    global MODEL
    if MODEL is None:
        print(f"Loading TinyTB-Net model checkpoint from {BEST_MODEL_PATH}...")
        MODEL = TinyTBNet(use_se=True).to(DEVICE)
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.eval()
        print("Model loaded successfully.")
    return MODEL


class TinyTBHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_path = os.path.join(PROJECT_ROOT, "web", "index.html")
            with open(html_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif parsed.path == "/health":
            resp = json.dumps({"status": "ok", "model": "TinyTB-Net (SE)", "device": str(DEVICE)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif parsed.path.startswith("/sample/"):
            fname = os.path.basename(parsed.path)
            img_path = os.path.join(PROJECT_ROOT, "data", "images", fname)
            if os.path.exists(img_path) and fname.endswith(".png"):
                with open(img_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/predict":
            query = parse_qs(parsed.query)
            threshold = float(query.get("threshold", [0.35])[0])

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_error(400, "Missing request body")
                return

            body = self.rfile.read(content_length)
            t0 = time.perf_counter()

            # Handle either raw 224x224 grayscale buffer (50,176 bytes) or standard image file
            if content_length == 224 * 224:
                # Raw uint8 array from browser canvas
                arr = np.frombuffer(body, dtype=np.uint8).reshape((224, 224))
            else:
                # Standard PNG/JPEG image upload
                try:
                    img = Image.open(io.BytesIO(body)).convert("L")
                    img = img.resize((224, 224), Image.Resampling.BILINEAR)
                    arr = np.array(img, dtype=np.uint8)
                except Exception as e:
                    self.send_error(400, f"Invalid image payload: {e}")
                    return

            # Preprocessing normalization: (x / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0]
            norm_tensor = (torch.from_numpy(arr).float() / 255.0 - 0.5) / 0.5
            input_tensor = norm_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, 224, 224]

            model = get_model()
            with torch.no_grad():
                logits = model(input_tensor).squeeze()
                prob = torch.sigmoid(logits).item()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            is_tb = bool(prob >= threshold)

            result = {
                "probability": round(prob, 4),
                "threshold": threshold,
                "is_tuberculosis": is_tb,
                "diagnosis": (
                    "Tuberculosis Detected (Flag for Confirmatory GeneXpert)"
                    if is_tb
                    else "Normal / Cleared"
                ),
                "operating_mode": (
                    "High Sensitivity Screening Mode"
                    if threshold <= 0.35
                    else "Standard Diagnostic Mode"
                ),
                "inference_time_ms": round(elapsed_ms, 2)
            }

            resp_bytes = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        else:
            self.send_response(404)
            self.end_headers()


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_server(start_port=8088, max_attempts=20):
    get_model()
    httpd = None
    port = start_port

    for attempt in range(max_attempts):
        try:
            server_address = ("0.0.0.0", port)
            httpd = ReusableHTTPServer(server_address, TinyTBHandler)
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"Port {port} is already in use, trying port {port + 1}...")
                port += 1
            else:
                raise

    if httpd is None:
        raise RuntimeError(f"Could not bind to any port between {start_port} and {start_port + max_attempts - 1}")

    print("=" * 60)
    print(f"TinyTB-Net Web Diagnostic Server running on http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  GET  /        -> Web App UI (Single-file HTML canvas)")
    print(f"  POST /predict  -> Run inference on raw 224x224 buffer")
    print(f"  GET  /health  -> Healthcheck")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    initial_port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8088))
    run_server(start_port=initial_port)
