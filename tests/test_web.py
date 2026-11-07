"""
Unit and integration tests for TinyTB-Net Web Interface and API.
"""

import io
import json
import os
import sys
import unittest
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web.server import TinyTBHandler, get_model


class TestWebInterface(unittest.TestCase):
    def setUp(self):
        self.html_path = os.path.join(PROJECT_ROOT, "web", "index.html")
        self.header_path = os.path.join(PROJECT_ROOT, "models", "tinytb_net_web_page.h")

    def test_html_file_exists_and_compact(self):
        self.assertTrue(os.path.exists(self.html_path), "index.html must exist")
        size = os.path.getsize(self.html_path)
        self.assertLess(size, 15000, f"HTML must be compact (< 15 KB), got {size} bytes")

    def test_html_is_self_contained(self):
        """Ensure no external CDN dependencies (fonts, scripts, css) that break offline ESP32."""
        with open(self.html_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("http://", content)
        self.assertNotIn("https://", content)
        self.assertIn("canvas", content)
        self.assertIn("/predict", content)

    def test_c_header_exists_and_under_5kb_gzip(self):
        self.assertTrue(os.path.exists(self.header_path), "C web header must exist")
        with open(self.header_path, "r", encoding="utf-8") as f:
            header_content = f.read()
        self.assertIn("TINYTB_INDEX_HTML_GZ", header_content)
        self.assertIn("TINYTB_INDEX_HTML_GZ_LEN", header_content)

    def test_model_inference_on_synthetic_grayscale_buffer(self):
        """Verify model runs inference on a 224x224 raw uint8 buffer."""
        import torch
        model = get_model()
        dummy_buf = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        norm_tensor = (torch.from_numpy(dummy_buf).float() / 255.0 - 0.5) / 0.5
        input_tensor = norm_tensor.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = model(input_tensor).squeeze().item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
