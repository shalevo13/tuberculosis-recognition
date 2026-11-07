"""
Unit test suite for TinyTB-Net, Squeeze-and-Excitation attention,
dataset loading, diagnostic evaluation metrics, and INT8 ESP32-S3 deployment.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import TinyTBNet, ResBlock, SEBlock, ConvBNReLU, count_parameters
from src.dataset import create_dataloaders, get_transforms
from src.train import calculate_metrics, evaluate, BEST_MODEL_PATH
from src.quantize import (
    analyze_esp32_s3_footprint,
    quantize_tinytb_net,
    QUANT_MODEL_PATH,
    ESP32_HEADER_PATH,
)


class TestModelArchitecture(unittest.TestCase):
    def test_se_block_forward(self):
        """Verify SEBlock preserves tensor dimensions and outputs finite values."""
        se = SEBlock(channels=32, reduction=8)
        x = torch.randn(2, 32, 28, 28)
        out = se(x)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_se_block_edge_cases(self):
        """Verify SEBlock handles edge cases: batch size 1, odd channels, and various reductions."""
        # Batch size 1
        se1 = SEBlock(channels=16, reduction=8)
        x1 = torch.randn(1, 16, 14, 14)
        out1 = se1(x1)
        self.assertEqual(out1.shape, (1, 16, 14, 14))

        # Odd channels (17 channels, minimum bottleneck channels = 4)
        se_odd = SEBlock(channels=17, reduction=4)
        x_odd = torch.randn(2, 17, 7, 7)
        out_odd = se_odd(x_odd)
        self.assertEqual(out_odd.shape, (2, 17, 7, 7))

        # Gating values are strictly within [0, 1] (Sigmoid range)
        scale = se1.fc(x1)
        self.assertTrue((scale >= 0.0).all() and (scale <= 1.0).all())

    def test_res_block_stride1_and_stride2(self):
        """Verify ResBlock with and without downsampling."""
        # Stride 1: same channels, no downsampling
        block_s1 = ResBlock(in_channels=16, out_channels=16, stride=1, use_se=True)
        x1 = torch.randn(2, 16, 56, 56)
        out1 = block_s1(x1)
        self.assertEqual(out1.shape, (2, 16, 56, 56))

        # Stride 2: downsampling and channel expansion
        block_s2 = ResBlock(in_channels=16, out_channels=32, stride=2, use_se=True)
        out2 = block_s2(x1)
        self.assertEqual(out2.shape, (2, 32, 28, 28))

    def test_tinytbnet_forward_and_params(self):
        """Verify TinyTBNet forward pass and parameter budget for ESP32-S3."""
        model = TinyTBNet(use_se=True)
        params = count_parameters(model)
        # Verify param count is under 330k (~320 KB INT8 budget)
        self.assertLess(params, 330_000, f"Param count {params} exceeds microcontroller budget")
        self.assertGreater(params, 300_000)

        # Batch forward
        x = torch.randn(2, 1, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 1))
        self.assertFalse(torch.isnan(logits).any())

        # Single item forward (batch size 1 edge case)
        x_single = torch.randn(1, 1, 224, 224)
        model.eval()
        with torch.no_grad():
            logits_single = model(x_single)
        self.assertEqual(logits_single.shape, (1, 1))

    def test_gradient_flow(self):
        """Verify end-to-end backprop flows gradients to all trainable parameters."""
        model = TinyTBNet(use_se=True)
        model.train()
        x = torch.randn(2, 1, 224, 224)
        target = torch.tensor([[1.0], [0.0]])
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient")


class TestDatasetAndMetadata(unittest.TestCase):
    def test_metadata_integrity(self):
        """Verify metadata.csv has expected columns, valid rows, and existing images."""
        meta_path = os.path.join(PROJECT_ROOT, "data", "metadata.csv")
        self.assertTrue(os.path.exists(meta_path), "metadata.csv missing")
        df = pd.read_csv(meta_path)
        self.assertEqual(len(df), 1103, f"Expected 1103 images, found {len(df)}")
        self.assertIn("filename", df.columns)
        self.assertIn("label", df.columns)
        self.assertIn("source", df.columns)

        # Verify binary labels
        unique_labels = set(df["label"].unique())
        self.assertEqual(unique_labels, {0, 1})

        # Verify sources
        sources = set(df["source"].unique())
        self.assertIn("NIAID TB Portals (Belarus)", sources)
        self.assertIn("Shenzhen Hospital", sources)
        self.assertIn("Montgomery County", sources)

        # Verify image existence (sample of 30 images across all sources)
        images_dir = os.path.join(PROJECT_ROOT, "data", "images")
        sample_rows = df.sample(n=min(30, len(df)), random_state=42)
        for _, row in sample_rows.iterrows():
            img_path = os.path.join(images_dir, row["filename"])
            self.assertTrue(os.path.exists(img_path), f"Image file not found: {img_path}")

    def test_dataloaders(self):
        """Verify dataloaders yield correct tensor shapes and types."""
        loaders = create_dataloaders(batch_size=8)
        train_loader = loaders["train_loader"]
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (8, 1, 224, 224))
        self.assertEqual(labels.shape, (8,))
        self.assertEqual(images.dtype, torch.float32)

    def test_path_independence(self):
        """Verify create_dataloaders resolves default data paths relative to PROJECT_ROOT."""
        loaders = create_dataloaders()
        self.assertEqual(len(loaders["train_df"]), 771)
        self.assertEqual(len(loaders["val_df"]), 166)
        self.assertEqual(len(loaders["test_df"]), 166)


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = calculate_metrics(y_true, y_probs, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], (2, 0, 0, 2))

    def test_boundary_and_tradeoff(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([0.4, 0.6, 0.4, 0.8])
        # at threshold 0.5:
        # preds: [0, 1, 0, 1]
        # TN=1, FP=1, FN=1, TP=1
        m50 = calculate_metrics(y_true, y_probs, threshold=0.5)
        self.assertEqual(m50["confusion_matrix"], (1, 1, 1, 1))
        self.assertEqual(m50["sensitivity"], 0.5)
        self.assertEqual(m50["specificity"], 0.5)

        # at threshold 0.35:
        # preds: [1, 1, 1, 1]
        # TN=0, FP=2, FN=0, TP=2
        m35 = calculate_metrics(y_true, y_probs, threshold=0.35)
        self.assertEqual(m35["sensitivity"], 1.0)
        self.assertEqual(m35["specificity"], 0.0)


class TestQuantizationAndESP32(unittest.TestCase):
    def test_checkpoint_safe_loading(self):
        """Verify checkpoint loads cleanly with modern PyTorch default weights_only=True."""
        self.assertTrue(os.path.exists(BEST_MODEL_PATH), "Checkpoint missing")
        checkpoint = torch.load(BEST_MODEL_PATH)  # Default weights_only=True
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("best_auc", checkpoint)
        self.assertIn("val_metrics", checkpoint)
        self.assertGreaterEqual(checkpoint["best_auc"], 0.90)

    def test_checkpoint_clinical_performance(self):
        """Verify trained model achieves clinical performance on the unseen test set."""
        checkpoint = torch.load(BEST_MODEL_PATH)
        model = TinyTBNet(use_se=checkpoint.get("use_se", True))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        loaders = create_dataloaders()
        test_m = evaluate(model, loaders["test_loader"], torch.nn.BCEWithLogitsLoss(), torch.device("cpu"))

        # Verify key clinical benchmarks
        self.assertGreaterEqual(test_m["roc_auc"], 0.88, f"ROC-AUC {test_m['roc_auc']} below threshold")
        self.assertGreaterEqual(test_m["accuracy"], 0.80, f"Accuracy {test_m['accuracy']} below threshold")
        self.assertGreaterEqual(test_m["sensitivity"], 0.80, f"Sensitivity {test_m['sensitivity']} below threshold")

    def test_esp32_memory_budget(self):
        """Verify model weights and activations fit within ESP32-S3 hardware specifications."""
        model = TinyTBNet(use_se=True)
        footprint = analyze_esp32_s3_footprint(model)

        # INT8 weights must fit within Flash (< 4MB) and be <= 320 KB
        self.assertLessEqual(footprint["int8_flash_kb"], 320.0, "Weights exceed 320 KB target budget")
        self.assertTrue(footprint["fits_flash"], "Weights do not fit within ESP32 SPI Flash")
        self.assertTrue(footprint["single_buffer_fits_sram"], "Single activation buffer exceeds internal SRAM")

    def test_int8_quantization_inference(self):
        """Verify INT8 FX quantized model runs inference, preserves shapes, and outputs finite logits."""
        model = TinyTBNet(use_se=True)
        model.eval()

        # Calibration with small dummy dataset
        dummy_calib = [(torch.randn(4, 1, 224, 224), torch.zeros(4)) for _ in range(3)]
        int8_model = quantize_tinytb_net(model, dummy_calib, num_batches=3)

        x = torch.randn(2, 1, 224, 224)
        out = int8_model(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertFalse(torch.isnan(out).any())

    def test_esp32_deployment_artifacts_exist(self):
        """Verify exported INT8 TorchScript model and C config header exist and are non-empty."""
        self.assertTrue(os.path.exists(QUANT_MODEL_PATH), "tinytb_net_int8.pt missing")
        self.assertTrue(os.path.exists(ESP32_HEADER_PATH), "tinytb_net_esp32_config.h missing")

        pt_size = os.path.getsize(QUANT_MODEL_PATH)
        self.assertGreater(pt_size, 300_000, "Quantized model file suspiciously small")

        with open(ESP32_HEADER_PATH, "r") as f:
            header_text = f.read()
        self.assertIn("TINYTB_WEIGHTS_SIZE_KB", header_text)
        self.assertIn("TINYTB_THRESHOLD_SCREENING", header_text)
        self.assertIn("tinytb_interpret_probability", header_text)


if __name__ == "__main__":
    unittest.main()
