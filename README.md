# TinyTB-Net: Edge AI Tuberculosis Screening on ESP32 Microcontrollers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Embedded: ESP32-S3](https://img.shields.io/badge/Target-ESP32--S3-red.svg)](https://www.espressif.com/)
[![Platform: PlatformIO](https://img.shields.io/badge/PlatformIO-ESP--IDF%20%2F%20Arduino-orange.svg)](https://platformio.org/)

An end-to-end, mathematically grounded, and hardware-verified Edge AI pipeline for screening Chest Radiographs (CXRs) for Pulmonary Tuberculosis (TB) directly on microcontrollers ($320\text{ KB}$ SRAM / $4\text{ MB}$ Flash).

---

## 📌 Key Highlights

- **Edge Footprint:** Full model weights compressed to **305.8 KB** via INT8 Post-Training Quantization (PTQ). Consumes only **~7.5%** of standard 4 MB ESP32 Flash.
- **SRAM-Friendly Peak Activation:** Single-buffer peak activation memory is only **196 KB** ($112 \times 112 \times 16$), fitting within the ESP32-S3 internal SRAM budget.
- **Clinical Performance:**
  - **Validation ROC-AUC:** **0.9010**
  - **Test ROC-AUC:** **0.8932** (FP32) $\to$ **0.8939** (INT8, **zero metric degradation**)
  - **Screening Sensitivity (Threshold 0.35):** **88.57%** (catches 93/105 confirmed TB infections in unseen multi-center testing)
- **Zero-Overhead Embedded Web Server:** Self-contained single-page diagnostic GUI embedded as a gzipped PROGMEM C header (**4.3 KB** Flash).
- **Client-Side Canvas Preprocessing:** Smartphone/desktop uploads (3–5 MB JPEGs) are resized and converted to Rec. 601 grayscale ($224 \times 224 \times 1$) in-browser, transmitting an exact **50,176-byte** raw payload to prevent microcontroller heap exhaustion.

---

## 🏥 Clinical Screening vs. Traditional Accuracy

In computer vision benchmarks, overall **Accuracy** is standard. In medical screening, accuracy can be dangerously misleading: treating a false alarm (sending a healthy person for confirmation) with the same penalty as a false negative (sending a contagious TB patient home).

```
                      GROUND TRUTH (Unseen Test Set - 166 Patients)
                               Has TB (105)        Healthy (61)
                       ┌───────────────────────┬───────────────────────┐
AI Said "TB"           │  True Positive (TP)   │ False Positive (FP)   │
(Risk ≥ 0.35)          │         93            │         24            │
                       ├───────────────────────┼───────────────────────┤
AI Said "Healthy"      │ False Negative (FN)   │  True Negative (TN)   │
(Risk < 0.35)          │         12            │         37            │
                       └───────────────────────┴───────────────────────┘
```

Using a calibrated clinical threshold ($\tau = 0.35$) paired with a weighted loss ($w_{\text{pos}} = 1.28$), the model operates as a high-sensitivity triage filter:

| Operating Threshold | Sensitivity (Recall) | Specificity | Accuracy | Clinical Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **$\tau = 0.35$** | **88.57% (93/105)** | 60.66% (37/61) | 78.31% | **Rural Screening & Triage (Default)** |
| **$\tau = 0.40$** | **86.67% (91/105)** | 70.49% (43/61) | 80.72% | Balanced Multi-Center Screening |
| **$\tau = 0.45$** | **84.76% (89/105)** | 77.05% (47/61) | 81.93% | High-Accuracy Outpatient Triage |
| **$\tau = 0.50$** | **82.86% (87/105)** | 78.69% (48/61) | 81.33% | Standard Decision Cutoff |

---

## 🧠 Architecture: TinyTB-Net with SE Attention

TinyTB-Net adapts modern deep residual learning for ultra-low-power microcontrollers:

```
Input: 224 x 224 x 1 (Grayscale CXR)
  │
  ├─► Conv2d(1 -> 16, k=3, s=2, p=1) + BatchNorm + ReLU   [112 x 112 x 16] (Peak RAM: 196 KB)
  │
  ├─► ResBlock_SE_1: 16 -> 16 channels, stride=1          [112 x 112 x 16]
  │
  ├─► ResBlock_SE_2: 16 -> 32 channels, stride=2          [56 x 56 x 32]
  │
  ├─► ResBlock_SE_3: 32 -> 64 channels, stride=2          [28 x 28 x 64]
  │
  ├─► ResBlock_SE_4: 64 -> 128 channels, stride=2         [14 x 14 x 128]
  │
  ├─► AdaptiveAvgPool2d((1, 1))                           [1 x 1 x 128]
  │
  └─► Linear(128 -> 1) + Dropout(0.2)                     [Logit Scalar]
```

### Squeeze-and-Excitation (SE) Dynamic Gain Control
Each residual block features an SE channel attention module ($r=8$ reduction ratio) that dynamically recalibrates channel activations:
$$\mathbf{s} = \sigma \left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{x})) \right), \quad \tilde{\mathbf{x}} = \mathbf{s} \odot \mathbf{x}$$
- **Parameter Overhead:** $+5,776$ parameters total ($307\text{K} \to 313\text{K}$).
- **Peak Activation Memory Impact:** **0 bytes** (computations execute over a 1D channel vector).

---

## 🔬 Quantization Verification (FP32 vs INT8)

Evaluated on the 166-patient holdout test set:

| Diagnostic Metric | FP32 Baseline | INT8 Quantized (PTQ) | Quantization Delta |
| :--- | :--- | :--- | :--- |
| **Test ROC-AUC** | **0.8932** | **0.8939** | **+0.0007** (Zero degradation) |
| **Test Accuracy** | **81.33%** | **81.33%** | **0.00%** |
| **Sensitivity (Recall)** | **82.86% (87/105)** | **81.90% (86/105)** | -0.95% (1 patient difference) |
| **Specificity** | **78.69% (48/61)** | **80.33% (49/61)** | **+1.64%** |
| **Mean Absolute Error (Prob)** | — | **0.0115** | Average drift only 1.15% |
| **Logit MSE** | — | **0.0366** | Minimal logit distortion |

---

## 📂 Repository Structure

```
.
├── DEVLOG.md                      # Detailed technical engineering journal and derivations
├── data/
│   └── metadata.csv               # Dataset splits and clinical metadata (Montgomery, Shenzhen, Belarus)
├── firmware/                      # PlatformIO ESP32 embedded firmware project
│   ├── platformio.ini             # Hardware definitions for ESP32-S3
│   └── src/
│       ├── main.cpp               # FreeRTOS setup, Wi-Fi web server, and inference runner
│       ├── tinytb_infer.h         # C++ streaming forward-pass engine
│       ├── tinytb_weights.h       # INT8 weights and quantization scales
│       ├── tinytb_samples.h       # Embedded clinical test CXR samples
│       └── tinytb_web_page.h      # Gzipped single-page web GUI in PROGMEM
├── models/
│   ├── best_tinytb_net.pth        # Trained PyTorch FP32 checkpoint
│   ├── tinytb_net_int8.pt         # Traced PyTorch FX INT8 quantized model
│   ├── tinytb_net_esp32_config.h  # C header with model dimensions and thresholds
│   └── tinytb_net_web_page.h      # Generated web UI header
├── reports/
│   └── training_curves.png        # Loss, Accuracy, and ROC-AUC training progression
├── scripts/
│   ├── export_c_model.py          # Quantized weight exporter to C headers
│   ├── test_engine_c.cpp          # Host C++ test harness for exact numerical verification
│   ├── test_streaming_engine.py   # Verification of memory-constrained line buffers
│   └── verify_esp32_hardware.py   # Serial communication and hardware benchmark script
├── src/
│   ├── dataset.py                 # PyTorch CXR Dataset with CLAHE & clinical augmentations
│   ├── model.py                   # TinyTB-Net architecture definition with SE blocks
│   ├── train.py                   # PyTorch training loop with cosine annealing & pos_weight
│   ├── quantize.py                # Graph Mode Post-Training Quantization pipeline
│   └── export_web_header.py       # Minifies & gzips web/index.html to PROGMEM C header
├── tests/
│   ├── test_pipeline.py           # Unit tests for model forward pass and loss functions
│   └── test_web.py                # Tests for client-side web endpoints and preprocessing
└── web/
    ├── index.html                 # Darkroom diagnostic single-page web application
    └── server.py                  # Lightweight standalone HTTP simulation server
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/shalevo13/tuberculosis-recognition.git
cd tuberculosis-recognition

python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas pillow opencv-python matplotlib scikit-learn
```

### 2. Run Local Web Diagnostic UI

Launch the local web server to test image uploads and run live inference against the trained PyTorch model:

```bash
python web/server.py 8088
```

Open `http://localhost:8088` in your browser. Upload any standard JPEG/PNG chest radiograph, or click the preloaded test cases to observe client-side canvas downsampling and diagnostic risk scoring.

### 3. Evaluate & Quantize

Run the Post-Training Quantization script to generate calibration statistics, verify INT8 accuracy, and export model artifacts:

```bash
python src/quantize.py
```

### 4. Build & Flash ESP32 Firmware

To compile and flash to an ESP32-S3 using [PlatformIO](https://platformio.org/):

```bash
cd firmware
pio run -t upload
pio device monitor
```

Once booted, the ESP32 hosts a soft Access Point (`TinyTB-Diagnostic-AP`) or connects to local Wi-Fi, serving the web interface directly from its internal Flash.

---

## 📜 Dataset Provenance & Attribution

This model is trained on a combined multi-center cohort ($N = 1,103$ frontal chest radiographs):
1. **Montgomery County CXR Set:** National Library of Medicine (NLM), USA.
2. **Shenzhen No. 3 Hospital Set:** Guangdong Medical College, Shenzhen, China.
3. **NIAID TB Portals Belarus Set:** Culture-confirmed pulmonary TB cases from Minsk, Belarus (Open-access mirror from *Nature Scientific Reports*, 2019).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
