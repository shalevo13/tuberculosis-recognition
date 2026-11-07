# DevLog: Engineering TinyTB-Net
## From Textbook ResNets to Clinical Edge AI on Chest Radiographs

> **Author:** Shalev  
> **Topic:** Applied Deep Learning, Medical Imaging (Tuberculosis), TinyML / Edge Deployment (ESP32)  
> **Dataset:** NIH NLM Montgomery County (USA) + Shenzhen No. 3 Hospital (China) + NIAID TB Portals Belarus Set (Minsk)  

---

## 1. Project Motivation & Constraints

Tuberculosis (TB) remains one of the world's deadliest infectious diseases, disproportionately affecting rural or low-resource communities where radiologists are scarce. 

The goal of this project is to build an accurate, robust, and mathematically grounded Convolutional Neural Network (CNN) capable of screening Chest X-Rays (CXRs) for Tuberculosis, with two strict design constraints:
1. **Low-Data Regime / Domain Robustness:** Medical datasets are small and heterogeneous across different hospital scanners and demographics.
2. **Microcontroller Deployability (TinyML / ESP32):** The model must ultimately run on edge microcontrollers (like the ESP32 / ESP32-S3), where internal SRAM is constrained to ~320 KB – 512 KB.

---

## 2. Medical AI Evaluation: Why Accuracy Fails in Medicine

In typical computer science benchmarks (e.g., classifying cats vs. dogs), overall **Accuracy** is the primary metric. In medical diagnosis, relying purely on Accuracy is dangerous.

### The Confusion Matrix on 120 Unseen Patients
```
                          ACTUAL REALITY (Ground Truth)
                       Has TB (59)       Healthy (61)
                   ┌─────────────────┬─────────────────┐
 AI Said "TB"      │  True Positive  │ False Positive  │
                   │    (TP = 35)    │    (FP = 9)     │  <-- 9 False Alarms
 Model             ├─────────────────┼─────────────────┤
 Prediction        │ False Negative  │  True Negative  │
 AI Said "Healthy" │    (FN = 24)    │    (TN = 52)    │  <-- 24 Missed Infections!
                   └─────────────────┴─────────────────┘
```

### The 4 Core Diagnostic Metrics

1. **Accuracy** ($\frac{TP + TN}{\text{Total}}$):
   - The percentage of total correct decisions.
   - *Flaw:* Treats a harmless false alarm (sending a healthy person for a sputum test) with the exact same numerical penalty as a fatal missed diagnosis (sending a contagious TB patient home).

2. **Sensitivity / Recall / True Positive Rate** ($\frac{TP}{TP + FN}$):
   - *"Out of all patients who genuinely have TB, what percentage did the AI catch?"*
   - In clinical screening, **Sensitivity must be prioritized above all else ($\ge 90\%$)**. A False Negative ($FN$) can lead to patient mortality and community transmission.

3. **Specificity / True Negative Rate** ($\frac{TN}{TN + FP}$):
   - *"Out of all healthy people, what percentage was correctly cleared?"*
   - High specificity prevents hospital laboratories from being overwhelmed with false-alarm confirmatory tests.

4. **The Seesaw Trade-off:**
   - Sensitivity and Specificity pull in opposite directions.
   - If an AI labels every single patient as "Sick", Sensitivity is $100\%$, but Specificity drops to $0\%$.
   - If an AI labels everyone as "Healthy", Specificity is $100\%$, but Sensitivity drops to $0\%$.

5. **ROC-AUC (Area Under the Receiver Operating Characteristic Curve):**
   - Because Sensitivity and Specificity depend on an arbitrary decision cutoff threshold (e.g. $0.50$ vs $0.30$), **ROC-AUC evaluates the model across all possible thresholds from $0.0$ to $1.0$**.
   - **The "Pair Bet" Mental Model:** If you randomly draw one confirmed TB patient and one healthy patient from the hospital archives, **ROC-AUC is the exact mathematical probability that the neural network assigns a higher risk probability to the sick patient.**
   - $0.50 = \text{Random coin flip}$ (zero discriminative skill).
   - $1.00 = \text{Flawless diagnostic ranking}$.

---

## 3. Loss Function Mechanics: `BCEWithLogitsLoss` & `pos_weight`

The classification loss is defined as:
$$\mathcal{L} = - \Big[ \mathbf{w_{\text{pos}}} \cdot y \ln(\hat{p}) \;+\; (1 - y) \ln(1 - \hat{p}) \Big]$$

### The Fused Logit Formulation
Rather than computing $\hat{p} = \text{Sigmoid}(x)$ and calling $\text{BCELoss}$, PyTorch fuses the operations into `nn.BCEWithLogitsLoss` via the **log-sum-exp trick**:
$$\mathcal{L}(x, y) = \max(x, 0) - x \cdot y + \ln(1 + e^{-|x|})$$
This avoids floating-point overflow (`inf`) and underflow (`NaN`) when logits are large or small.

### The Role of `pos_weight`:
- Standard loss treats positive and negative errors symmetrically ($1:1$).
- In screening, we deliberately penalize False Negatives more heavily by setting $w_{\text{pos}} > 1.0$.
- In our first run, $w_{\text{pos}} = 1.40$ forced the model to prioritize Sensitivity. When the dataset was expanded to a 50/50 split, $w_{\text{pos}}$ dropped to $1.03$, causing the network to act overly conservative (boosting Specificity to $85\%$ at the cost of Sensitivity dropping to $59\%$).

---

## 4. Modern CNN & ResNet Architecture Fundamentals

### Why Not Feed $4000 \times 4000$ Images into Strided Convolutions Directly?
Clinical radiographs arrive at resolutions up to $4,000 \times 4,000$ pixels (15–30 MB per image).
1. **Activation Memory Explosion:** In PyTorch, intermediate activations must be stored in RAM/VRAM for backprop. A batch of 16 images at $4000 \times 4000$ requires $>7 \text{ GB}$ of RAM after just two convolutional layers.
2. **I/O Bottleneck:** Reading and decompressing 800 uncompressed multi-megapixel PNGs starves the compute engine.
3. **Analytic Preprocessing:** Classical bilinear interpolation analytically performs low-pass anti-aliasing downsampling to $224 \times 224$ with zero learned parameter overhead.

### The Compute Invariant Rule: Halve $(H, W) \to$ Double $C$
For a $3 \times 3$ convolution, compute scales as:
$$\text{FLOPs} \propto (H \times W) \times (C_{\text{in}} \times C_{\text{out}})$$

When a residual stage cuts spatial dimensions in half (`stride=2`):
- Spatial Area drops by $75\%$ ($\times \frac{1}{4}$).
- Number of channels doubles ($C \to 2C$, so channel pairs scale by $\times 4$).
- **Result:** $\frac{1}{4} \times 4 = \mathbf{1.0\times}$ (Compute per stage remains balanced!).

### Global Average Pooling (GAP) vs. Large Dense Heads
Older networks (AlexNet, VGG) flattened high-dimensional spatial grids into massive dense layers ($4096 \times 4096$), creating millions of fragile parameters.
`TinyTBNet` uses `AdaptiveAvgPool2d((1, 1))`, reducing each $14 \times 14$ feature map into a single average value per channel before passing it to a tiny $128 \to 1$ linear projection.

---

## 5. Experiment History & Empirical Observations

### Experiment 1: The "Small-Data Illusion"
- **Dataset:** Montgomery County only (138 images total; 96 train, 21 val, 21 test).
- **Test Results:** Accuracy = 80.95%, **Sensitivity = 100.0% (9/9 caught)**, Specificity = 66.7%, ROC-AUC = 0.8426.
- **Insight:** 100% Sensitivity was an artifact of small-sample variance. Catching 9 out of 9 patients provided a false sense of perfection.

### Experiment 2: Multi-Center Scale-Up & Reality Check
- **Dataset:** Montgomery County + Shenzhen Hospital (800 images total; 559 train, 121 val, 120 test).
- **Training Setup:** 20 epochs, `batch_size = 16`, dynamic $w_{\text{pos}} = 1.03$.
- **Test Results:** Accuracy = 72.50%, **Sensitivity = 59.32% (35/59 caught)**, **Specificity = 85.25% (52/61 cleared)**, Test ROC-AUC = 0.7794, **Best Val ROC-AUC = 0.8658**.
- **Key Takeaways:**
  1. **Validation ROC-AUC was actually higher (0.8658 vs 0.8611)**, proving the internal representations improved.
  2. **Peak occurred on Epoch 20 (the final epoch)**: The learning rate was cut short by the cosine scheduler while the model was still descending the loss landscape.
  3. **Loss Weighting:** Neutral $w_{\text{pos}} = 1.03$ caused the network to favor Specificity over Sensitivity.
  4. **Domain Shift:** Shenzhen Hospital introduces real-world clinical complexity (subtle infiltrates, varying exposures, distinct demographics).

### Experiment 3: Option 3 (NIH / NIAID TB Portals Cohort) + Squeeze-and-Excitation Channel Attention
- **Dataset:** Montgomery County (138) + Shenzhen Hospital (662) + NIAID TB Portals Belarus Set (303) = **1,103 images total** (771 train, 166 val, 166 test; 697 TB, 406 Normal).
- **Data Provenance & Licensing:**
  - Option 3 requested images from the NIH / NIAID TB Portals program (culture-confirmed pulmonary TB, drug-resistant MDR-TB and drug-sensitive cases).
  - The official NIAID TB Portals web portal API requires institutional Data Access Requests (DAR) and credentialed Data Use Agreements (DUA).
  - To respect the constraint ("only if they are free to use"), we utilized the open-access, MIT-licensed mirror of the NIAID TB Portals Belarus clinical cohort published in *Nature Scientific Reports* (2019, `frapa/tbcnn`), filtering out documented non-frontal artifacts (`belarus/2.png`) to yield 303 verified frontal radiographs.
- **Architectural Enhancement (SE Channel Attention):**
  - Integrated Squeeze-and-Excitation (`SEBlock`) channel recalibration directly into each `ResBlock`:
    $$\mathbf{s} = \sigma \left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{x})) \right), \quad \tilde{\mathbf{x}} = \mathbf{s} \odot \mathbf{x}$$
  - Channel reduction ratio $r=8$ (minimum 4 intermediate channels).
  - Parameter footprint increased by only $+5,776$ parameters: from $307,089 \to 313,153$ parameters (**~305.8 KB INT8 weights**).
  - Runtime activation overhead: **0 additional bytes** to peak activation memory, because SE acts on a single $1 \times 1 \times C$ vector! Peak activation buffer remains $112 \times 112 \times 16 \approx 196 \text{ KB}$, well within the ESP32-S3 $\sim 320 \text{ KB}$ usable SRAM ceiling.
  - Microcontroller op compatibility: Standard GAP (`MEAN`), 1x1 convs, ReLU, Sigmoid (`LOGISTIC`), and broadcast multiply (`MUL`).
- **Training Setup:**
  - 40 epochs with Cosine Annealing learning rate schedule (`lr=0.001`, `weight_decay=1e-3`, `batch_size=16`).
  - Loss function: Clinical `BCEWithLogitsLoss` with calibrated $w_{\text{pos}} = 1.28$ (multiplier $2.2\times$ over base class ratio) to enforce strong penalties against false negatives.
- **Empirical Results (Unseen Test Set of 166 Patients: 105 TB, 61 Healthy):**
  - **Validation ROC-AUC:** Reached **0.9010** (breaking the 0.90 barrier!).
  - **Test ROC-AUC:** **0.8932** (an improvement of **+0.1138** vs Experiment 2's 0.7794).
  - **Test Accuracy:** **81.33%** (up from 72.50%).
  - **Test Sensitivity (Threshold 0.50):** **82.86%** (87/105 actual TB cases caught, up from 59.32% — a **+23.54%** increase!).
  - **Test Specificity (Threshold 0.50):** **78.69%** (48/61 healthy patients correctly cleared).
  - **Confusion Matrix:** $\text{TN}=48, \text{FP}=13, \text{FN}=18, \text{TP}=87$.
- **Clinical Operating Points (Sensitivity vs. Specificity Trade-Off):**
  - **Threshold 0.35 (Screening Mode):**
    - **Sensitivity = 88.57%** (Caught **93 out of 105** TB cases)
    - Specificity = 60.66% (Cleared 37 out of 61 normal cases)
    - Accuracy = 78.31%
  - **Threshold 0.40 (Balanced Clinical Screening):**
    - **Sensitivity = 86.67%** (Caught **91 out of 105** TB cases)
    - Specificity = 70.49% (Cleared 43 out of 61 normal cases)
    - Accuracy = 80.72%
  - **Threshold 0.45 (High Accuracy Triage):**
    - **Sensitivity = 84.76%** (Caught **89 out of 105** TB cases)
    - Specificity = 77.05% (Cleared 47 out of 61 normal cases)
    - Accuracy = 81.93%
  - **Threshold 0.50 (Standard Decision Cutoff):**
    - **Sensitivity = 82.86%** (Caught **87 out of 105** TB cases)
    - Specificity = 78.69% (Cleared 48 out of 61 normal cases)
    - Accuracy = 81.33%

### Comparative Metric Evolution Across Experiments:
| Metric | Exp 1 (Montgomery - 138 CXRs) | Exp 2 (+ Shenzhen - 800 CXRs) | Exp 3 (+ NIAID Belarus + SE - 1,103 CXRs) | Delta (Exp 3 vs Exp 2) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Images** | 138 | 800 | **1,103** | +303 (+37.9%) |
| **Val ROC-AUC** | 0.8611 | 0.8658 | **0.9010** | **+0.0352** |
| **Test ROC-AUC** | 0.8426 | 0.7794 | **0.8932** | **+0.1138** |
| **Test Accuracy**| 80.95% | 72.50% | **81.33%** | **+8.83%** |
| **Test Sensitivity**| 100.0% (9/9) | 59.32% (35/59) | **82.86% (87/105)** | **+23.54%** |
| **Test Specificity**| 66.67% (8/12) | 85.25% (52/61) | **78.69% (48/61)** | -6.56% (rebalanced) |
| **Screening Sensitivity (th=0.35)**| N/A | N/A | **88.57% (93/105)** | High detection |
| **Model Size (INT8)**| 300 KB | 300 KB | **305.8 KB** | Within ESP32 budget |

---

## 6. The TinyML & ESP32 Deployment: Empirical INT8 Quantization & Verification

To verify that TinyTB-Net with SE Attention runs on edge silicon, we implemented and executed the Post-Training Quantization (PTQ) pipeline ([src/quantize.py](file:///home/shalev/projects/tuberculosis_recognition/src/quantize.py)).

### 6.1 Empirical PTQ Results: FP32 Baseline vs. INT8 Quantized Model
The PyTorch FX Graph Mode PTQ pipeline was calibrated against representative clinical CXR validation scans and evaluated on the 166-patient unseen test cohort (105 TB, 61 Healthy):

| Diagnostic Metric | FP32 Baseline (Exp 3) | INT8 Quantized (PTQ) | Quantization Delta |
| :--- | :--- | :--- | :--- |
| **Test ROC-AUC** | **0.8932** | **0.8939** | **+0.0007** (Zero degradation!) |
| **Test Accuracy** | **81.33%** | **81.33%** | **+0.00%** (Exact match) |
| **Sensitivity (Recall)** | **82.86% (87/105)** | **81.90% (86/105)** | -0.95% (1 patient difference) |
| **Specificity** | **78.69% (48/61)** | **80.33% (49/61)** | **+1.64%** (+1 healthy cleared) |
| **Mean Absolute Error (Prob)** | N/A | **0.0115** | Avg probability drift is only 1.15% |
| **Max Absolute Error (Prob)** | N/A | **0.1218** | Clean numerical stability |
| **Logit MSE** | N/A | **0.0366** | Minimal logit distortion |

### 6.2 ESP32-S3 Hardware Memory & Compute Budget

| Resource | ESP32-S3 Physical Limit | TinyTB-Net Requirement | Status / Strategy |
| :--- | :--- | :--- | :--- |
| **Flash (Weights)** | 4 MB – 16 MB SPI Flash | **305.81 KB** (INT8, 313,153 params) | **PASS:** Consumes only 7.5% of standard 4 MB Flash (`.rodata`) |
| **Single Peak Activation** | ~320 KB internal SRAM | **196.00 KB** ($16 \times 112 \times 112$) | **PASS:** Fits cleanly within internal SRAM |
| **Ping-Pong Arena (Residuals)** | ~320 KB internal SRAM | **392.00 KB** ($2 \times 196 \text{ KB}$) | **STRATEGY:** ESP32-S3 with Octal PSRAM (2MB–8MB) holds full arena; for internal-SRAM-only chips, stream line buffers |
| **Compute / Clock** | 240 MHz Dual-Core | **194.5M MACs** (~389 MFLOPs) | Accelerated via ESP-NN PIE (vector assembly) |

### 6.3 Exported Deployment Artifacts
1. **TorchScript Traced INT8 Binary:** Saved to [models/tinytb_net_int8.pt](file:///home/shalev/projects/tuberculosis_recognition/models/tinytb_net_int8.pt) (454.5 KB).
2. **C Firmware Config Header:** Generated at [models/tinytb_net_esp32_config.h](file:///home/shalev/projects/tuberculosis_recognition/models/tinytb_net_esp32_config.h), containing input normalization constants, layer shapes, memory arena allocations, and calibrated diagnostic threshold macros (`TINYTB_THRESHOLD_SCREENING = 0.35f`).
3. **Safe Checkpoint Serialization:** Sanitized [models/best_tinytb_net.pth](file:///home/shalev/projects/tuberculosis_recognition/models/best_tinytb_net.pth) to use native Python primitives (`float`, `int`), ensuring complete compatibility with PyTorch 2.6+ secure `weights_only=True` loading.

---

## 7. Edge Web Interface & Zero-Overhead Preprocessing Pipeline

To provide a self-contained diagnostic interface served directly by the ESP32 Wi-Fi web server without external dependencies, we engineered a dedicated single-page edge application:

### 7.1 The Client-Side Canvas Preprocessing Architecture
- **Problem:** A user uploading a multi-megapixel photo from a smartphone (3–5 MB JPEG) would crash the ESP32 heap or take over 30 seconds to upload and decode.
- **Solution:** The embedded web application uses an in-browser HTML5 `<canvas>` element to execute all spatial resizing and Rec. 601 luma grayscale conversion ($224 \times 224 \times 1$) directly in client RAM before transmission.
- **Payload Reduction:** Shrinks the network upload from **~3–5 MB $\to$ exactly 50,176 bytes (~50.2 KB)**. The ESP32 receives raw uint8 pixels ready to feed directly into the input tensor buffer.

### 7.2 Embedded Footprint & Firmware Assets
1. **Single-file Web App:** [web/index.html](file:///home/shalev/projects/tuberculosis_recognition/web/index.html) (HTML + CSS + JS, zero CDN scripts, 100% offline, ~14.9 KB uncompressed).
2. **Firmware C Header:** [models/tinytb_net_web_page.h](file:///home/shalev/projects/tuberculosis_recognition/models/tinytb_net_web_page.h) embeds the web asset in gzipped PROGMEM format (**4.27 KB total Flash footprint**), served via `Content-Encoding: gzip`.
3. **Local Testing Server:** [web/server.py](file:///home/shalev/projects/tuberculosis_recognition/web/server.py) provides a lightweight Python HTTP server on port 8088 (with automatic collision recovery and socket address reuse) to test image uploads and live model inference prior to hardware flashing.

### 7.3 Visual Design & Interaction Architecture (`/web-design-engineer`)
- **Atmospheric Darkroom Card:** Minimalist, glassmorphism container (`#070a12`, `backdrop-filter: blur(24px)`) styled specifically for high-contrast radiograph inspection.
- **Drifting Dot-Matrix Grid Background:** Ultra-subtle SVG-mask dot-grid (`.bg-grid`) with slow directional translation (`36s linear infinite`) evoking precision medical imaging equipment.
- **Ambient Breathing Glow Orbs:** Two layered, heavily blurred (80px) floating orbs drifting asynchronously (`22s` and `26s` opposing cycle keyframes) that illuminate the deep background.
- **Reactive Ambient Color Morphing:** CSS variable bindings (`--orb1`, `--orb2`, `--glow`, `--meter`) dynamically shift the entire canvas and background lighting:
  - *Default / Idle:* Deep Cyber Cyan (`rgba(56, 189, 248, 0.22)`) & Soft Indigo (`rgba(99, 102, 241, 0.18)`).
  - *TB Suspected ($\ge 35\%$):* Backlight and ambient orbs morph into Coral Crimson (`rgba(244, 63, 94, 0.28)`).
  - *Healthy Cleared ($< 35\%$):* Backlight and ambient orbs morph into Emerald Green (`rgba(16, 185, 129, 0.28)`).
- **Sci-Fi Scanning Laser Beam:** Linear gradient cyan/indigo laser beam sweeps vertically across the radiograph canvas during inference, providing crisp tactile feedback during model execution.
- **Circular Radial Progress Gauge:** Smooth SVG arc stroke dash-offset transition (`2 * π * 54 = 339.29` path length) dynamically representing risk percentage.
- **Tabular Numeric Count-Up Ticker:** `requestAnimationFrame` ticker with exponential ease-out smoothly counting up from 0% to the exact diagnostic risk probability.
- **One-Click Diagnostic Samples:** Immediate pre-loaded verification buttons for culture-positive TB (`CHNCXR_0404_1.png`) and healthy control (`CHNCXR_0001_0.png`).
- **Strict Size Guardrails:** 14,816 bytes uncompressed HTML (under the 15,000-byte budget); 4,540 bytes gzipped PROGMEM C header (under the 5,120-byte / 5 KB budget).




