"""
Exports TinyTB-Net quantized INT8 model parameters and test CXRs to C header files
for ESP32 microcontroller execution.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import TinyTBNet
from src.dataset import create_dataloaders
from src.quantize import quantize_tinytb_net, BEST_MODEL_PATH

FIRMWARE_SRC = os.path.join(PROJECT_ROOT, "firmware", "src")
WEIGHTS_HEADER = os.path.join(FIRMWARE_SRC, "tinytb_weights.h")
SAMPLES_HEADER = os.path.join(FIRMWARE_SRC, "tinytb_samples.h")

def export_model_to_c():
    print("Loading model and performing deterministic PTQ INT8 quantization...")
    cp = torch.load(BEST_MODEL_PATH, map_location="cpu")
    fp32_model = TinyTBNet(use_se=cp.get("use_se", True))
    fp32_model.load_state_dict(cp["model_state_dict"])
    fp32_model.eval()

    loaders = create_dataloaders(batch_size=16)
    q_model = quantize_tinytb_net(fp32_model, loaders["val_loader"], num_batches=15)
    modules = dict(q_model.named_modules())

    print("Extracting weights, biases, scales, and zero points...")
    
    # Input quantization parameters
    in_scale = getattr(q_model, "stem_0_block_0_input_scale_0").item()
    in_zp = getattr(q_model, "stem_0_block_0_input_zero_point_0").item()

    # Intermediate activation scales and zero points
    stage1_se_scale = getattr(q_model, "stage1_se_scale_0").item()
    stage1_se_zp = getattr(q_model, "stage1_se_zero_point_0").item()
    stage1_relu_scale = getattr(q_model, "stage1_relu_scale_0").item()
    stage1_relu_zp = getattr(q_model, "stage1_relu_zero_point_0").item()

    stage2_se_scale = getattr(q_model, "stage2_se_scale_0").item()
    stage2_se_zp = getattr(q_model, "stage2_se_zero_point_0").item()
    stage2_relu_scale = getattr(q_model, "stage2_relu_scale_0").item()
    stage2_relu_zp = getattr(q_model, "stage2_relu_zero_point_0").item()

    stage3_se_scale = getattr(q_model, "stage3_se_scale_0").item()
    stage3_se_zp = getattr(q_model, "stage3_se_zero_point_0").item()
    stage3_relu_scale = getattr(q_model, "stage3_relu_scale_0").item()
    stage3_relu_zp = getattr(q_model, "stage3_relu_zero_point_0").item()

    stage4_se_scale = getattr(q_model, "stage4_se_scale_0").item()
    stage4_se_zp = getattr(q_model, "stage4_se_zero_point_0").item()
    stage4_relu_scale = getattr(q_model, "stage4_relu_scale_0").item()
    stage4_relu_zp = getattr(q_model, "stage4_relu_zero_point_0").item()

    # Function to format arrays
    def fmt_i8(arr, name):
        flat = arr.flatten().astype(np.int8)
        lines = [f"// {name} shape: {list(arr.shape)}, size: {len(flat)} bytes"]
        lines.append(f"const int8_t {name}[{len(flat)}] PROGMEM = {{")
        chunk_size = 16
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i+chunk_size]
            lines.append("    " + ", ".join(f"{int(x)}" for x in chunk) + ",")
        lines.append("};\n")
        return "\n".join(lines)

    def fmt_f32(arr, name):
        flat = arr.flatten().astype(np.float32)
        lines = [f"// {name} shape: {list(arr.shape)}, size: {len(flat)} floats"]
        lines.append(f"const float {name}[{len(flat)}] PROGMEM = {{")
        chunk_size = 8
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i+chunk_size]
            lines.append("    " + ", ".join(f"{float(x):.8e}f" for x in chunk) + ",")
        lines.append("};\n")
        return "\n".join(lines)

    c_out = []
    c_out.append("/**")
    c_out.append(" * Auto-generated TinyTB-Net INT8 Quantized Model Parameters")
    c_out.append(" * Target: ESP32 Microcontroller (Flash PROGMEM .rodata)")
    c_out.append(" */")
    c_out.append("#ifndef TINYTB_WEIGHTS_H")
    c_out.append("#define TINYTB_WEIGHTS_H")
    c_out.append("#include <stdint.h>")
    c_out.append("#include <pgmspace.h>\n")

    c_out.append(f"#define INPUT_SCALE        {in_scale:.8e}f")
    c_out.append(f"#define INPUT_ZERO_POINT    {in_zp}\n")

    # Layer extraction helper
    def extract_conv(mod, prefix):
        w = mod.weight()
        w_int = w.int_repr().numpy().astype(np.int8)
        w_scale = w.q_per_channel_scales().numpy().astype(np.float32)
        b = mod.bias().detach().numpy().astype(np.float32)
        s_out = float(getattr(mod, "scale", 1.0))
        zp_out = int(getattr(mod, "zero_point", 0))

        sec = []
        sec.append(f"// Layer: {prefix}")
        sec.append(f"#define {prefix.upper()}_OUT_SCALE       {s_out:.8e}f")
        sec.append(f"#define {prefix.upper()}_OUT_ZERO_POINT {zp_out}")
        sec.append(fmt_i8(w_int, f"{prefix}_weight"))
        sec.append(fmt_f32(w_scale, f"{prefix}_w_scale"))
        sec.append(fmt_f32(b, f"{prefix}_bias"))
        return "\n".join(sec)

    # 1. Stem
    c_out.append(extract_conv(modules["stem.0.block.0"], "stem_conv"))

    # 2. Stage 1
    c_out.append(extract_conv(modules["stage1.conv1"], "s1_c1"))
    c_out.append(extract_conv(modules["stage1.conv2"], "s1_c2"))
    c_out.append(extract_conv(modules["stage1.se.fc.1"], "s1_se1"))
    c_out.append(extract_conv(modules["stage1.se.fc.3"], "s1_se2"))
    c_out.append(f"#define S1_SE_SCALE       {stage1_se_scale:.8e}f")
    c_out.append(f"#define S1_SE_ZERO_POINT  {stage1_se_zp}")
    c_out.append(f"#define S1_RELU_SCALE      {stage1_relu_scale:.8e}f")
    c_out.append(f"#define S1_RELU_ZERO_POINT {stage1_relu_zp}\n")

    # 3. Stage 2
    c_out.append(extract_conv(modules["stage2.shortcut.0"], "s2_sc"))
    c_out.append(extract_conv(modules["stage2.conv1"], "s2_c1"))
    c_out.append(extract_conv(modules["stage2.conv2"], "s2_c2"))
    c_out.append(extract_conv(modules["stage2.se.fc.1"], "s2_se1"))
    c_out.append(extract_conv(modules["stage2.se.fc.3"], "s2_se2"))
    c_out.append(f"#define S2_SE_SCALE       {stage2_se_scale:.8e}f")
    c_out.append(f"#define S2_SE_ZERO_POINT  {stage2_se_zp}")
    c_out.append(f"#define S2_RELU_SCALE      {stage2_relu_scale:.8e}f")
    c_out.append(f"#define S2_RELU_ZERO_POINT {stage2_relu_zp}\n")

    # 4. Stage 3
    c_out.append(extract_conv(modules["stage3.shortcut.0"], "s3_sc"))
    c_out.append(extract_conv(modules["stage3.conv1"], "s3_c1"))
    c_out.append(extract_conv(modules["stage3.conv2"], "s3_c2"))
    c_out.append(extract_conv(modules["stage3.se.fc.1"], "s3_se1"))
    c_out.append(extract_conv(modules["stage3.se.fc.3"], "s3_se2"))
    c_out.append(f"#define S3_SE_SCALE       {stage3_se_scale:.8e}f")
    c_out.append(f"#define S3_SE_ZERO_POINT  {stage3_se_zp}")
    c_out.append(f"#define S3_RELU_SCALE      {stage3_relu_scale:.8e}f")
    c_out.append(f"#define S3_RELU_ZERO_POINT {stage3_relu_zp}\n")

    # 5. Stage 4
    c_out.append(extract_conv(modules["stage4.shortcut.0"], "s4_sc"))
    c_out.append(extract_conv(modules["stage4.conv1"], "s4_c1"))
    c_out.append(extract_conv(modules["stage4.conv2"], "s4_c2"))
    c_out.append(extract_conv(modules["stage4.se.fc.1"], "s4_se1"))
    c_out.append(extract_conv(modules["stage4.se.fc.3"], "s4_se2"))
    c_out.append(f"#define S4_SE_SCALE       {stage4_se_scale:.8e}f")
    c_out.append(f"#define S4_SE_ZERO_POINT  {stage4_se_zp}")
    c_out.append(f"#define S4_RELU_SCALE      {stage4_relu_scale:.8e}f")
    c_out.append(f"#define S4_RELU_ZERO_POINT {stage4_relu_zp}\n")

    # 6. Classifier Head (Linear)
    fc = modules["classifier.1"]
    fc_w = fc.weight()
    fc_w_int = fc_w.int_repr().numpy().astype(np.int8)
    fc_w_scale = fc_w.q_per_channel_scales().numpy().astype(np.float32)
    fc_b = fc.bias().detach().numpy().astype(np.float32)
    c_out.append(fmt_i8(fc_w_int, "fc_weight"))
    c_out.append(fmt_f32(fc_w_scale, "fc_w_scale"))
    c_out.append(fmt_f32(fc_b, "fc_bias"))
    fc_out_scale = float(getattr(fc, "scale", 1.0))
    fc_out_zp = int(getattr(fc, "zero_point", 0))
    c_out.append(f"#define FC_OUT_SCALE       {fc_out_scale:.8e}f")
    c_out.append(f"#define FC_OUT_ZERO_POINT  {fc_out_zp}\n")

    c_out.append("#endif // TINYTB_WEIGHTS_H\n")

    os.makedirs(FIRMWARE_SRC, exist_ok=True)
    with open(WEIGHTS_HEADER, "w") as f:
        f.write("\n".join(c_out))
    print(f"Exported model weights to {WEIGHTS_HEADER} ({os.path.getsize(WEIGHTS_HEADER)/1024:.1f} KB)")

    # Export test sample radiographs
    print("Exporting test sample CXRs for on-device benchmark...")
    # Load 1 TB CXR and 1 Normal CXR
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tb_img_path = os.path.join(PROJECT_ROOT, "data", "images", "CHNCXR_0404_1.png")
    norm_img_path = os.path.join(PROJECT_ROOT, "data", "images", "CHNCXR_0001_0.png")

    img_tb = Image.open(tb_img_path).convert("L")
    img_norm = Image.open(norm_img_path).convert("L")

    t_tb = val_tf(img_tb).unsqueeze(0)
    t_norm = val_tf(img_norm).unsqueeze(0)

    with torch.no_grad():
        pt_tb_logit = q_model(t_tb).item()
        pt_tb_prob = 1.0 / (1.0 + np.exp(-pt_tb_logit))
        pt_norm_logit = q_model(t_norm).item()
        pt_norm_prob = 1.0 / (1.0 + np.exp(-pt_norm_logit))

    print(f"  TB Sample (CHNCXR_0404_1.png): Reference Logit={pt_tb_logit:.4f}, Prob={pt_tb_prob:.4f}")
    print(f"  Normal Sample (CHNCXR_0001_0.png): Reference Logit={pt_norm_logit:.4f}, Prob={pt_norm_prob:.4f}")

    # Quantize input images to uint8
    q_tb = torch.quantize_per_tensor(t_tb, in_scale, in_zp, torch.quint8).int_repr().squeeze().numpy()
    q_norm = torch.quantize_per_tensor(t_norm, in_scale, in_zp, torch.quint8).int_repr().squeeze().numpy()

    s_out = []
    s_out.append("/**")
    s_out.append(" * Pre-quantized Test CXR Images for Standalone Hardware Verification")
    s_out.append(" */")
    s_out.append("#ifndef TINYTB_SAMPLES_H")
    s_out.append("#define TINYTB_SAMPLES_H")
    s_out.append("#include <stdint.h>")
    s_out.append("#include <pgmspace.h>\n")
    s_out.append(f"#define REF_TB_LOGIT       {pt_tb_logit:.6f}f")
    s_out.append(f"#define REF_TB_PROB        {pt_tb_prob:.6f}f")
    s_out.append(f"#define REF_NORM_LOGIT     {pt_norm_logit:.6f}f")
    s_out.append(f"#define REF_NORM_PROB      {pt_norm_prob:.6f}f\n")

    def fmt_u8(arr, name):
        flat = arr.flatten().astype(np.uint8)
        lines = [f"// {name} size: {len(flat)} bytes"]
        lines.append(f"const uint8_t {name}[{len(flat)}] PROGMEM = {{")
        chunk_size = 16
        for i in range(0, len(flat), chunk_size):
            chunk = flat[i:i+chunk_size]
            lines.append("    " + ", ".join(f"{int(x)}" for x in chunk) + ",")
        lines.append("};\n")
        return "\n".join(lines)

    s_out.append(fmt_u8(q_tb, "sample_tb_img"))
    s_out.append(fmt_u8(q_norm, "sample_norm_img"))
    s_out.append("#endif // TINYTB_SAMPLES_H\n")

    with open(SAMPLES_HEADER, "w") as f:
        f.write("\n".join(s_out))
    print(f"Exported sample CXRs to {SAMPLES_HEADER} ({os.path.getsize(SAMPLES_HEADER)/1024:.1f} KB)")
    print("Done!")

if __name__ == "__main__":
    export_model_to_c()
