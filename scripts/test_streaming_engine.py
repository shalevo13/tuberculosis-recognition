"""
Bit-exact streaming line-buffered TinyTB-Net INT8 Inference Engine Simulator
Verifies mathematical equivalence to PyTorch FX INT8 model.
"""

import os, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import TinyTBNet
from src.dataset import create_dataloaders
from src.quantize import quantize_tinytb_net, BEST_MODEL_PATH

cp = torch.load(BEST_MODEL_PATH, map_location="cpu")
fp32_model = TinyTBNet(use_se=cp.get("use_se", True))
fp32_model.load_state_dict(cp["model_state_dict"])
fp32_model.eval()

loaders = create_dataloaders(batch_size=16)
q_model = quantize_tinytb_net(fp32_model, loaders["val_loader"], num_batches=15)
modules = dict(q_model.named_modules())

def run_test(img_name):
    img_path = os.path.join("data", "images", img_name)
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(img_path).convert("L")
    t_img = val_tf(img).unsqueeze(0)

    # Reference Torch output
    with torch.no_grad():
        ref_logit = q_model(t_img).item()
        ref_prob = 1.0 / (1.0 + np.exp(-ref_logit))

    print(f"\n--- Testing {img_name} ---")
    print(f"PyTorch Reference: Logit = {ref_logit:.6f}, Prob = {ref_prob:.6f}")

    in_scale = getattr(q_model, "stem_0_block_0_input_scale_0").item()
    in_zp = getattr(q_model, "stem_0_block_0_input_zero_point_0").item()
    q_input = torch.quantize_per_tensor(t_img, in_scale, in_zp, torch.quint8).int_repr().squeeze().numpy().astype(np.uint8)

    # Helper for quantized conv 1x1
    def conv1x1_pixel(in_vec, in_ch, weights, w_scale, bias, in_sc, in_z, out_sc, out_z, relu=False):
        # in_vec: [in_ch], weights: [out_ch, in_ch]
        out_ch = weights.shape[0]
        out_vec = np.zeros(out_ch, dtype=np.uint8)
        for oc in range(out_ch):
            acc = sum((int(in_vec[ic]) - in_z) * int(weights[oc, ic]) for ic in range(in_ch))
            r = acc * (in_sc * float(w_scale[oc])) + float(bias[oc])
            if relu and r < 0.0: r = 0.0
            q = int(np.round(r / out_sc)) + out_z
            out_vec[oc] = np.clip(q, 0, 255)
        return out_vec

    # 1. STEM
    stem_w = modules["stem.0.block.0"].weight().int_repr().numpy() # [16, 1, 3, 3]
    stem_w_sc = modules["stem.0.block.0"].weight().q_per_channel_scales().numpy()
    stem_b = modules["stem.0.block.0"].bias().detach().numpy()
    stem_s = modules["stem.0.block.0"].scale
    stem_zp = modules["stem.0.block.0"].zero_point

    def get_stem_row(r):
        # r in [0, 111], produces [16, 112]
        row_out = np.zeros((16, 112), dtype=np.uint8)
        ih_base = r * 2 - 1
        for oc in range(16):
            w_sc = float(stem_w_sc[oc])
            b = float(stem_b[oc])
            w_oc = stem_w[oc, 0] # [3, 3]
            for c in range(112):
                iw_base = c * 2 - 1
                acc = 0
                for kh in range(3):
                    ih = ih_base + kh
                    if 0 <= ih < 224:
                        for kw in range(3):
                            iw = iw_base + kw
                            if 0 <= iw < 224:
                                acc += (int(q_input[ih, iw]) - in_zp) * int(w_oc[kh, kw])
                r_val = acc * (in_scale * w_sc) + b
                if r_val < 0.0: r_val = 0.0
                q = int(np.round(r_val / stem_s)) + stem_zp
                row_out[oc, c] = np.clip(q, 0, 255)
        return row_out

    # 2. STAGE 1
    # conv1: 16 -> 16, 3x3, stride 1, relu
    s1_c1_w = modules["stage1.conv1"].weight().int_repr().numpy() # [16, 16, 3, 3]
    s1_c1_sc = modules["stage1.conv1"].weight().q_per_channel_scales().numpy()
    s1_c1_b = modules["stage1.conv1"].bias().detach().numpy()
    s1_c1_s = modules["stage1.conv1"].scale
    s1_c1_zp = modules["stage1.conv1"].zero_point

    # conv2: 16 -> 16, 3x3, stride 1, no relu
    s1_c2_w = modules["stage1.conv2"].weight().int_repr().numpy() # [16, 16, 3, 3]
    s1_c2_sc = modules["stage1.conv2"].weight().q_per_channel_scales().numpy()
    s1_c2_b = modules["stage1.conv2"].bias().detach().numpy()
    s1_c2_s = modules["stage1.conv2"].scale
    s1_c2_zp = modules["stage1.conv2"].zero_point

    # Compute a 3x3 conv row from 3 input rows: [3, in_ch, width]
    def conv3x3_row(in_3rows, in_ch, width, out_ch, weights, w_scale, bias, in_sc, in_z, out_sc, out_z, relu=False):
        row_out = np.zeros((out_ch, width), dtype=np.uint8)
        for oc in range(out_ch):
            w_sc = float(w_scale[oc])
            b = float(bias[oc])
            w_oc = weights[oc] # [in_ch, 3, 3]
            for c in range(width):
                iw_base = c - 1
                acc = 0
                for ic in range(in_ch):
                    w_k = w_oc[ic]
                    for kh in range(3):
                        in_p = in_3rows[kh][ic] if in_3rows[kh] is not None else None
                        if in_p is not None:
                            for kw in range(3):
                                iw = iw_base + kw
                                if 0 <= iw < width:
                                    acc += (int(in_p[iw]) - in_z) * int(w_k[kh, kw])
                r_val = acc * (in_sc * w_sc) + b
                if relu and r_val < 0.0: r_val = 0.0
                q = int(np.round(r_val / out_sc)) + out_z
                row_out[oc, c] = np.clip(q, 0, 255)
        return row_out

    # PASS 1: Accumulate Stage 1 SE GAP
    print("Simulating Stage 1 Pass 1...")
    s1_c2_sums = np.zeros(16, dtype=np.int64)

    # Line buffers
    # To compute c2 row r: needs c1 rows r-1, r, r+1
    # To compute c1 row k: needs stem rows k-1, k, k+1
    c1_buffer = [None, None, None] # slots 0, 1, 2
    stem_buffer = [None, None, None]

    # Pre-load stem rows 0 and 1
    stem_buffer[1] = get_stem_row(0)
    stem_buffer[2] = get_stem_row(1)

    for r in range(112):
        # We need c1 row r+1 to compute c2 row r
        # Shift c1 buffer
        c1_buffer[0] = c1_buffer[1]
        c1_buffer[1] = c1_buffer[2]
        
        # Compute c1 row r+1 (if r+1 < 112)
        if r + 1 < 112:
            stem_buffer[0] = stem_buffer[1]
            stem_buffer[1] = stem_buffer[2]
            stem_buffer[2] = get_stem_row(r + 2) if (r + 2 < 112) else None
            c1_in = np.array([stem_buffer[0], stem_buffer[1], stem_buffer[2]], dtype=object)
            c1_buffer[2] = conv3x3_row(c1_in, 16, 112, 16, s1_c1_w, s1_c1_sc, s1_c1_b, stem_s, stem_zp, s1_c1_s, s1_c1_zp, True)
        else:
            c1_buffer[2] = None

        if r == 0:
            # Also compute c1 row 0
            c1_in_0 = np.array([None, stem_buffer[0], stem_buffer[1]], dtype=object) # row -1 is None
            c1_buffer[1] = conv3x3_row(c1_in_0, 16, 112, 16, s1_c1_w, s1_c1_sc, s1_c1_b, stem_s, stem_zp, s1_c1_s, s1_c1_zp, True)
            c1_buffer[0] = None

        # Compute c2 row r
        c2_in = np.array([c1_buffer[0], c1_buffer[1], c1_buffer[2]], dtype=object)
        c2_r = conv3x3_row(c2_in, 16, 112, 16, s1_c2_w, s1_c2_sc, s1_c2_b, s1_c1_s, s1_c1_zp, s1_c2_s, s1_c2_zp, False)
        for oc in range(16):
            s1_c2_sums[oc] += np.sum(c2_r[oc])

    # Compute SE sigmoid
    s1_se_gap = np.round(s1_c2_sums / (112.0 * 112.0)).astype(np.uint8)
    s1_se1_w = modules["stage1.se.fc.1"].weight().int_repr().numpy().squeeze() # [4, 16]
    s1_se1_sc = modules["stage1.se.fc.1"].weight().q_per_channel_scales().numpy()
    s1_se1_b = modules["stage1.se.fc.1"].bias().detach().numpy()
    s1_se1_s = modules["stage1.se.fc.1"].scale
    s1_se1_zp = modules["stage1.se.fc.1"].zero_point

    s1_se1_out = np.zeros(4, dtype=np.uint8)
    for oc in range(4):
        acc = sum((int(s1_se_gap[ic]) - s1_c2_zp) * int(s1_se1_w[oc, ic]) for ic in range(16))
        r_val = acc * (s1_c2_s * float(s1_se1_sc[oc])) + float(s1_se1_b[oc])
        if r_val < 0.0: r_val = 0.0
        q = int(np.round(r_val / s1_se1_s)) + s1_se1_zp
        s1_se1_out[oc] = np.clip(q, 0, 255)

    s1_se2_w = modules["stage1.se.fc.3"].weight().int_repr().numpy().squeeze() # [16, 4]
    s1_se2_sc = modules["stage1.se.fc.3"].weight().q_per_channel_scales().numpy()
    s1_se2_b = modules["stage1.se.fc.3"].bias().detach().numpy()
    s1_se2_s = modules["stage1.se.fc.3"].scale
    s1_se2_zp = modules["stage1.se.fc.3"].zero_point

    s1_sig = np.zeros(16, dtype=np.float32)
    for oc in range(16):
        acc = sum((int(s1_se1_out[ic]) - s1_se1_zp) * int(s1_se2_w[oc, ic]) for ic in range(4))
        r_val = acc * (s1_se1_s * float(s1_se2_sc[oc])) + float(s1_se2_b[oc])
        q = np.clip(int(np.round(r_val / s1_se2_s)) + s1_se2_zp, 0, 255)
        r_deq = (q - s1_se2_zp) * s1_se2_s
        # Sigmoid: quantize with scale=1/256, zp=0
        sig_f = 1.0 / (1.0 + np.exp(-r_deq))
        q_sig = np.clip(int(np.round(sig_f * 256.0)), 0, 255)
        s1_sig[oc] = q_sig / 256.0

    print("Stage 1 SE computed successfully!")

    # Check Stage 1 SE vs reference
    with torch.no_grad():
        in_q = torch.quantize_per_tensor(t_img, in_scale, in_zp, torch.quint8)
        pt_stem = modules["stem.0.block.0"](in_q)
        pt_c1 = modules["stage1.conv1"](pt_stem)
        pt_c2 = modules["stage1.conv2"](pt_c1)
        pt_pool = modules["stage1.se.fc.0"](pt_c2)
        pt_se1 = modules["stage1.se.fc.1"](pt_pool)
        pt_se2 = modules["stage1.se.fc.3"](pt_se1)
        pt_sig = modules["stage1.se.fc.4"](pt_se2)
    print("Diff in Stage 1 SE pool:", np.max(np.abs(pt_pool.int_repr().squeeze().numpy() - s1_se_gap)))
    print("Diff in Stage 1 SE fc1:", np.max(np.abs(pt_se1.int_repr().squeeze().numpy() - s1_se1_out)))
    print("Diff in Stage 1 SE sig:", np.max(np.abs(pt_sig.int_repr().squeeze().numpy() - np.round(s1_sig * 256).astype(int))))

run_test("CHNCXR_0404_1.png")
