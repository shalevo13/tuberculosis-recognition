"""
Full Pipeline Verification Simulator for TinyTB-Net INT8
Simulates the exact memory-efficient streaming inference engine in Python
and compares layer-by-layer and end-to-end against PyTorch FX INT8 model.
"""

import os, sys
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

cp = torch.load(BEST_MODEL_PATH, map_location="cpu")
fp32_model = TinyTBNet(use_se=cp.get("use_se", True))
fp32_model.load_state_dict(cp["model_state_dict"])
fp32_model.eval()

loaders = create_dataloaders(batch_size=16)
q_model = quantize_tinytb_net(fp32_model, loaders["val_loader"], num_batches=15)
modules = dict(q_model.named_modules())

def test_image(img_name):
    img_path = os.path.join(PROJECT_ROOT, "data", "images", img_name)
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(img_path).convert("L")
    t_img = val_tf(img).unsqueeze(0)

    with torch.no_grad():
        ref_logit = q_model(t_img).item()
        ref_prob = 1.0 / (1.0 + np.exp(-ref_logit))

    print(f"\n=======================================================")
    print(f"Testing {img_name}")
    print(f"PyTorch Reference: Logit = {ref_logit:.6f}, Prob = {ref_prob:.6f}")
    print(f"=======================================================")

    in_scale = getattr(q_model, "stem_0_block_0_input_scale_0").item()
    in_zp = getattr(q_model, "stem_0_block_0_input_zero_point_0").item()
    q_input = torch.quantize_per_tensor(t_img, in_scale, in_zp, torch.quint8).int_repr().squeeze().numpy().astype(np.uint8)

    # Fast 2D conv for verification
    def conv2d(in_arr, weights, w_scale, bias, in_sc, in_z, out_sc, out_z, stride=1, padding=1, relu=False):
        # in_arr: [in_c, in_h, in_w]
        # weights: [out_c, in_c, kh, kw]
        out_c, in_c, kh, kw = weights.shape
        in_c_chk, in_h, in_w = in_arr.shape
        out_h = (in_h + 2 * padding - kh) // stride + 1
        out_w = (in_w + 2 * padding - kw) // stride + 1
        
        # pad input
        padded = np.pad(in_arr.astype(np.int32) - in_z, ((0,0), (padding, padding), (padding, padding)), constant_values=0)
        out = np.zeros((out_c, out_h, out_w), dtype=np.uint8)
        
        for oc in range(out_c):
            w_oc = weights[oc].astype(np.int32)
            w_sc = float(w_scale[oc])
            b = float(bias[oc])
            
            # correlate
            acc = np.zeros((out_h, out_w), dtype=np.int32)
            for ic in range(in_c):
                p_ic = padded[ic]
                w_ic = w_oc[ic]
                for h in range(kh):
                    for w in range(kw):
                        acc += p_ic[h:h + out_h * stride:stride, w:w + out_w * stride:stride] * w_ic[h, w]
            
            r = acc * (in_sc * w_sc) + b
            if relu:
                r = np.maximum(r, 0.0)
            q = np.round(r / out_sc) + out_z
            out[oc] = np.clip(q, 0, 255).astype(np.uint8)
        return out

    def se_block(in_arr, in_sc, in_zp, fc1_w, fc1_sc, fc1_b, fc1_sc_out, fc1_zp_out,
                 fc2_w, fc2_sc, fc2_b, fc2_sc_out, fc2_zp_out):
        # in_arr: [C, H, W]
        C, H, W = in_arr.shape
        gap = np.round(np.mean(in_arr.astype(np.float32), axis=(1,2))).astype(np.uint8)
        
        # fc1: C -> C//4
        mid_c = fc1_w.shape[0]
        fc1_out = np.zeros(mid_c, dtype=np.uint8)
        for oc in range(mid_c):
            acc = int(np.sum((gap.astype(np.int32) - in_zp) * fc1_w[oc].astype(np.int32)))
            r = acc * (in_sc * float(fc1_sc[oc])) + float(fc1_b[oc])
            if r < 0.0: r = 0.0
            q = int(np.round(r / fc1_sc_out)) + fc1_zp_out
            fc1_out[oc] = np.clip(q, 0, 255)

        # fc2: C//4 -> C
        sig_scale = 1.0 / 256.0
        sig_out = np.zeros(C, dtype=np.float32)
        for oc in range(C):
            acc = int(np.sum((fc1_out.astype(np.int32) - fc1_zp_out) * fc2_w[oc].astype(np.int32)))
            r = acc * (fc1_sc_out * float(fc2_sc[oc])) + float(fc2_b[oc])
            q = np.clip(int(np.round(r / fc2_sc_out)) + fc2_zp_out, 0, 255)
            r_deq = (q - fc2_zp_out) * fc2_sc_out
            sig_f = 1.0 / (1.0 + np.exp(-r_deq))
            q_sig = np.clip(int(np.round(sig_f * 256.0)), 0, 255)
            sig_out[oc] = q_sig / 256.0
        return sig_out

    def add_relu_mul(c2, sig, sc, c2_sc, c2_zp, sc_sc, sc_zp, se_sc, se_zp, relu_sc, relu_zp):
        # Mul: c2 * sig
        C, H, W = c2.shape
        r_c2 = (c2.astype(np.float32) - c2_zp) * c2_sc
        r_mul = r_c2 * sig[:, None, None]
        q_mul = np.clip(np.round(r_mul / se_sc) + se_zp, 0, 255).astype(np.uint8)
        
        # AddReLU: q_mul + sc
        r_se = (q_mul.astype(np.float32) - se_zp) * se_sc
        r_sc = (sc.astype(np.float32) - sc_zp) * sc_sc
        r_sum = r_se + r_sc
        r_sum = np.maximum(r_sum, 0.0)
        out = np.clip(np.round(r_sum / relu_sc) + relu_zp, 0, 255).astype(np.uint8)
        return out

    # 1. Stem
    print("Computing Stem...")
    stem_w = modules["stem.0.block.0"].weight().int_repr().numpy()
    stem_sc = modules["stem.0.block.0"].weight().q_per_channel_scales().numpy()
    stem_b = modules["stem.0.block.0"].bias().detach().numpy()
    stem_out = conv2d(q_input[None, :, :], stem_w, stem_sc, stem_b, in_scale, in_zp,
                      modules["stem.0.block.0"].scale, modules["stem.0.block.0"].zero_point,
                      stride=2, padding=1, relu=True)

    with torch.no_grad():
        pt_in = torch.quantize_per_tensor(t_img, in_scale, in_zp, torch.quint8)
        pt_stem = modules["stem.0.block.0"](pt_in).int_repr().squeeze().numpy()
    print("  Stem max diff vs PyTorch:", np.max(np.abs(stem_out - pt_stem)))

    # 2. Stage 1
    print("Computing Stage 1...")
    s1_c1_w = modules["stage1.conv1"].weight().int_repr().numpy()
    s1_c1_sc = modules["stage1.conv1"].weight().q_per_channel_scales().numpy()
    s1_c1_b = modules["stage1.conv1"].bias().detach().numpy()
    s1_c1 = conv2d(stem_out, s1_c1_w, s1_c1_sc, s1_c1_b, modules["stem.0.block.0"].scale, modules["stem.0.block.0"].zero_point,
                   modules["stage1.conv1"].scale, modules["stage1.conv1"].zero_point, stride=1, padding=1, relu=True)

    s1_c2_w = modules["stage1.conv2"].weight().int_repr().numpy()
    s1_c2_sc = modules["stage1.conv2"].weight().q_per_channel_scales().numpy()
    s1_c2_b = modules["stage1.conv2"].bias().detach().numpy()
    s1_c2 = conv2d(s1_c1, s1_c2_w, s1_c2_sc, s1_c2_b, modules["stage1.conv1"].scale, modules["stage1.conv1"].zero_point,
                   modules["stage1.conv2"].scale, modules["stage1.conv2"].zero_point, stride=1, padding=1, relu=False)

    s1_se1_w = modules["stage1.se.fc.1"].weight().int_repr().numpy().squeeze()
    s1_se1_sc = modules["stage1.se.fc.1"].weight().q_per_channel_scales().numpy()
    s1_se1_b = modules["stage1.se.fc.1"].bias().detach().numpy()
    s1_se2_w = modules["stage1.se.fc.3"].weight().int_repr().numpy().squeeze()
    s1_se2_sc = modules["stage1.se.fc.3"].weight().q_per_channel_scales().numpy()
    s1_se2_b = modules["stage1.se.fc.3"].bias().detach().numpy()

    s1_sig = se_block(s1_c2, modules["stage1.conv2"].scale, modules["stage1.conv2"].zero_point,
                      s1_se1_w, s1_se1_sc, s1_se1_b, modules["stage1.se.fc.1"].scale, modules["stage1.se.fc.1"].zero_point,
                      s1_se2_w, s1_se2_sc, s1_se2_b, modules["stage1.se.fc.3"].scale, modules["stage1.se.fc.3"].zero_point)

    s1_out = add_relu_mul(s1_c2, s1_sig, stem_out,
                          modules["stage1.conv2"].scale, modules["stage1.conv2"].zero_point,
                          modules["stem.0.block.0"].scale, modules["stem.0.block.0"].zero_point,
                          getattr(q_model, "stage1_se_scale_0").item(), getattr(q_model, "stage1_se_zero_point_0").item(),
                          getattr(q_model, "stage1_relu_scale_0").item(), getattr(q_model, "stage1_relu_zero_point_0").item())

    # Stage 1 complete
    # 3. Stage 2
    print("Computing Stage 2...")
    s2_sc_w = modules["stage2.shortcut.0"].weight().int_repr().numpy()
    s2_sc_sc = modules["stage2.shortcut.0"].weight().q_per_channel_scales().numpy()
    s2_sc_b = modules["stage2.shortcut.0"].bias().detach().numpy()
    s2_sc = conv2d(s1_out, s2_sc_w, s2_sc_sc, s2_sc_b, getattr(q_model, "stage1_relu_scale_0").item(), getattr(q_model, "stage1_relu_zero_point_0").item(),
                   modules["stage2.shortcut.0"].scale, modules["stage2.shortcut.0"].zero_point, stride=2, padding=0, relu=False)

    s2_c1_w = modules["stage2.conv1"].weight().int_repr().numpy()
    s2_c1_sc = modules["stage2.conv1"].weight().q_per_channel_scales().numpy()
    s2_c1_b = modules["stage2.conv1"].bias().detach().numpy()
    s2_c1 = conv2d(s1_out, s2_c1_w, s2_c1_sc, s2_c1_b, getattr(q_model, "stage1_relu_scale_0").item(), getattr(q_model, "stage1_relu_zero_point_0").item(),
                   modules["stage2.conv1"].scale, modules["stage2.conv1"].zero_point, stride=2, padding=1, relu=True)

    s2_c2_w = modules["stage2.conv2"].weight().int_repr().numpy()
    s2_c2_sc = modules["stage2.conv2"].weight().q_per_channel_scales().numpy()
    s2_c2_b = modules["stage2.conv2"].bias().detach().numpy()
    s2_c2 = conv2d(s2_c1, s2_c2_w, s2_c2_sc, s2_c2_b, modules["stage2.conv1"].scale, modules["stage2.conv1"].zero_point,
                   modules["stage2.conv2"].scale, modules["stage2.conv2"].zero_point, stride=1, padding=1, relu=False)

    s2_se1_w = modules["stage2.se.fc.1"].weight().int_repr().numpy().squeeze()
    s2_se1_sc = modules["stage2.se.fc.1"].weight().q_per_channel_scales().numpy()
    s2_se1_b = modules["stage2.se.fc.1"].bias().detach().numpy()
    s2_se2_w = modules["stage2.se.fc.3"].weight().int_repr().numpy().squeeze()
    s2_se2_sc = modules["stage2.se.fc.3"].weight().q_per_channel_scales().numpy()
    s2_se2_b = modules["stage2.se.fc.3"].bias().detach().numpy()

    s2_sig = se_block(s2_c2, modules["stage2.conv2"].scale, modules["stage2.conv2"].zero_point,
                      s2_se1_w, s2_se1_sc, s2_se1_b, modules["stage2.se.fc.1"].scale, modules["stage2.se.fc.1"].zero_point,
                      s2_se2_w, s2_se2_sc, s2_se2_b, modules["stage2.se.fc.3"].scale, modules["stage2.se.fc.3"].zero_point)

    s2_out = add_relu_mul(s2_c2, s2_sig, s2_sc,
                          modules["stage2.conv2"].scale, modules["stage2.conv2"].zero_point,
                          modules["stage2.shortcut.0"].scale, modules["stage2.shortcut.0"].zero_point,
                          getattr(q_model, "stage2_se_scale_0").item(), getattr(q_model, "stage2_se_zero_point_0").item(),
                          getattr(q_model, "stage2_relu_scale_0").item(), getattr(q_model, "stage2_relu_zero_point_0").item())

    # Stage 2 complete
    # 4. Stage 3
    print("Computing Stage 3...")
    s3_sc_w = modules["stage3.shortcut.0"].weight().int_repr().numpy()
    s3_sc_sc = modules["stage3.shortcut.0"].weight().q_per_channel_scales().numpy()
    s3_sc_b = modules["stage3.shortcut.0"].bias().detach().numpy()
    s3_sc = conv2d(s2_out, s3_sc_w, s3_sc_sc, s3_sc_b, getattr(q_model, "stage2_relu_scale_0").item(), getattr(q_model, "stage2_relu_zero_point_0").item(),
                   modules["stage3.shortcut.0"].scale, modules["stage3.shortcut.0"].zero_point, stride=2, padding=0, relu=False)

    s3_c1_w = modules["stage3.conv1"].weight().int_repr().numpy()
    s3_c1_sc = modules["stage3.conv1"].weight().q_per_channel_scales().numpy()
    s3_c1_b = modules["stage3.conv1"].bias().detach().numpy()
    s3_c1 = conv2d(s2_out, s3_c1_w, s3_c1_sc, s3_c1_b, getattr(q_model, "stage2_relu_scale_0").item(), getattr(q_model, "stage2_relu_zero_point_0").item(),
                   modules["stage3.conv1"].scale, modules["stage3.conv1"].zero_point, stride=2, padding=1, relu=True)

    s3_c2_w = modules["stage3.conv2"].weight().int_repr().numpy()
    s3_c2_sc = modules["stage3.conv2"].weight().q_per_channel_scales().numpy()
    s3_c2_b = modules["stage3.conv2"].bias().detach().numpy()
    s3_c2 = conv2d(s3_c1, s3_c2_w, s3_c2_sc, s3_c2_b, modules["stage3.conv1"].scale, modules["stage3.conv1"].zero_point,
                   modules["stage3.conv2"].scale, modules["stage3.conv2"].zero_point, stride=1, padding=1, relu=False)

    s3_se1_w = modules["stage3.se.fc.1"].weight().int_repr().numpy().squeeze()
    s3_se1_sc = modules["stage3.se.fc.1"].weight().q_per_channel_scales().numpy()
    s3_se1_b = modules["stage3.se.fc.1"].bias().detach().numpy()
    s3_se2_w = modules["stage3.se.fc.3"].weight().int_repr().numpy().squeeze()
    s3_se2_sc = modules["stage3.se.fc.3"].weight().q_per_channel_scales().numpy()
    s3_se2_b = modules["stage3.se.fc.3"].bias().detach().numpy()

    s3_sig = se_block(s3_c2, modules["stage3.conv2"].scale, modules["stage3.conv2"].zero_point,
                      s3_se1_w, s3_se1_sc, s3_se1_b, modules["stage3.se.fc.1"].scale, modules["stage3.se.fc.1"].zero_point,
                      s3_se2_w, s3_se2_sc, s3_se2_b, modules["stage3.se.fc.3"].scale, modules["stage3.se.fc.3"].zero_point)

    s3_out = add_relu_mul(s3_c2, s3_sig, s3_sc,
                          modules["stage3.conv2"].scale, modules["stage3.conv2"].zero_point,
                          modules["stage3.shortcut.0"].scale, modules["stage3.shortcut.0"].zero_point,
                          getattr(q_model, "stage3_se_scale_0").item(), getattr(q_model, "stage3_se_zero_point_0").item(),
                          getattr(q_model, "stage3_relu_scale_0").item(), getattr(q_model, "stage3_relu_zero_point_0").item())

    # Stage 3 complete
    # 5. Stage 4
    print("Computing Stage 4...")
    s4_sc_w = modules["stage4.shortcut.0"].weight().int_repr().numpy()
    s4_sc_sc = modules["stage4.shortcut.0"].weight().q_per_channel_scales().numpy()
    s4_sc_b = modules["stage4.shortcut.0"].bias().detach().numpy()
    s4_sc = conv2d(s3_out, s4_sc_w, s4_sc_sc, s4_sc_b, getattr(q_model, "stage3_relu_scale_0").item(), getattr(q_model, "stage3_relu_zero_point_0").item(),
                   modules["stage4.shortcut.0"].scale, modules["stage4.shortcut.0"].zero_point, stride=2, padding=0, relu=False)

    s4_c1_w = modules["stage4.conv1"].weight().int_repr().numpy()
    s4_c1_sc = modules["stage4.conv1"].weight().q_per_channel_scales().numpy()
    s4_c1_b = modules["stage4.conv1"].bias().detach().numpy()
    s4_c1 = conv2d(s3_out, s4_c1_w, s4_c1_sc, s4_c1_b, getattr(q_model, "stage3_relu_scale_0").item(), getattr(q_model, "stage3_relu_zero_point_0").item(),
                   modules["stage4.conv1"].scale, modules["stage4.conv1"].zero_point, stride=2, padding=1, relu=True)

    s4_c2_w = modules["stage4.conv2"].weight().int_repr().numpy()
    s4_c2_sc = modules["stage4.conv2"].weight().q_per_channel_scales().numpy()
    s4_c2_b = modules["stage4.conv2"].bias().detach().numpy()
    s4_c2 = conv2d(s4_c1, s4_c2_w, s4_c2_sc, s4_c2_b, modules["stage4.conv1"].scale, modules["stage4.conv1"].zero_point,
                   modules["stage4.conv2"].scale, modules["stage4.conv2"].zero_point, stride=1, padding=1, relu=False)

    s4_se1_w = modules["stage4.se.fc.1"].weight().int_repr().numpy().squeeze()
    s4_se1_sc = modules["stage4.se.fc.1"].weight().q_per_channel_scales().numpy()
    s4_se1_b = modules["stage4.se.fc.1"].bias().detach().numpy()
    s4_se2_w = modules["stage4.se.fc.3"].weight().int_repr().numpy().squeeze()
    s4_se2_sc = modules["stage4.se.fc.3"].weight().q_per_channel_scales().numpy()
    s4_se2_b = modules["stage4.se.fc.3"].bias().detach().numpy()

    s4_sig = se_block(s4_c2, modules["stage4.conv2"].scale, modules["stage4.conv2"].zero_point,
                      s4_se1_w, s4_se1_sc, s4_se1_b, modules["stage4.se.fc.1"].scale, modules["stage4.se.fc.1"].zero_point,
                      s4_se2_w, s4_se2_sc, s4_se2_b, modules["stage4.se.fc.3"].scale, modules["stage4.se.fc.3"].zero_point)

    s4_out = add_relu_mul(s4_c2, s4_sig, s4_sc,
                          modules["stage4.conv2"].scale, modules["stage4.conv2"].zero_point,
                          modules["stage4.shortcut.0"].scale, modules["stage4.shortcut.0"].zero_point,
                          getattr(q_model, "stage4_se_scale_0").item(), getattr(q_model, "stage4_se_zero_point_0").item(),
                          getattr(q_model, "stage4_relu_scale_0").item(), getattr(q_model, "stage4_relu_zero_point_0").item())

    # 6. Global Average Pooling & Classifier
    print("Computing GAP & Classifier...")
    gap_out = np.round(np.mean(s4_out.astype(np.float32), axis=(1, 2))).astype(np.uint8)
    gap_sc = getattr(q_model, "stage4_relu_scale_0").item()
    gap_zp = getattr(q_model, "stage4_relu_zero_point_0").item()

    fc = modules["classifier.1"]
    fc_w = fc.weight().int_repr().numpy().squeeze()
    fc_w_sc = fc.weight().q_per_channel_scales().numpy().item()
    fc_b = fc.bias().detach().numpy().item()

    acc = np.sum((gap_out.astype(np.int32) - gap_zp) * fc_w.astype(np.int32))
    sim_logit = acc * (gap_sc * fc_w_sc) + fc_b
    sim_prob = 1.0 / (1.0 + np.exp(-sim_logit))

    print(f"\nRESULTS for {img_name}:")
    print(f"  Simulated Logit = {sim_logit:.6f}, Prob = {sim_prob:.6f}")
    print(f"  PyTorch   Logit = {ref_logit:.6f}, Prob = {ref_prob:.6f}")
    print(f"  Logit Difference = {abs(sim_logit - ref_logit):.8f}")
    print(f"  Prob  Difference = {abs(sim_prob - ref_prob):.8f}")

if __name__ == "__main__":
    test_image("CHNCXR_0404_1.png")
    test_image("CHNCXR_0001_0.png")
