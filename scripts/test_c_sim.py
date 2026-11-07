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

# Load input
tb_img_path = "data/images/CHNCXR_0404_1.png"
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
img_tb = Image.open(tb_img_path).convert("L")
t_tb = val_tf(img_tb).unsqueeze(0)

# Input quantization params
in_scale = getattr(q_model, "stem_0_block_0_input_scale_0").item()
in_zp = getattr(q_model, "stem_0_block_0_input_zero_point_0").item()
q_input = torch.quantize_per_tensor(t_tb, in_scale, in_zp, torch.quint8).int_repr().squeeze().numpy().astype(np.uint8)

def conv2d_core(input_chw, in_channels, in_h, in_w, out_channels, out_h, out_w, 
                kernel_size, stride, padding, in_scale, in_zp,
                weights, w_scale, bias, out_scale, out_zp, has_relu):
    # input_chw: list of 1D arrays or shape (C, H, W)
    out = np.zeros((out_channels, out_h, out_w), dtype=np.uint8)
    pad = padding
    
    for oc in range(out_channels):
        w_sc = w_scale[oc]
        b = bias[oc]
        # weights shape: [out_channels, in_channels, kernel_size, kernel_size]
        w_oc = weights[oc]
        
        for oh in range(out_h):
            ih_base = oh * stride - pad
            for ow in range(out_w):
                iw_base = ow * stride - pad
                
                acc = 0
                for ic in range(in_channels):
                    in_plane = input_chw[ic]
                    w_kernel = w_oc[ic]
                    for kh in range(kernel_size):
                        ih = ih_base + kh
                        if 0 <= ih < in_h:
                            for kw in range(kernel_size):
                                iw = iw_base + kw
                                if 0 <= iw < in_w:
                                    x_val = int(in_plane[ih, iw]) - in_zp
                                    w_val = int(w_kernel[kh, kw])
                                    acc += x_val * w_val
                
                # Real output
                r = acc * (in_scale * w_sc) + b
                if has_relu and r < 0.0:
                    r = 0.0
                
                # Quantize
                q = int(np.round(r / out_scale)) + out_zp
                if q < 0: q = 0
                if q > 255: q = 255
                out[oc, oh, ow] = q
    return out

print("Simulating layers...")
# 1. Stem
stem_w = modules["stem.0.block.0"].weight().int_repr().numpy()
stem_w_sc = modules["stem.0.block.0"].weight().q_per_channel_scales().numpy()
stem_b = modules["stem.0.block.0"].bias().detach().numpy()
stem_s = modules["stem.0.block.0"].scale
stem_zp = modules["stem.0.block.0"].zero_point

stem_out = conv2d_core(q_input[np.newaxis, :, :], 1, 224, 224, 16, 112, 112,
                       3, 2, 1, in_scale, in_zp,
                       stem_w, stem_w_sc, stem_b, stem_s, stem_zp, True)

print("Stem max:", stem_out.max(), "min:", stem_out.min())

# Check against PyTorch stem
with torch.no_grad():
    pt_stem = modules["stem.0.block.0"](torch.quantize_per_tensor(t_tb, in_scale, in_zp, torch.quint8)).int_repr().squeeze().numpy()
print("Stem diff:", np.max(np.abs(stem_out - pt_stem)))

def run_resblock_sim(in_tensor, in_ch, out_ch, size, stride,
                     c1_mod, c2_mod, sc_mod, se1_mod, se2_mod,
                     se_scale, se_zp, relu_scale, relu_zp):
    out_size = size // stride
    # 1. conv1
    c1_w = c1_mod.weight().int_repr().numpy()
    c1_w_sc = c1_mod.weight().q_per_channel_scales().numpy()
    c1_b = c1_mod.bias().detach().numpy()
    c1_out = conv2d_core(in_tensor, in_ch, size, size, out_ch, out_size, out_size,
                         3, stride, 1, in_scale if False else None, None,
                         c1_w, c1_w_sc, c1_b, c1_mod.scale, c1_mod.zero_point, True) # we will pass in_scale
    return c1_out
