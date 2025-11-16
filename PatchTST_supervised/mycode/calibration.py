import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def pseudo_quantize_tensor( w, 
                            n_bit=8, 
                            name=None,
                            ):

    org_tensor = w
    org_w_shape = w.shape

    max_val = w.amax(dim=-1, keepdim=True)
    min_val = w.amin(dim=-1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w_int = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) # quantized INT8

    if 'down_proj' not in name:
        clamp_block_size = 64
        w_int = w_int.reshape(-1, clamp_block_size) # clamp block size
        max = w_int.amax(dim=-1, keepdim=True)
        min = w_int.amin(dim=-1, keepdim=True) 
        present_range = max - min
        w_int -= min

        clamp_idx_int1 = present_range <= 1
        clamp_idx_int2 = (present_range <= 3) * (present_range > 1)
        clamp_idx_int3 = (present_range <= 7) * (present_range > 3)
        clamp_idx_int4 = (present_range <= 15) * (present_range > 7)
        clamp_idx_int5 = (present_range <= 31) * (present_range > 15)
        clamp_idx_int6 = (present_range <= 63) * (present_range > 31)
        clamp_idx_int7 = (present_range <= 127) * (present_range > 63)
        clamp_idx_int8 = present_range > 127

        clamp_idx_int1 =clamp_idx_int1.expand(-1, w_int.size(-1)) # scales.squeeze().expand(-1, mse.size(-1))
        clamp_idx_int2 =clamp_idx_int2.expand(-1, w_int.size(-1))
        clamp_idx_int3 =clamp_idx_int3.expand(-1, w_int.size(-1))
        clamp_idx_int4 =clamp_idx_int4.expand(-1, w_int.size(-1))
        clamp_idx_int5 =clamp_idx_int5.expand(-1, w_int.size(-1)) 
        clamp_idx_int6 =clamp_idx_int6.expand(-1, w_int.size(-1))
        clamp_idx_int7 =clamp_idx_int7.expand(-1, w_int.size(-1))
        clamp_idx_int8 =clamp_idx_int8.expand(-1, w_int.size(-1))

        w_int1_clamp = w_int.masked_fill(~clamp_idx_int1, 0)
        w_int2_clamp = w_int.masked_fill(~clamp_idx_int2, 0)
        w_int3_clamp = w_int.masked_fill(~clamp_idx_int3, 0)
        w_int4_clamp = w_int.masked_fill(~clamp_idx_int4, 0)
        w_int5_clamp = w_int.masked_fill(~clamp_idx_int5, 0)
        w_int6_clamp = w_int.masked_fill(~clamp_idx_int6, 0)
        w_int7_clamp = w_int.masked_fill(~clamp_idx_int7, 0)
        w_int8_clamp = w_int.masked_fill(~clamp_idx_int8, 0)

        clamp_type = 'int5 int6'
        # default: int1~4 clamp, int5~6 speculative clamp, int7~8 no clamp
        # decoder: int8(int7, int6, int5), int6(clamp) << 2, int5(clamp, int4) << 1
        w_int1_clamp = (w_int1_clamp // 2) * 2
        w_int2_clamp = (w_int2_clamp // 2) * 2
        w_int3_clamp = (w_int3_clamp // 2) * 2
        w_int4_clamp = (w_int4_clamp // 2) * 2

        mse = torch.mean((w_int5_clamp - (w_int5_clamp // 2) * 2) ** 2, dim=1)
        mse = mse.reshape(org_w_shape[1], -1)
        mse = mse * (scales.squeeze(0))
        mse = mse.reshape(-1, 1)
        mse_int5 = mse[clamp_idx_int5[:,0]]

        if 'self_attn' in name:
            w_int5_clamp_idx = ((mse < 0.05).expand(-1, w_int.size(-1))) * clamp_idx_int5
        if 'mlp' in name:
            w_int5_clamp_idx = ((mse < 0.01).expand(-1, w_int.size(-1))) * clamp_idx_int5
        w_int5_clamp_real = w_int5_clamp.masked_fill(~w_int5_clamp_idx, 0)
        w_int5_clamp_false = w_int5_clamp.masked_fill(w_int5_clamp_idx, 0)

        # breakpoint() # 
        with open('log/mse_int5.txt', 'a') as f:
            if mse_int5.numel() > 0:
                f.writelines(f'clamp int5: {w_int5_clamp_idx[:,0].sum():>7} ({w_int5_clamp_idx[:,0].sum()/clamp_idx_int5[:,0].sum():.4f}), '
                                f'int5: {clamp_idx_int5[:,0].sum():>7} ({clamp_idx_int5[:,0].sum()/mse.size(0):.4f}), total: {mse.size(0):>7} '
                                f'({mse_int5.min().item():.4f}, {mse_int5.mean().item():.4f}, {mse_int5.max().item():.4f}) '
                                f'>> {name}\n')

        mse = torch.mean((w_int6_clamp - (w_int6_clamp // 4) * 4) ** 2, dim=1)
        mse = mse.reshape(org_w_shape[1], -1)
        mse = mse * (scales.squeeze(0))
        mse = mse.reshape(-1, 1)
        mse_int6 = mse[clamp_idx_int6[:,0]]

        if 'self_attn' in name:
            w_int6_clamp_idx = ((mse < 0.08).expand(-1, w_int.size(-1))) * clamp_idx_int6
        if 'mlp' in name:
            w_int6_clamp_idx = ((mse < 0.02).expand(-1, w_int.size(-1))) * clamp_idx_int6
        w_int6_clamp_real = w_int6_clamp.masked_fill(~w_int6_clamp_idx, 0)
        w_int6_clamp_false = w_int6_clamp.masked_fill(w_int6_clamp_idx, 0)

        w_int5_clamp_real = (w_int5_clamp_real // 2) * 2 if 'int5' in clamp_type else w_int5_clamp_real
        w_int6_clamp_real = (w_int6_clamp_real // 4) * 4 if 'int6' in clamp_type else w_int6_clamp_real

        # breakpoint() # 
        with open('log/mse_int6.txt', 'a') as f:
            if mse_int6.numel() > 0:
                f.writelines(f'clamp int6: {w_int6_clamp_idx[:,0].sum():>7} ({w_int6_clamp_idx[:,0].sum()/clamp_idx_int6[:,0].sum():.4f}), '
                                f'int6: {clamp_idx_int6[:,0].sum():>7} ({clamp_idx_int6[:,0].sum()/mse.size(0):.4f}), total: {mse.size(0):>7} '
                                f'({mse_int6.min().item():.4f}, {mse_int6.mean().item():.4f}, {mse_int6.max().item():.4f}) '
                                f'>> {name}\n')

        w_int = w_int1_clamp + w_int2_clamp + w_int3_clamp + w_int4_clamp \
            + w_int5_clamp_real + w_int5_clamp_false + w_int6_clamp_real + w_int6_clamp_false \
                + w_int7_clamp + w_int8_clamp + min

        w_int = w_int.reshape(org_w_shape)
        
    w = (w_int - zeros) * scales

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w, scales

def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        print(name, tensor.shape)
        # hidden_dim = tensor.shape[-1]
        # tensor = tensor.view(-1, hidden_dim).abs().detach()
        # comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        # if name in act_scales:
        #     act_scales[name] = torch.max(act_scales[name], comming_max)
        # else:
        #     act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales

