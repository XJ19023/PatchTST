'''
    -计算int4 int5_6 int7_8的占比
'''
import argparse
import os
import random
import sys
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file
from matplotlib import pyplot as plt
import torch.nn as nn

import time
start_time = time.time()
# ----------------------------------------------------------

# 创建MSE损失函数对象
loss = nn.MSELoss()

parser = argparse.ArgumentParser()
parser.add_argument("--group_size", type=int, default=128)
args = parser.parse_args()

methods = ['qwt_wikitext', 'qwt_c4']
methods = ['w4a8', 'w4a8']
task = ['wikitext', 'c4']

methods = ['w4a8']
task = 'wikitext'


models = [
            'TinyLlama-1.1B-Chat-v1.0',
            'llama-2-7b-hf',
            'Meta-Llama-3-8B',
            'Qwen2.5-0.5B',
            'Qwen2.5-1.5B',
            'Qwen2.5-7B'
            ]
models = ['TinyLlama-1.1B-Chat-v1.0']
data_type_ratio = {}
# Ensure the parent directory of 'saved_tensor' is in sys.path
saved_tensor_dir = os.path.join(os.path.dirname(__file__), '..')
saved_tensor_dir = os.path.abspath(saved_tensor_dir)
if saved_tensor_dir not in sys.path:
    sys.path.append(saved_tensor_dir)

for method in methods:
    data_type_ratio[method] = [[None for _ in range(5)] for _ in range(len(models))]
    for model_idx, model in enumerate(models):
        print(f'======={model}=======')
        if 'wiki' in task:
            if model == 'TinyLlama-1.1B-Chat-v1.0':
                from saved_tensor.TinyLlama_1_1B_Chat_v1_0_quant_wikitext.activation_key import act_keys
                from saved_tensor.TinyLlama_1_1B_Chat_v1_0_quant_wikitext.weight_key import wgt_keys
            elif model == 'llama-2-7b-hf':
                from saved_tensor.llama_2_7b_hf_quant_wikitext.activation_key import act_keys
                from saved_tensor.llama_2_7b_hf_quant_wikitext.weight_key import wgt_keys
            elif model == 'Meta-Llama-3-8B':
                from saved_tensor.Meta_Llama_3_8B_quant_wikitext.activation_key import act_keys
                from saved_tensor.Meta_Llama_3_8B_quant_wikitext.weight_key import wgt_keys
            elif model == 'Qwen2.5-0.5B':
                from saved_tensor.Qwen2_5_0_5B_quant_wikitext.activation_key import act_keys
                from saved_tensor.Qwen2_5_0_5B_quant_wikitext.weight_key import wgt_keys
            elif model == 'Qwen2.5-1.5B':
                from saved_tensor.Qwen2_5_1_5B_quant_wikitext.activation_key import act_keys
                from saved_tensor.Qwen2_5_1_5B_quant_wikitext.weight_key import wgt_keys
            elif model == 'Qwen2.5-7B':
                from saved_tensor.Qwen2_5_7B_quant_wikitext.activation_key import act_keys
                from saved_tensor.Qwen2_5_7B_quant_wikitext.weight_key import wgt_keys
        if 'c4' in task:
            if model == 'TinyLlama-1.1B-Chat-v1.0':
                from saved_tensor.TinyLlama_1_1B_Chat_v1_0_quant_c4.activation_key import act_keys
                from saved_tensor.TinyLlama_1_1B_Chat_v1_0_quant_c4.weight_key import wgt_keys
            elif model == 'llama-2-7b-hf':
                from saved_tensor.llama_2_7b_hf_quant_c4.activation_key import act_keys
                from saved_tensor.llama_2_7b_hf_quant_c4.weight_key import wgt_keys
            elif model == 'Meta-Llama-3-8B':
                from saved_tensor.Meta_Llama_3_8B_quant_c4.activation_key import act_keys
                from saved_tensor.Meta_Llama_3_8B_quant_c4.weight_key import wgt_keys
            elif model == 'Qwen2.5-0.5B':
                from saved_tensor.Qwen2_5_0_5B_quant_c4.activation_key import act_keys
                from saved_tensor.Qwen2_5_0_5B_quant_c4.weight_key import wgt_keys
            elif model == 'Qwen2.5-1.5B':
                from saved_tensor.Qwen2_5_1_5B_quant_c4.activation_key import act_keys
                from saved_tensor.Qwen2_5_1_5B_quant_c4.weight_key import wgt_keys
            elif model == 'Qwen2.5-7B':
                from saved_tensor.Qwen2_5_7B_quant_c4.activation_key import act_keys
                from saved_tensor.Qwen2_5_7B_quant_c4.weight_key import wgt_keys

        model_rename = model.replace("-", "_").replace(".", "_")
        if 'w4a8' in method: # squeeze activation
            safetensors_path = f"saved_tensor/{model_rename}_quant_{task}/activation.safetensors"
            keys = act_keys
        state_dict = load_file(safetensors_path)

        group_sizes = [32, 64, 128, 256]
        group_sizes = [64]
        clamp_type = ['int5 int6']
        for group_size in group_sizes:
            with open(f'intRatio_{method}.txt', 'a') as f:
                f.writelines(f'--{model_rename} {group_size}\n')
                for idx, key in enumerate(keys):
                    w_int = state_dict[key].to('cuda').squeeze()

                    org_w_shape = w_int.shape
                    w_int = w_int.reshape(-1, group_size)
                    max = w_int.amax(dim=-1, keepdim=True)
                    min = w_int.amin(dim=-1, keepdim=True) 
                    present_range = max - min
                    # even = (max + min) // 2 # stored tensor
                    w_int -= min

                    clamp_idx_int1 = present_range <= 1
                    clamp_idx_int2 = (present_range <= 3) * (present_range > 1)
                    clamp_idx_int3 = (present_range <= 7) * (present_range > 3)
                    clamp_idx_int4 = (present_range <= 15) * (present_range > 7)
                    clamp_idx_int5 = (present_range <= 31) * (present_range > 15)
                    clamp_idx_int6 = (present_range <= 63) * (present_range > 31)
                    clamp_idx_int7 = (present_range <= 127) * (present_range > 63)
                    clamp_idx_int8 = present_range > 127

                    clamp_idx_int1 =clamp_idx_int1.expand(-1, w_int.size(-1))
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

                    # default: int1~4 clamp, int5~6 speculative clamp, int7~8 no clamp
                    # decoder: int8(int7, int6, int5), int6(clamp) << 2, int5(clamp, int4) << 1
                    w_int1_clamp = (w_int1_clamp // 2) * 2
                    w_int2_clamp = (w_int2_clamp // 2) * 2
                    w_int3_clamp = (w_int3_clamp // 2) * 2
                    w_int4_clamp = (w_int4_clamp // 2) * 2

                    w_int5_error = w_int5_clamp - (w_int5_clamp // 2) * 2
                    breakpoint()
                    w_int5_error_ = w_int5_clamp[clamp_idx_int5] - (w_int5_clamp[clamp_idx_int5] // 2) * 2
                    w_int5_error_.numel()
                    mse_int5 = loss(w_int5_clamp[clamp_idx_int5], (w_int5_clamp[clamp_idx_int5] // 2) * 2)

                    w_int5_clamp = (w_int5_clamp // 2) * 2 if 'int5' in clamp_type else w_int5_clamp
                    w_int6_clamp = (w_int6_clamp // 4) * 4 if 'int6' in clamp_type else w_int6_clamp

                    w_int = w_int1_clamp + w_int2_clamp + w_int3_clamp + w_int4_clamp \
                        + w_int5_clamp + w_int6_clamp + w_int7_clamp + w_int8_clamp + min
                    
                    w_int = w_int.reshape(org_w_shape)
                    exit()
# ----------------------------------------------------------
with open(f'intRatio.txt', 'a') as f:
    end_time = time.time()
    duration = end_time - start_time
    hour = duration // 3600
    minute = (duration % 3600) // 60
    second = duration % 60
    # f.writelines(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n\n')
    print(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s')
                                




