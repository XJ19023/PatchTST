'''
    -计算int4 int5_6 int7_8的占比
'''
import os
import random
import sys
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file
from matplotlib import pyplot as plt

from plot import plot_fig

import time
start_time = time.time()
# ----------------------------------------------------------
methods = ['qwt_wikitext']

models = [
            # 'llama-2-7b-hf',
            'Qwen2.5-7B'
            ]
for method in methods:
    for model in models:
        print(f'======={model}=======')
        if model == 'llama-2-7b-hf':
            from saved_tensor.llama_2_7b_hf_qwt_wikitext.activation_key import act_keys
        elif model == 'Qwen2.5-7B':
            from saved_tensor.Qwen2_5_7B_qwt_wikitext.activation_key import act_keys

        model_rename = model.replace("-", "_").replace(".", "_")
        safetensors_path_act = f"saved_tensor/{model_rename}_{method}/activation.safetensors"
        state_dict = load_file(safetensors_path_act)
        keys = act_keys

        w_int8 = state_dict['model.layers.3.self_attn.q_proj.act_0'].squeeze()
        org_shape = w_int8.shape
        w_int8 = w_int8.reshape(-1, 64)
        min = w_int8.amin(dim=-1, keepdim=True)
        w_int8 = w_int8 - min
        w_int8 = w_int8.reshape(org_shape)
        w_int8 = w_int8[:256, :256]
        # print(w_int8)
        w_int8 = w_int8.float()
        # print(w_int8)
        # int4 = tensor < 16
        # print(int4.sum(), int4.numel(), int4.sum()/int4.numel())
        plot_fig(w_int8.cpu().numpy())


        profile_data = False
        if profile_data:
            int4_counter_org = int5_6_counter_org = int7_8_counter_org = 0
            int4_counter_shift = int5_6_counter_shift = int7_8_counter_shift = 0
            sub_tensor_count = 0    
            with open(f'intRatio_{method}.txt', 'a') as f:
                f.writelines(f'--{model_rename}\n')
                for idx, key in enumerate(keys):
                    w_int8 = state_dict[key].to('cuda').squeeze()


                    int7_8 = w_int8 > 15
                    int7_8_counter_org = int7_8.sum()

                    w_int8 = w_int8.reshape(-1, 128)
                    max = w_int8.amax(dim=-1, keepdim=True)
                    min = w_int8.amin(dim=-1, keepdim=True)
                    present_range = max - min
                    int4 = present_range <= 15
                    int4_counter_shift += int4.sum()
                    int5_6 = (present_range <= 63) * (present_range > 15)
                    int5_6_counter_shift += int5_6.sum()
                    int7_8 = present_range > 63
                    int7_8_counter_shift = int7_8.sum()
                    sub_tensor_count = w_int8.size(0)
            
                    f.writelines(f'{key} ({int7_8_counter_shift*100/sub_tensor_count:5.2f}% -- {int7_8_counter_org*100/w_int8.numel():5.2f}%)\n')
    # ----------------------------------------------------------
    with open(f'intRatio_{method}.txt', 'a') as f:
        end_time = time.time()
        duration = end_time - start_time
        hour = duration // 3600
        minute = (duration % 3600) // 60
        second = duration % 60
        f.writelines(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n\n')
                                




