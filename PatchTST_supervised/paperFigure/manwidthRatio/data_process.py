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

import time
start_time = time.time()
# ----------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--group_size", type=int, default=128)
args = parser.parse_args()

methods = ['qwt_wikitext', 'qwt_c4']
methods = ['w4a8', 'w8a8']
task = ['wikitext', 'c4']

methods = ['w8a8']
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
for method in methods:
    data_type_ratio[method] = [[None for _ in range(5)] for _ in range(len(models))]
    for model_idx, model in enumerate(models):
        print(f'======={model}=======')
        if 'wiki' in task:
            if model == 'TinyLlama-1.1B-Chat-v1.0':
                from saved_tensor.w4a8.TinyLlama_1_1B_Chat_v1_0_quant_wikitext.activation_key import act_keys
                from saved_tensor.w8a8.TinyLlama_1_1B_Chat_v1_0_quant_wikitext.weight_key import wgt_keys
            elif model == 'llama-2-7b-hf':
                from saved_tensor.w4a8.llama_2_7b_hf_quant_wikitext.activation_key import act_keys
                from saved_tensor.w8a8.llama_2_7b_hf_quant_wikitext.weight_key import wgt_keys
            elif model == 'Meta-Llama-3-8B':
                from saved_tensor.w4a8.Meta_Llama_3_8B_quant_wikitext.activation_key import act_keys
                from saved_tensor.w8a8.Meta_Llama_3_8B_quant_wikitext.weight_key import wgt_keys
            elif model == 'Qwen2.5-0.5B':
                from saved_tensor.w4a8.Qwen2_5_0_5B_quant_wikitext.activation_key import act_keys
                from saved_tensor.w8a8.Qwen2_5_0_5B_quant_wikitext.weight_key import wgt_keys
            elif model == 'Qwen2.5-1.5B':
                from saved_tensor.w4a8.Qwen2_5_1_5B_quant_wikitext.activation_key import act_keys
                from saved_tensor.w8a8.Qwen2_5_1_5B_quant_wikitext.weight_key import wgt_keys
            elif model == 'Qwen2.5-7B':
                from saved_tensor.w4a8.Qwen2_5_7B_quant_wikitext.activation_key import act_keys
                from saved_tensor.w8a8.Qwen2_5_7B_quant_wikitext.weight_key import wgt_keys
        if 'c4' in task:
            if model == 'TinyLlama-1.1B-Chat-v1.0':
                from saved_tensor.w4a8.TinyLlama_1_1B_Chat_v1_0_quant_c4.activation_key import act_keys
                from saved_tensor.w8a8.TinyLlama_1_1B_Chat_v1_0_quant_c4.weight_key import wgt_keys
            elif model == 'llama-2-7b-hf':
                from saved_tensor.w4a8.llama_2_7b_hf_quant_c4.activation_key import act_keys
                from saved_tensor.w8a8.llama_2_7b_hf_quant_c4.weight_key import wgt_keys
            elif model == 'Meta-Llama-3-8B':
                from saved_tensor.w4a8.Meta_Llama_3_8B_quant_c4.activation_key import act_keys
                from saved_tensor.w8a8.Meta_Llama_3_8B_quant_c4.weight_key import wgt_keys
            elif model == 'Qwen2.5-0.5B':
                from saved_tensor.w4a8.Qwen2_5_0_5B_quant_c4.activation_key import act_keys
                from saved_tensor.w8a8.Qwen2_5_0_5B_quant_c4.weight_key import wgt_keys
            elif model == 'Qwen2.5-1.5B':
                from saved_tensor.w4a8.Qwen2_5_1_5B_quant_c4.activation_key import act_keys
                from saved_tensor.w8a8.Qwen2_5_1_5B_quant_c4.weight_key import wgt_keys
            elif model == 'Qwen2.5-7B':
                from saved_tensor.w4a8.Qwen2_5_7B_quant_c4.activation_key import act_keys
                from saved_tensor.w8a8.Qwen2_5_7B_quant_c4.weight_key import wgt_keys

        model_rename = model.replace("-", "_").replace(".", "_")
        if 'w4a8' in method: # squeeze activation
            safetensors_path = f"saved_tensor/{method}/{model_rename}_quant_{task}/activation.safetensors"
            keys = act_keys
        if 'w8a8' in method: # squeeze weight
            safetensors_path = f"saved_tensor/{method}/{model_rename}_quant_{task}/weight.safetensors"
            keys = wgt_keys
        state_dict = load_file(safetensors_path)

        group_sizes = [32, 64, 128, 256]
        group_sizes = [128]
        for group_size in group_sizes:
            int4_counter_org = int5_6_counter_org = int7_8_counter_org = 0
            int4_counter_shift = 0
            int5_counter_shift = 0
            int6_counter_shift = 0
            int7_counter_shift = 0
            int8_counter_shift = 0
            sub_tensor_count = 0

            org_bits = enc_bits = 0
            with open(f'intRatio_{method}.txt', 'a') as f:
                f.writelines(f'--{model_rename} {group_size}\n')
                for idx, key in enumerate(keys):
                    w_int8 = state_dict[key].to('cuda').squeeze()
                    w_int8 = w_int8.reshape(-1, group_size)
                    max = w_int8.amax(dim=-1, keepdim=True)
                    min = w_int8.amin(dim=-1, keepdim=True)
                    '''
                    present_range = max
                    int4 = present_range <= 15
                    int4_counter_org += int4.sum()
                    int5_6 = (present_range <= 63) * (present_range > 15)
                    int5_6_counter_org += int5_6.sum()
                    int7_8 = present_range > 63
                    int7_8_counter_org += int7_8.sum()
                    '''

                    present_range = max - min
                    int4 = present_range <= 15
                    int5 = (present_range <= 31) * (present_range > 15)
                    int6 = (present_range <= 63) * (present_range > 31)
                    int7 = (present_range <= 127) * (present_range > 63)
                    int8 = present_range > 127
                    int4_counter_shift += int4.sum()
                    int5_counter_shift += int5.sum()
                    int6_counter_shift += int6.sum()
                    int7_counter_shift += int7.sum()
                    int8_counter_shift += int8.sum()

                    sub_tensor_count += w_int8.size(0)

                assert (int4_counter_shift + int5_counter_shift + int6_counter_shift + int7_counter_shift + int8_counter_shift) == sub_tensor_count
                # f.writelines(f'int4: {int4_counter_org:>8} ({int4_counter_org*100/sub_tensor_count:5.2f}%), int5_6: {int5_6_counter_org:>8} ({int5_6_counter_org*100/sub_tensor_count:5.2f}%), int7_8: {int7_8_counter_org:>8} ({int7_8_counter_org*100/sub_tensor_count:5.2f}%)\n')
                f.writelines(
                    f'int4: {int4_counter_shift:>8} ({int4_counter_shift*100/sub_tensor_count:5.2f}%), '
                    f'int5: {int5_counter_shift:>8} ({int5_counter_shift*100/sub_tensor_count:5.2f}%), '
                    f'int6: {int6_counter_shift:>8} ({int6_counter_shift*100/sub_tensor_count:5.2f}%), '
                    f'int7: {int7_counter_shift:>8} ({int7_counter_shift*100/sub_tensor_count:5.2f}%), '
                    f'int8: {int8_counter_shift:>8} ({int8_counter_shift*100/sub_tensor_count:5.2f}%)\n\n'
                )
                data_type_ratio[method][model_idx] = [int4_counter_shift*100/sub_tensor_count, int5_counter_shift*100/sub_tensor_count, int6_counter_shift*100/sub_tensor_count, int7_counter_shift*100/sub_tensor_count, int8_counter_shift*100/sub_tensor_count]
    data_type_ratio[method] = torch.tensor(data_type_ratio[method])
    save_file(data_type_ratio, f"data_type_ratio_{method}_{task}.safetensors")
# ----------------------------------------------------------
with open(f'intRatio.txt', 'a') as f:
    end_time = time.time()
    duration = end_time - start_time
    hour = duration // 3600
    minute = (duration % 3600) // 60
    second = duration % 60
    # f.writelines(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n\n')
    print(f'>>>RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s')
                                




