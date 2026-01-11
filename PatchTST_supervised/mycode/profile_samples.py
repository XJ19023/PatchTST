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

import torch
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

start_time = time.time()
# ----------------------------------------------------------

# 创建MSE损失函数对象
loss = nn.MSELoss()

import sys
sys.path.append('/cephfs/juxin/PatchTST/PatchTST_supervised')
# from logs.weather_336_336.save_tensors.activation_key import act_keys

act_keys = ['model.backbone.encoder.layers.0.self_attn.W_Q',
'model.backbone.encoder.layers.0.self_attn.to_out.0',
'model.backbone.encoder.layers.0.ff.0',
'model.backbone.encoder.layers.0.ff.3',
'model.backbone.encoder.layers.1.self_attn.W_Q',
'model.backbone.encoder.layers.1.self_attn.to_out.0',
'model.backbone.encoder.layers.1.ff.0',
'model.backbone.encoder.layers.1.ff.3',
'model.backbone.encoder.layers.2.self_attn.W_Q',
'model.backbone.encoder.layers.2.self_attn.to_out.0',
'model.backbone.encoder.layers.2.ff.0',
'model.backbone.encoder.layers.2.ff.3',]

keys = act_keys
safetensors_path = f"logs/weather_336_336/fp32_tensor/case1/activation.safetensors"
state_dict_case1 = load_file(safetensors_path)
safetensors_path = f"logs/weather_336_336/fp32_tensor/case2/activation.safetensors"
state_dict_case2 = load_file(safetensors_path)
safetensors_path = f"logs/weather_336_336/fp32_tensor/case3/activation.safetensors"
state_dict_case3 = load_file(safetensors_path)


for k in keys:

    data_case1 = state_dict_case1[k].reshape(-1, state_dict_case1[k].shape[-1])
    data_case2 = state_dict_case2[k].reshape(-1, state_dict_case2[k].shape[-1])
    data_case3 = state_dict_case3[k].reshape(-1, state_dict_case3[k].shape[-1])
    # mse = loss(data_fp32, data_quant)

    colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255), (70/255, 120/255, 142/255)]

    # 创建一个大图，包含两个3D子图
    fig = plt.figure(figsize=(12, 4), dpi=300)
    
    # 第一个子图 - data_fp32
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    tensor_2d = data_case1[:200, :]
    data = tensor_2d.numpy()
    # 创建坐标
    x_pos, y_pos = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(data.flatten())
    dx = dy = 0.08
    dz = data.flatten()
    # 根据数值设置颜色
    colors = plt.cm.viridis((dz - dz.min()) / (dz.max() - dz.min()))
    ax1.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Token')
    ax1.set_zlabel('Value')
    ax1.set_title('Sample-1 Data Distribution')
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=dz.min(), vmax=dz.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5, pad=0.1)

    # 第二个子图 - data_fp32
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    tensor_2d = data_case2[:200, :]
    data = tensor_2d.numpy()
    # 创建坐标
    x_pos, y_pos = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(data.flatten())
    dx = dy = 0.08
    dz = data.flatten()
    # 根据数值设置颜色
    colors = plt.cm.viridis((dz - dz.min()) / (dz.max() - dz.min()))
    ax1.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Token')
    ax1.set_zlabel('Value')
    ax1.set_title('Sample-2 Data Distribution')
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=dz.min(), vmax=dz.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5, pad=0.1)

    # 第三个子图 - data_fp32
    ax1 = fig.add_subplot(1, 3, 3, projection='3d')
    tensor_2d = data_case3[:200, :]
    data = tensor_2d.numpy()
    # 创建坐标
    x_pos, y_pos = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(data.flatten())
    dx = dy = 0.08
    dz = data.flatten()
    # 根据数值设置颜色
    colors = plt.cm.viridis((dz - dz.min()) / (dz.max() - dz.min()))
    ax1.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors)
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Token')
    ax1.set_zlabel('Value')
    ax1.set_title('Sample-3 Data Distribution')
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=dz.min(), vmax=dz.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5, pad=0.1)



    # # 在两个子图之间添加文字
    # fig.text(0.5, 0.8,  # x=0.5是中间位置，y=0.5是垂直中间
    #         f'mse = {mse:.8f}',
    #         ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #         transform=fig.transFigure)  # 使用figure坐标系统

    plt.tight_layout()

    dir_path = f'logs/weather_336_336/fig_case'
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(f'{dir_path}/{k[23:]}.png')
    plt.close()
    print(f'{dir_path}/{k[23:]}.png')

    # break