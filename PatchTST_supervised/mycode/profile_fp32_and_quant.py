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
# from logs.traffic_336_336.save_tensors.activation_key import act_keys

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

safetensors_path = f"logs/traffic_336_336/save_tensors/activation.safetensors"
keys = act_keys
state_dict = load_file(safetensors_path)


for k in keys:

    data_fp32 = state_dict[f'{k}_fp32'].reshape(-1, state_dict[f'{k}_fp32'].shape[-1])
    data_quant = state_dict[f'{k}_quant'].reshape(-1, state_dict[f'{k}_quant'].shape[-1])
    mse = loss(data_fp32, data_quant)

    colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255), (70/255, 120/255, 142/255)]

    # 创建一个大图，包含两个3D子图
    fig = plt.figure(figsize=(12, 4), dpi=300)

    # 第一个子图 - data_fp32
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    tensor_2d = data_fp32[:100, :]
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
    ax1.set_title('FP32 Data Distribution')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=dz.min(), vmax=dz.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=5, pad=0.1)

    # 第二个子图 - data_quant
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    tensor_2d = data_quant[:100, :]
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
    ax2.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Token')
    ax2.set_zlabel('Value')
    ax2.set_title('Quantized Data Distribution')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=dz.min(), vmax=dz.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, shrink=0.5, aspect=5, pad=0.1)

    # 在两个子图之间添加文字
    fig.text(0.5, 0.8,  # x=0.5是中间位置，y=0.5是垂直中间
            f'mse = {mse:.8f}',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            transform=fig.transFigure)  # 使用figure坐标系统

    plt.tight_layout()
    os.makedirs(f'logs/traffic_336_336/fig', exist_ok=True)
    plt.savefig(f'logs/traffic_336_336/fig/{k[23:]}.png')
    plt.close()
    print(f'logs/traffic_336_336/fig/{k[23:]}.png')

    # break