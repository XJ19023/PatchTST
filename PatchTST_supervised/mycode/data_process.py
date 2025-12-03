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


from save_tensors.activation_key import act_keys

safetensors_path = f"save_tensors/activation.safetensors"
keys = act_keys
state_dict = load_file(safetensors_path)

# print(state_dict['model.backbone.encoder.layers.1.self_attn.W_K'].equal(state_dict['model.backbone.encoder.layers.1.self_attn.W_Q']))
# exit()

for k in keys:
    # print(k, state_dict[k].shape)
    if 'layers.1' not in k or 'W_K' in k or 'W_V' in k:
        continue

    data = state_dict[k].reshape(-1, state_dict[k].shape[-1]) 
    # print(data.shape)        

    # break             



    # 创建示例数据
    tensor_2d = data[:100, :]  # 取前100个token进行可视化
    data = tensor_2d.numpy()

    # 创建3D条形图
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # 创建坐标
    x_pos, y_pos = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x_pos = x_pos.flatten()
    y_pos = y_pos.flatten()
    z_pos = np.zeros_like(data.flatten())
    dx = dy = 0.08  # 条形的宽度
    dz = data.flatten()

    # 根据数值设置颜色
    colors = plt.cm.viridis((dz - dz.min()) / (dz.max() - dz.min()))

    # 绘制3D条形
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors)

    # 设置标签
    ax.set_xlabel('Channel')
    ax.set_ylabel('Token')
    ax.set_zlabel('Value')
    # ax.set_title('2D Tensor Data Distribution - 3D Bar Chart')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=dz.min(), vmax=dz.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(f'fig/{k}.pdf')
    plt.close()
    print(f'fig/{k}.png saved.')

    # break
