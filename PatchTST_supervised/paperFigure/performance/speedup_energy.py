import os
import random
import numpy as np
import torch
import math
from matplotlib import pyplot as plt
import torch.nn.functional as F
import logging
import time
from matplotlib.ticker import PercentFormatter

import time

import pandas as pd
import torch


def main():
    start_time = time.time()
    # ----------------------------------------------------------
    models = [
                'Mistral-7B',
                'LLaMa-7B',
                'LLaMa3-8B',
                'Qwen2.5-7B',
                'Qwen3-8B',
                'LLaMa2-13B',
                'Qwen3-14B',
                'Geomean'
                ]
    hw_arch = ['MANT', 'OliVe', 'ANT', 'SPARK', 'WARD']
    labels = ['DRAM', 'Buffer', 'Core', 'En/Decode']
    colors = ['#8ecfc9', '#f6c177', '#f6cae5', '#EEF8B4', '#fa7f6f', '#c39797']  # 为每个硬件架构分配不同的颜色
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), dpi=300, gridspec_kw={'height_ratios': [1, 1.15]}) # 上下图比例1：1

    plt.subplot(2, 1, 1)
    # 读 CSV
    df = pd.read_csv("my_res.csv", header=None)
    # print(df)
    # 去掉第一行（表头），去掉第一列（Static/Dram/...），去掉最后一列空的逗号
    values = df.iloc[2:3, 1:-1].astype(float).values
    tensor_org = torch.tensor(values, dtype=torch.float32)
    new_row = tensor_org[0:2].sum(dim=0)   # 把前两行相加
    tensor = torch.vstack([new_row, tensor_org[2:]])
    tensor.squeeze_(0)
    tensor = 1 / tensor
    # print(tensor)
    # 创建柱状图
    # 为每个柱状图分配不同的颜色
    plt.ylabel('Speedup')
    # ax.set_yticks(np.arange(0, 1.5, 0.25))
    # plt.ylim(0, 1)
    plt.xlim(-1, 43.5)
    # 设置柱子的宽度
    bar_width = 0.7
    xticks = []
    xticks_final = torch.tensor([])
    for idx, item in enumerate(np.arange(0, len(hw_arch) * len(models), len(hw_arch))):
        xticks.append(item + idx*0.5)
    xticks = torch.tensor(xticks)
    for i in range(len(hw_arch)):
        xticks_final = torch.cat((xticks_final, xticks+i), dim=0)
    xticks, _ = xticks_final.sort()
    plt.xticks(xticks, [f'{dut}' for dut in hw_arch * len(models)], rotation=90, fontsize=9)
    gap = 0.2
    base = np.arange(0, len(hw_arch) * len(models), len(hw_arch))
    for i in range(len(hw_arch)):
        idx = i + base
        plt.bar(xticks[idx], tensor[idx], width=bar_width, edgecolor='black', color=colors[i], linewidth=0.5, label=hw_arch[i], zorder=3)
    for i in range(len(hw_arch)-1, len(hw_arch) * len(models), len(hw_arch)):
        plt.text(xticks[i]+0.04, tensor[i] - 0.8, f'{tensor[i]:.2f}', ha='center', va='bottom', rotation=90)
    # 在柱子之间画竖线
    for i in range(len(models)-1):
        plt.axvline(x=(xticks[len(hw_arch)-1+i*len(hw_arch)] + xticks[len(hw_arch)+i*len(hw_arch)])/2, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    # 只显示水平方向的网格线
    plt.grid(True, axis='y', linestyle='--', color='gray', zorder=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), ncol=len(hw_arch), fontsize=9) 
    for i in range(len(models)):
        plt.text((xticks[1+i*len(hw_arch)] + xticks[2+i*len(hw_arch)])/2, -1.5, models[i], fontsize=10, ha='center', va='center')
    
    plt.subplot(2, 1, 2)
    plt.xticks(xticks, [f'{dut}' for dut in hw_arch * len(models)], rotation=90, fontsize=9)
    # 读 CSV（假设文件叫 energy.csv）
    df = pd.read_csv("my_res.csv", header=None)

    # 去掉第一行（表头），去掉第一列（Static/Dram/...），去掉最后一列空的逗号
    values = df.iloc[5:9, 1:-1].astype(float).values

    tensor = torch.tensor(values, dtype=torch.float32)
    # new_row = tensor_org[0:2].sum(dim=0)   # 把前两行相加
    # tensor = torch.vstack([new_row, tensor_org[2:]])
    # print(tensor)

    plt.ylabel('Normalized Energy')
    # ax.set_yticks(np.arange(0, 1.5, 0.25))


    plt.ylim(0, 1)
    plt.xlim(-1, 43.5)


    # 设置柱子的宽度
    # 迭代 tensor 的第一维度，并生成堆积柱状图
    gap = 0.2

    for i in range(tensor.shape[0]):
        data = tensor[i]
        bottom = torch.zeros_like(data) if i == 0 else bottom + tensor[i-1]
        data_label = data + bottom
        bar = plt.bar(xticks, data, width=bar_width, edgecolor='black', color=colors[i], linewidth=0.5, bottom=bottom, label=labels[i], zorder=3)
        # 添加数据标签
    # plt.bar_label(bar, label_type='edge')
    # print(bar)
    # for i, val in enumerate(y):
        # print(xticks, data_label)
    for i in range(len(hw_arch)-1, len(xticks), len(hw_arch)):
        plt.text(xticks[i], data_label[i] + 0.03, f'{data_label[i]:.2f}', ha='center', va='bottom', rotation=90)

    # plt.hlines(y = 0.5, xmin = -0.5, xmax = 7.5, color ='r', zorder=4)


    # 在柱子之间画竖线
    for i in range(len(models)-1):
        plt.axvline(x=(xticks[len(hw_arch)-1+i*len(hw_arch)] + xticks[len(hw_arch)+i*len(hw_arch)])/2, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    # 只显示水平方向的网格线
    plt.grid(True, axis='y', linestyle='--', color='gray', zorder=0)

    # 将图例放置在坐标轴框线外的正上方
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=len(hw_arch), fontsize=9) # 控制图形和文本之间的间距
    # ax.legend(bbox_to_anchor=(0.7, 1.2), ncol=4)
    for i in range(len(models)):
        plt.text((xticks[1+i*len(hw_arch)] + xticks[2+i*len(hw_arch)])/2, -0.46, models[i], fontsize=10, ha='center', va='center')
    
    plt.subplots_adjust(hspace=0.7) 

    plt.tight_layout()
    plt.savefig(f'aaa.png')
    plt.savefig(f'speedup_energy.pdf')
    plt.close()
    # ----------------------------------------------------------
    end_time = time.time()
    hour = (end_time-start_time)//360
    min = (end_time-start_time)//60 - hour * 60
    sec = (end_time-start_time) - min * 60
    print(f'RUNING TIME: {int(hour)}h-{int(min)}m-{int(sec)}s')

if __name__ == '__main__':
    main()
