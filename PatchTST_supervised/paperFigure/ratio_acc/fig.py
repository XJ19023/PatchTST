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
start_time = time.time()
# ----------------------------------------------------------

acc_tensor = torch.tensor([10.2072,
10.1859,
10.1396,
10.1223,
10.1031,
10.1061,
10.0736,
10.0708,
10.0669,
10.0718,
10.0714,] ) # wikitext
# loss_tensor = 10.0539 - acc_tensor
# loss_tensor *= 100
print(acc_tensor.min(), acc_tensor.max())
ratio_tensor = torch.tensor([0.4827,
0.4656,
0.4232,
0.3873,
0.3756,
0.3487,
0.2895,
0.2530,
0.2476,
0.2342,
0.2122,]) # int4 ratio
ratio_tensor = 1 - ratio_tensor # int8 ratio

plot_fig = 1
if plot_fig:

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    # 为每个柱状图分配不同的颜色
    colors = ['#8ecfc9', '#f6cae5', '#bb9727', '#f6c177', '#d43f3b', '#fa7f6f']  # 颜色列表

    # 移动底部的spine（x轴），保持x轴在y=0处
    # ax.spines['bottom'].set_position(('data', 0))
    # 设置x轴刻度标签和旋转角度
    # plt.yticks(np.arange(0, 110, 25))
    # plt.ylabel('Percentage (%)')
    # ax.set_yticks(np.arange(0, 1.5, 0.25))


    plt.ylim(0, 1)
    plt.xlim(-0.8, len(acc_tensor)-0.2)

    xticks = np.arange(ratio_tensor.shape[0])
    ax.set_xticks(xticks)
    xtickslabels = torch.tensor([0.00200,
0.00180,
0.00140,
0.00110,
0.00100,
0.00080,
0.00040,
0.00010,
0.00008,
0.00004,
0.00001,])
    xtickslabels *= 10**4

    xtickslabels_t = []
    for i in xtickslabels:
        if i >= 1:
            xtickslabels_t.append(f'{i:.0f}')
        else:
            xtickslabels_t.append(f'{i:.1f}')


    ax.set_xticklabels(xtickslabels_t, rotation=0)

    # 设置柱子的宽度
    # 迭代 tensor 的第一维度，并生成堆积柱状图
    # print(xticks)
    gap = 0.2
    bar_width = 0.4
    ln1 = ax.bar(xticks, 1-ratio_tensor, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
    ln2 = ax.bar(xticks, ratio_tensor, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=1-ratio_tensor, zorder=3, label='INT8')

    # plt.hlines(y = 0.5, xmin = -0.5, xmax = 7.5, color ='r', zorder=4)


    # 在柱子之间画竖线
    # for i in range(len(duts), n, len(duts)):
    #     ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    # 只显示水平方向的网格线
    # ax.grid(True, axis='y', linestyle='--', color='gray', zorder=0)
    # Change y-axis to percentage format
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
    # ax.tick_params(axis='y', labelcolor='black')
    # # ax.set_xlabel('Columns')

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.scatter(xticks, loss_tensor, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)
    ln3, = ax2.plot(xticks, acc_tensor, color=colors[2], marker="o", markeredgecolor='black',markersize='5',markeredgewidth=0.5, zorder=4, linewidth=0.8, label='PPL ')

    ax2.plot(xticks[4], acc_tensor[4], color=colors[4], marker="*", markeredgecolor='black',markersize='10',markeredgewidth=0.5, zorder=5)
    plt.ylim(10.04, 10.21)

    ax.set_ylabel('Percentage (%)', fontsize=9, labelpad=-0.1)
    ax.set_xlabel('MSE (e-4)', fontsize=9, labelpad=2)
    ax2.set_ylabel('Perplexity (PPL)', fontsize=9, labelpad=4)
    
    # # ax.set_title('Stacked Bar Chart of Tensor with Custom Style')
    # plt.axvline(x=(xticks[5] + xticks[6])/2, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    gap = -0.26
    fontsize = 8
    # ax.text(xticks[1], gap, 'LLaMa2-7B', fontsize=fontsize, ha='center', va='center')
    # ax.text(xticks[4], gap, 'Qwen2.5-7B', fontsize=fontsize, ha='center', va='center')
    # ax.text(xticks[7], gap, 'LLaMa2-7B', fontsize=fontsize, ha='center', va='center')
    # ax.text(xticks[10], gap, 'Qwen2.5-7B', fontsize=fontsize, ha='center', va='center')
    gap = 1.08
    fontsize = 9
    # ax.text((xticks[2] + xticks[3])/2, gap, 'WikiText2', fontsize=fontsize, ha='center', va='center')
    # ax.text((xticks[8] + xticks[9])/2, gap, 'C4', fontsize=fontsize, ha='center', va='center')
    # 将图例放置在坐标轴框线外的正上方
    # plt.legend(loc='upper center', ncol=3, fontsize=9) # 控制图形和文本之间的间距
    # ax.legend(loc='upper left')
    # 合并两个轴的 legend
    lines = [ln1, ln2, ln3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", ncol=3, fontsize=7, handlelength=1.5)

    ax.tick_params(axis='both', which='major', labelsize=8, bottom=False, left=False)
    ax2.tick_params(axis='both', which='major', labelsize=8, right=False)

    plt.hlines(y = 10.15, xmin = -0.8, xmax = len(acc_tensor)-0.2, color = '#e9212c', zorder=4)
    plt.text(len(acc_tensor)-4, 10.13, '1% Loss', fontsize=9,
         bbox=dict(facecolor="white", alpha=0.8, edgecolor=colors[4], boxstyle="round,pad=0.2"))
    plt.text(len(acc_tensor)-10.5, 10.08, 'Optimal trade-off', fontsize=9, color=colors[4], 
         bbox=dict(facecolor="white", alpha=0.8, edgecolor=colors[4], boxstyle="round,pad=0.2"))


    ax.tick_params(axis='x', pad=1)  # X 轴刻度与轴线的距离
    ax.tick_params(axis='y', pad=1)
    ax2.tick_params(axis='y', pad=1)

    plt.tight_layout()
    plt.subplots_adjust(left=0.11, right=0.83, bottom=0.18, top=0.97)


    plt.savefig(f'aaa.png')
    # plt.savefig('energy.png', bbox_inches='tight')
    plt.savefig('ratio_acc.pdf')
    plt.close()


# ----------------------------------------------------------
end_time = time.time()
hour = (end_time-start_time)//360
min = (end_time-start_time)//60 - hour * 60
sec = (end_time-start_time) - min * 60
print(f'RUNING TIME: {int(hour)}h-{int(min)}m-{int(sec)}s')


