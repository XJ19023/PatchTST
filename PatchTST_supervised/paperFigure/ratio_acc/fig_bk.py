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

acc_tensor = torch.tensor([5.6788, 5.6947, 5.6944, 7.3952, 7.4024, 7.4724, # wikitext
                           7.222, 7.2125, 7.2384, 10.9771, 10.9851, 11.0303] ) # c4
ratio_tensor = torch.tensor([0.2649, 0.3515, 0.4667, 0.1295, 0.222, 0.3577,   # int8 ratio
                             0.2696, 0.3519, 0.4603, 0.1220, 0.2062, 0.3320])

plot_fig = 1
if plot_fig:

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)
    # 为每个柱状图分配不同的颜色
    colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255)]
    # 移动底部的spine（x轴），保持x轴在y=0处
    # ax.spines['bottom'].set_position(('data', 0))
    # 设置x轴刻度标签和旋转角度
    # plt.yticks(np.arange(0, 110, 25))
    # plt.ylabel('Percentage (%)')
    # ax.set_yticks(np.arange(0, 1.5, 0.25))


    plt.ylim(0, 1)
    # plt.xlim(-0.5, 7.5)

    tensor = torch.arange(12).reshape(4, 3)
    xticks = torch.cat((tensor[0], tensor[1]+0.3, tensor[2]+0.8, tensor[3]+1.1), dim=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{dut}' for dut in [32, 64, 128] * 4], rotation=0, fontsize=9)

    # 设置柱子的宽度
    bar_width = 0.4
    # 迭代 tensor 的第一维度，并生成堆积柱状图
    # print(xticks)
    gap = 0.2
    ax.bar(xticks, ratio_tensor, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
    ax.bar(xticks, 1-ratio_tensor, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=ratio_tensor, zorder=3, label='INT8')

    # plt.hlines(y = 0.5, xmin = -0.5, xmax = 7.5, color ='r', zorder=4)


    # 在柱子之间画竖线
    # for i in range(len(duts), n, len(duts)):
    #     ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1, alpha=0.8)
    # 只显示水平方向的网格线
    ax.grid(True, axis='y', linestyle='--', color='gray', zorder=0)
    # Change y-axis to percentage format
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
    # ax.tick_params(axis='y', labelcolor='black')
    # # ax.set_xlabel('Columns')

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.scatter(xticks, acc_tensor, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)

    # ax.set_ylabel('Normalized Energy (%)', labelpad=-3)
    # # ax.set_title('Stacked Bar Chart of Tensor with Custom Style')
    plt.tick_params(bottom=False, left=False)
    # 将图例放置在坐标轴框线外的正上方
    # plt.legend(loc='upper center', ncol=3, fontsize=9) # 控制图形和文本之间的间距
    # ax.legend(bbox_to_anchor=(0.7, 1.2), ncol=4)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.895)


    plt.savefig(f'aaa.png')
    # plt.savefig('energy.png', bbox_inches='tight')
    # plt.savefig('energy.pdf')
    plt.close()


# ----------------------------------------------------------
end_time = time.time()
hour = (end_time-start_time)//360
min = (end_time-start_time)//60 - hour * 60
sec = (end_time-start_time) - min * 60
print(f'RUNING TIME: {int(hour)}h-{int(min)}m-{int(sec)}s')


