
from safetensors.torch import load_file
state_dict = load_file('../../log/Qwen3-8B/delta_int5.safetensors')
from delta_key import delta_key

tensor = []
for idx, k in enumerate(delta_key):
    if idx <= 100:
        tensor.append(state_dict[k].item())
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
import torch
from safetensors.torch import load_file



import time
start_time = time.time()
# ----------------------------------------------------------

# 创建柱状图
fig, ax = plt.subplots(figsize=(5, 1.8), dpi=300)
# 为每个柱状图分配不同的颜色
colors = ['#8983BF', '#8ecfc9', '#bb9727', '#54b345', '#32b897']
# 移动底部的spine（x轴），保持x轴在y=0处
# ax.spines['bottom'].set_position(('data', 0))
# 设置x轴刻度标签和旋转角度
ax.set_xticks(np.arange(len(tensor)))

plt.xticks(np.arange(0, len(tensor), 25), rotation=0, fontsize=9)
# plt.yticks(np.arange(0, 110, 25))
# plt.ylabel('Percentage (%)')
# ax.set_yticks(np.arange(0, 1.5, 0.25))


# plt.ylim(0, 100)
plt.xlim(-2, 102)


# 设置柱子的宽度
bar_width = 0.3
# 迭代 tensor 的第一维度，并生成堆积柱状图
gap = 0.2
ax.bar(np.arange(len(tensor)), tensor, width=0.6, color=colors[1], edgecolor='#9e9e9e', linewidth=0.05)

# plt.hlines(y = 0.5, xmin = -0.5, xmax = 7.5, color ='r', zorder=4)


# 在柱子之间画竖线
# for i in range(len(duts), n, len(duts)):
#     ax.axvline(x=i - 0.5, color='grey', linestyle='--', linewidth=1, alpha=0.8)
# 只显示水平方向的网格线
# ax.grid(True, axis='y', linestyle='--', color='gray', zorder=0)
# Change y-axis to percentage format
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
# ax.tick_params(axis='y', labelcolor='black')
# # ax.set_xlabel('Columns')

# ax.set_ylabel('Normalized Energy (%)', labelpad=-3)
# # ax.set_title('Stacked Bar Chart of Tensor with Custom Style')
plt.tick_params(bottom=False, left=False)
# 将图例放置在坐标轴框线外的正上方
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, fontsize=8) # 控制图形和文本之间的间距
# ax.legend(bbox_to_anchor=(0.7, 1.2), ncol=4)
ax.set_ylabel(r'Rounding Error ($\delta$)', fontsize=9, labelpad=1)
ax.set_xlabel('First 100 Layer', fontsize=9, labelpad=2)
ax.tick_params(axis='x', pad=1)  # X 轴刻度与轴线的距离
ax.tick_params(axis='y', pad=1)
plt.tight_layout()
plt.subplots_adjust(left=0.085, right=0.987, bottom=0.21, top=0.96)


plt.savefig(f'aaa.png')
# plt.savefig('energy.png', bbox_inches='tight')
plt.savefig('delta.pdf')
plt.close()


# ----------------------------------------------------------
end_time = time.time()
hour = (end_time-start_time)//360
min = (end_time-start_time)//60 - hour * 60
sec = (end_time-start_time) - min * 60
print(f'RUNING TIME: {int(hour)}h-{int(min)}m-{int(sec)}s')


