import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch


w4a8_mse = torch.tensor([
    [0.00633, 0.00355, 0.00956, 0.04170, 0.08214, 0.03137, 0.03125, 0.06620, 0.04944, 0.25129],
    [0.28481, 0.07083, 0.25910, 0.60854, 0.38643, 0.74272, 0.90071, 0.35392, 0.59873, 0.67430],
    [0.29603, 0.02461, 0.04861, 0.03437, 0.11316, 0.17400, 0.22700, 0.24992, 0.08383, 0.14047]
])

# base: 0.151886             int3,     int4,     int5,     int6,     int7,     int8
weather_96  = torch.tensor([[0.176795, 0.154636, 0.151919, 0.151455, 0.151463, 0.151363],   # block size = 8
                            [0.171750, 0.154068, 0.151873, 0.151504, 0.151502, 0.151365],   # block size = 16
                            [0.170869, 0.154171, 0.151878, 0.151503, 0.151513, 0.151357],   # block size = 32
                            [0.170359, 0.154824, 0.151959, 0.151502, 0.151504, 0.151359]])  # block size = 64

# base: 0.319332             int3,     int4,     int5,     int6,     int7,     int8
weather_720 = torch.tensor([[0.331064, 0.319961, 0.319482, 0.319530, 0.319561, 0.319629],   # block size = 8  
                            [0.328337, 0.320211, 0.319561, 0.319576, 0.319555, 0.319636],   # block size = 16
                            [0.327661, 0.320259, 0.319499, 0.319626, 0.319541, 0.319639],   # block size = 32
                            [0.327672, 0.320480, 0.319490, 0.319681, 0.319538, 0.319640]])   # block size = 64  

# weather_720 = weather_720.transpose(0, 1)

# 绘图
fig, ax = plt.subplots(figsize=(2.6, 2.2), dpi=300, constrained_layout=True)
colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255), (70/255, 120/255, 142/255)]
# 移动底部的spine（x轴），保持x轴在y=0处
# ax.spines['bottom'].set_position(('data', 0))
# 设置x轴刻度标签和旋转角度
# ax.set_ylabel('Percentage (%)', fontsize=9)
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))

xticks = torch.arange(weather_720.size(1))

sqz_label = ['8', '16', '32', '64']
# sqz_label = ['int3', 'int4', 'int5', 'int6', 'int7', 'int8']
# ax.plot(xticks, w4a8_mse[0] / w4a8_mse[0], color='#d43f3b', marker="o", markeredgecolor='black', markersize='3', markeredgewidth=0.2, zorder=1, linewidth=0.8, label='Baseline (INT8)')
for i in range(weather_720.size(0)):
    ax.plot(xticks, weather_720[i], color=colors[i], marker="^", markeredgecolor='black', markersize='3', markeredgewidth=0.2, zorder=1, linewidth=0.8, label=sqz_label[i])
    # ax.plot(xticks, qwt_mse[i] / w4a8_mse[i], color=colors[i], marker="s", markeredgecolor='black', markersize='3', markeredgewidth=0.2, zorder=1, linewidth=0.8, label=sqzComp_label[i])
plt.legend(loc='upper center', bbox_to_anchor=(0.66, 1), ncol=2, fontsize=7, handlelength=1.2, columnspacing=1, handletextpad=0.3)

ax.set_xticks(xticks)
ax.set_xticklabels(['INT3', 'INT4', 'INT5', 'INT6', 'INT7', 'INT8'], rotation=0, fontsize=8)
ax.set_ylabel('MSE', fontsize=9, labelpad=1)
ax.set_xlabel('Mantissa Width', fontsize=9, labelpad=1)

plt.tick_params(axis='both', labelsize=8)

# ax.set_ylim(0, 6)
# plt.yticks(np.arange(0, 7, 2))

plt.tight_layout
plt.savefig(f'aaa.png')
plt.savefig(f'mse.pdf')
# plt.savefig(f'mse.svg')