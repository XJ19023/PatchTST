import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch



weather_720 = torch.tensor([[0.31963634490966796875, 0.31963598728179931641, 0.31963598728179931641, 0.31963598728179931641,]])

weather_720 = torch.tensor([[0.00004469671694096178, 0.00001118301679525757, 0.00000279621463050717, 0.00000069820362114115, 0.00000017460246226619, 0.00000004365545436258, 0.00000001098045210313, 0.00000000283564727255, 0.00000000077096073969, 0.00000000024781021679, 0.00000000013447286340, 0.00000000011208686063, 0.00000000010646304671, 0.00000000010454152821, 0.00000000010385042826, 0.00000000010361903696, 0.00000000010355375585, 0.00000000010353688046, 0.00000000010352559088, 0.00000000010352375207, 0.00000000010352434882, 0.00000000010352411289, 0.00000000010352447372, 0.00000000010352443902, 0.00000000010352441820, 0.00000000010352440433, 0.00000000010352440433]]) # random

# 绘图
fig, ax = plt.subplots(figsize=(6, 4), dpi=300, constrained_layout=True)
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
# ax.set_xticklabels(['15', '20', '30', '40'], rotation=0, fontsize=8)
ax.set_xticklabels(np.arange(14,41), rotation=0, fontsize=8)
ax.set_ylabel('MSE', fontsize=9, labelpad=1)
ax.set_xlabel('Accumulate Width', fontsize=9, labelpad=1)

plt.tick_params(axis='both', labelsize=8)

# ax.set_ylim(0, 6)
# plt.yticks(np.arange(0, 7, 2))

plt.tight_layout
plt.savefig(f'aaa.png')
# plt.savefig(f'mse.pdf')
# plt.savefig(f'mse.svg')