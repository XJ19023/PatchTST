import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch

scale = 200
acc_drop = torch.tensor([2239-13, 1*scale, 4142-9, 5*scale, 16155-7, 2*scale])
ratio_tensor = torch.tensor([1, 0.001333, 1, 0.001263, 1, 0.000862]) # Qwen2.5

# 绘图
fig, ax = plt.subplots(figsize=(2.6, 2), dpi=300, constrained_layout=True)
colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255)]
# 移动底部的spine（x轴），保持x轴在y=0处
# ax.spines['bottom'].set_position(('data', 0))
# 设置x轴刻度标签和旋转角度
# plt.yticks(np.arange(0, 110, 25))
plt.ylabel('Percentage (%)', fontsize=9)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
# ax.set_yticks(np.arange(0, 1.5, 0.25))


plt.ylim(0, 1)
# plt.xlim(-0.5, 7.5)

# tensor = torch.arange(12).reshape(4, 3)
# xticks = torch.cat((tensor[0], tensor[1]+0.3, tensor[2]+0.8, tensor[3]+1.1), dim=0)
xticks = torch.arange(6)
ax.set_xticks(xticks)
ax.set_xticklabels([f'{dut}' for dut in ['W4A4', 'W4A8'] * 3], rotation=45, fontsize=8)
ax.tick_params(axis='y', labelsize=8)

# 设置柱子的宽度
# 迭代 tensor 的第一维度，并生成堆积柱状图
# print(xticks)
gap = 0.2
bar_width = 0.4
ln1 = ax.bar(xticks, 1-ratio_tensor, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT8')
ln2 = ax.bar(xticks, ratio_tensor, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=1-ratio_tensor, zorder=3, label='INT4')

# 嵌入绘制局部放大图的坐标系
# 在子坐标系中绘制原始数据
axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.2, 0.56, 1.6, 1),
                   bbox_transform=ax.transAxes)
axins.set_zorder(5)  # 设置子坐标系的层级，使其在父坐标系之上
axins.set_facecolor("#fae6e9")
axins.bar(xticks, 1-ratio_tensor, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT8')
axins.bar(xticks, ratio_tensor, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=1-ratio_tensor, zorder=3, label='INT4')

# 调整子坐标系的显示范围
# axins.set_xlim(0, 5)
axins.set_ylim(1-0.002, 1)
axins.set_yticks([1-0.002, 1])
axins.set_yticklabels([0.8, 1], fontsize=8)
ax.add_patch(patches.Rectangle((-0.5, 0.9), 6, 0.06, edgecolor="black", linewidth=0.6, facecolor="#fae6e9", fill=True))
ax.tick_params(axis='both', which='major', bottom=False, left=False)
axins.tick_params(axis='both', which='major', bottom=False, left=False)
axins.get_xaxis().set_visible(False)

# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec='k', lw=0.6)
# plt.tight_layout()

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylim(0, 9000)
ax2.set_yticks([0, 1000, 3000, 6000, 9000])
ax2.set_yticklabels([0, '5', '3e3', '6e3', '9e3'], fontsize=8)
# ax2.scatter(xticks, loss_tensor, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)
for i in range(3):
    ln3, = ax2.plot(xticks[0+i*2:2+i*2], acc_drop[0+i*2:2+i*2], color=colors[2], marker="^", markeredgecolor='black',markersize='4',markeredgewidth=0.5, zorder=1, linewidth=0.8, label='AccDrop ')

ax.set_ylabel('Percentage (%)', fontsize=8)
ax2.set_ylabel('Acc. Drop', fontsize=8)

ax.plot([1.2, 1.8], [0.3, 0.2], color="black", linewidth=1)
lines = [ln1, ln2, ln3]
labels = [l.get_label() for l in lines]
leg = ax.legend(lines, labels, loc="upper right", bbox_to_anchor=(1, 0.5), fontsize=6, handlelength=1)
leg.set_zorder(10)   # 设置一个比其他元素更大的 zorder



plt.savefig(f'aaa.png')
plt.savefig(f'motivation.pdf')