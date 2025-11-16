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

acc_tensor = torch.tensor([5.6788, 5.6947, 5.6944, 7.3952, 7.4024, 7.4724, # wikitext
                           7.222, 7.2125, 7.2384, 10.9771, 10.9851, 11.0303] ) # c4
ratio_tensor = torch.tensor([0.2649, 0.3515, 0.4667, 0.1295, 0.222, 0.3577,   # int8 ratio
                             0.2696, 0.3519, 0.4603, 0.1220, 0.2062, 0.3320])
colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255)]
fig, axes = plt.subplots(1, 4, figsize=(6, 2), sharey=True, dpi=300) # 上下图比例1：1

ax1 = plt.subplot(1, 4, 1)
ax1.tick_params(bottom=False, left=False)
plt.ylim(0, 1)
plt.yticks([0, 0.25, 0.50, 0.75, 1], fontsize=9)
bar_width = 0.4
xticks = [0, 1, 2]
data = ratio_tensor[0:3]
ax1.bar(xticks, data, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
ax1.bar(xticks, 1-data, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=data, zorder=3, label='INT8')
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
plt.tick_params(bottom=False, left=False, right=False)
ax1.set_xticks(xticks)
ax1.set_xticklabels([f'{dut}' for dut in [32, 64, 128]], rotation=0, fontsize=9)

ax12 = ax1.twinx()
data = acc_tensor[0:3]
ax12.scatter(xticks, data, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)
plt.ylim(5.6, 5.7)
plt.yticks([5.6, 5.7], fontsize=8)
plt.tick_params(bottom=False, left=False, right=False)


ax2 = plt.subplot(1, 4, 2)
data = ratio_tensor[3:6]
ax2.bar(xticks, data, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
ax2.bar(xticks, 1-data, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=data, zorder=3, label='INT8')
plt.tick_params(bottom=False, left=False, right=False)
ax2.set_xticks(xticks)
ax2.set_xticklabels([f'{dut}' for dut in [32, 64, 128]], rotation=0, fontsize=9)

ax22 = ax2.twinx()
data = acc_tensor[3:6]
ax22.scatter(xticks, data, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)
plt.ylim(7.3, 7.5)
plt.yticks([7.3, 7.5], fontsize=8)
plt.tick_params(bottom=False, left=False, right=False)


ax3 = plt.subplot(1, 4, 3)
data = ratio_tensor[6:9]
ax3.bar(xticks, data, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
ax3.bar(xticks, 1-data, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=data, zorder=3, label='INT8')
plt.tick_params(bottom=False, left=False, right=False)
ax3.set_xticks(xticks)
ax3.set_xticklabels([f'{dut}' for dut in [32, 64, 128]], rotation=0, fontsize=9)

ax32 = ax3.twinx()
data = acc_tensor[6:9]
ax32.scatter(xticks, data, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)
plt.ylim(7.2, 7.3)
plt.yticks([7.2, 7.3], fontsize=8)
plt.tick_params(bottom=False, left=False, right=False)


ax4 = plt.subplot(1, 4, 4)
data = ratio_tensor[9:12]
ax4.bar(xticks, data, width=bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
ax4.bar(xticks, 1-data, width=bar_width, edgecolor='black', color=colors[1], linewidth=0.5, bottom=data, zorder=3, label='INT8')
plt.tick_params(bottom=False, left=False, right=False)
ax4.set_xticks(xticks)
ax4.set_xticklabels([f'{dut}' for dut in [32, 64, 128]], rotation=0, fontsize=9)

ax42 = ax4.twinx()
data = acc_tensor[9:12]
ax42.scatter(xticks, data, color=colors[2], edgecolor='black', marker="^", label='Accuracy', zorder=5)
plt.ylim(10.9, 11.1)
plt.yticks([10.9, 11.1], fontsize=8)
ax42.tick_params(bottom=False, right=False)


plt.tight_layout()
# plt.subplots_adjust(left=0.05, right=0.995, bottom=0.01, top=0.895)


plt.savefig(f'aaa.png')
# plt.savefig('energy.png', bbox_inches='tight')
plt.savefig('ratio_acc.pdf')
plt.close()