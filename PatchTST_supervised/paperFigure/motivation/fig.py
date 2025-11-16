import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch

# 绘图
fig, ax = plt.subplots(figsize=(5, 2), dpi=300)
colors = ['#8ecfc9', '#f6c177', '#f6cae5', '#fa7f6f']  # 颜色列表
# 移动底部的spine（x轴），保持x轴在y=0处
# ax.spines['bottom'].set_position(('data', 0))
# 设置x轴刻度标签和旋转角度
# plt.yticks(np.arange(0, 110, 25))


ratio_tensor = torch.tensor([[0.0948, 0.2558, 0.3301, 0.0654],  # int4, int5, int6, int8
                            [0.0754, 0.2181, 0.3384, 0.1146], 
                            [0.1758, 0.2689, 0.3387, 0.0463], 
                            [0.0528, 0.1818, 0.3534, 0.0996], 
                            [0.1497, 0.3062, 0.3549, 0.0411], 
                            [0.1044, 0.2597, 0.3199, 0.1072], 
                            [0.1799, 0.3114, 0.3319, 0.0356], 
                            ]) # Qwen2.5
ratio_tensor = ratio_tensor.T # int4, int6, int8
# print(ratio_tensor)
# exit()

xticks = torch.arange(ratio_tensor.shape[1])
bar_width = 0.4
ax.barh(xticks, ratio_tensor[0], bar_width, edgecolor='black', color=colors[0], linewidth=0.5, zorder=3, label='INT4')
ax.barh(xticks, ratio_tensor[1]+ratio_tensor[2], bar_width, edgecolor='black', color=colors[1], linewidth=0.5, zorder=3, label='INT5/6', hatch='///', left=ratio_tensor[0])
ax.barh(xticks, 1-(ratio_tensor[0]+ratio_tensor[1]+ratio_tensor[2]), bar_width, edgecolor='black', color=colors[2], linewidth=0.5, left=ratio_tensor[0]+ratio_tensor[1]+ratio_tensor[2], zorder=3, label='INT7/8')

ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0, symbol=''))
ax.invert_yaxis() 
# ax.set_yticklabels([])  # 删掉文字，但刻度线还在
ax.set_yticks(xticks)
ax.set_yticklabels(['Mistral-7B', 'LLaMa2-7B', 'LLaMa3-8B', 'Qwen2.5-7B', 'Qwen3-8B', 'LLaMa2-13B', 'Qwen3-14B'], rotation=0, fontsize=9)
plt.xlabel('Percentage (%)', fontsize=9)
# 或者
ax.tick_params(axis='both', which='major', bottom=False, left=False)
ax.legend(loc='upper right', ncol=1, fontsize=8)


plt.tight_layout()
fig.subplots_adjust(left=0.18, right=0.96, top=0.95, bottom=0.23)
plt.savefig(f'aaa.png')
plt.savefig(f'ratio.pdf')