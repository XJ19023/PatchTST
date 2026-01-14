import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter
import torch

torch.set_printoptions(
    precision=8,
    sci_mode=False,   # 关闭科学计数法
    linewidth=120,    # 一行显示更多
)

int8 = torch.tensor([0.00000578, 0.00000582, 0.00000527, 0.00000299, 0.00000990, 0.00000691, 0.00000931, 0.00000948, 0.00001090, 0.00003354, 0.00000953, 0.00000529, 0.00000803, 0.00000870, 0.00000867, 0.00002020, 0.00001136, 0.00000251,])

bfp8_16 = torch.tensor([0.00001570, 0.00001558, 0.00001716, 0.00000374, 0.00003123, 0.00001554, 0.00003644, 0.00003102, 0.00005370, 0.00004216, 0.00003276, 0.00001837, 0.00003585, 0.00003278, 0.00005291, 0.00002681, 0.00002587, 0.00000949,])

bfp8_128 = torch.tensor([0.00001839, 0.00001822, 0.00001968, 0.00000797, 0.00003484, 0.00002195, 0.00003997, 0.00003430, 0.00006422, 0.00008763, 0.00003562, 0.00002240, 0.00003854, 0.00003501, 0.00005541, 0.00004844, 0.00002821, 0.00001116,])

# print(tensor.min())





colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255), (70/255, 120/255, 142/255)]
width = 0.6
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

xticks = np.arange(0, int8.size(0))
xticks = torch.tensor(xticks)


plt.plot(xticks, int8, linewidth=0.8, zorder=3, label='INT8')
plt.plot(xticks, bfp8_16, linewidth=0.8, zorder=3, label='BFP8 (Block=16)')
plt.plot(xticks, bfp8_128, linewidth=0.8, zorder=3, label='BFP8 (Block=128)')
# plt.xlim(-0.5, 4.5)
plt.xticks(xticks, labels=[f'{(dut)}' for dut in xticks], fontsize=9)
plt.tick_params(bottom=False, left=False)
plt.xlabel('Layer ID')
plt.ylabel('MSE')
plt.grid(True, axis='y', linestyle='--', color='lightgray', zorder=0)
plt.legend()
plt.tight_layout()
plt.savefig(f'aaa.png')
# plt.savefig(f'alpha.pdf')
plt.close()
