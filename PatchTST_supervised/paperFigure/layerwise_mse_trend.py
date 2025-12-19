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

base = torch.tensor([0.1492039 , 0.14929476, 0.1486522 , 0.15035428,])
tensor_without_smooth = torch.tensor([[0.00000591, 0.00000642, 0.00000592, 0.00000907, 0.00000884, 0.00000735, 0.00000728, 0.00000878, 0.00000798, 0.00002157, 0.00000806, 0.00000563, 0.00000777, 0.00000775, 0.00000778, 0.00001643, 0.00000785, 0.00000178], [0.00193582, 0.00213990, 0.00190547, 0.00254001, 0.00290182, 0.00348759, 0.00285369, 0.00268799, 0.00298785, 0.00665081, 0.00251667, 0.00290873, 0.00266948, 0.00224976, 0.00276636, 0.00528278, 0.00249152, 0.00098553], [0.00001576, 0.00001460, 0.00001704, 0.00000722, 0.00002994, 0.00001606, 0.00002572, 0.00002119, 0.00004617, 0.00004100, 0.00002968, 0.00001963, 0.00003188, 0.00002707, 0.00003877, 0.00002264, 0.00002064, 0.00000790], [0.00241056, 0.00246963, 0.00238602, 0.00109708, 0.00393134, 0.00244159, 0.00387973, 0.00357105, 0.00404705, 0.00487692, 0.00329267, 0.00251582, 0.00331673, 0.00336859, 0.00320230, 0.00465199, 0.00354514, 0.00071348]])

tensor_with_smooth = torch.tensor([[0.00000713, 0.00000778, 0.00000724, 0.00000374, 0.00001525, 0.00000632, 0.00001447, 0.00001630, 0.00001437, 0.00001632, 0.00001377, 0.00000559, 0.00001350, 0.00001063, 0.00001285, 0.00001004, 0.00001057, 0.00000139,], [0.00234139, 0.00233632, 0.00229134, 0.00136113, 0.00470232, 0.00247362, 0.00467679, 0.00474068, 0.00525954, 0.00549008, 0.00483330, 0.00170625, 0.00403895, 0.00357048, 0.00402627, 0.00331956, 0.00361052, 0.00048130, ], [0.00001635, 0.00001526, 0.00001765, 0.00000592, 0.00003873, 0.00001686, 0.00003483, 0.00002603, 0.00004884, 0.00003406, 0.00003493, 0.00001711, 0.00003531, 0.00003152, 0.00004539, 0.00001584, 0.00002393, 0.00000674,], [0.00243374, 0.00246261, 0.00267521, 0.00083130, 0.00581900, 0.00207367, 0.00589349, 0.00547720, 0.00562741, 0.00480848, 0.00442205, 0.00177450, 0.00419871, 0.00506600, 0.00504599, 0.00332830, 0.00453604, 0.00042409,]])

tensor = tensor_without_smooth

colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255), (70/255, 120/255, 142/255)]
width = 0.6

xticks = np.arange(0, tensor.size(1))
xticks = torch.tensor(xticks)


# fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
# for i in range(tensor.size(0)):
#     plt.subplot(2, 2, i + 1)
#     plt.plot(xticks, tensor[i], linewidth=0.5, zorder=3)
#     plt.tick_params(bottom=False, left=False)
#     plt.grid(True, axis='y', linestyle='--', color='lightgray', zorder=0)

fig, ax = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
plt.subplot(2, 2, 1)
plt.plot(xticks, tensor[0], linewidth=0.5, zorder=3, label='int8')
plt.plot(xticks, tensor[2], linewidth=0.5, zorder=3, label='BFP8')
plt.legend(loc='upper right')
plt.subplot(2, 2, 2)
plt.plot(xticks, tensor[1], linewidth=0.5, zorder=3, label='int4')
plt.plot(xticks, tensor[3], linewidth=0.5, zorder=3, label='BFP4')
plt.legend(loc='upper right')

plt.subplot(2, 2, 3)
plt.plot(xticks, tensor[0]*10**2, linewidth=0.5, zorder=3, label='int8')
plt.plot(xticks, tensor[1], linewidth=0.5, zorder=3, label='int4')
plt.legend(loc='upper right')
plt.subplot(2, 2, 4)
plt.plot(xticks, tensor[2]*10**1.5, linewidth=0.5, zorder=3, label='BFP8')
plt.plot(xticks, tensor[3], linewidth=0.5, zorder=3, label='BFP4')
plt.legend(loc='upper right')

fig.supxlabel("without smooth")


fig.tight_layout()
plt.savefig(f'aaa.png', dpi=300)
plt.close()
