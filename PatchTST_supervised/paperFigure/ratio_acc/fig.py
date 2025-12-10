from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch

colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255), (70/255, 120/255, 142/255)]
width = 0.6
fig, ax = plt.subplots(2, 2, figsize=(10, 4), dpi=300)
quant_meths = ['mx-int8', 'mx-int4', 'int8', 'int4']
tensor = torch.tensor([[0.31967518, 0.31964812, 0.31963459, 0.3196187 , 0.3196753 ,],
[0.3192293 , 0.31933603, 0.31951982, 0.31922892, 0.31941029,],
[0.31936687, 0.3193545 , 0.31934151, 0.31935915, 0.31935507,],
[0.3207534 , 0.32031778, 0.32010975, 0.31975532, 0.32011974,]])
base = [0.31963599, 0.31928575, 0.31935588, 0.32027951]
ylims = [(0.3196, 0.31968), (0.3192, 0.31955), (0.31932, 0.31937), (0.3197, 0.3208)]
xticks = np.arange(0, tensor.size(1))
xticks = torch.tensor(xticks)

for i in range(4):
    plt.subplot(2, 2, i+1)
    data = tensor[i]

    plt.bar(xticks, data, width, edgecolor='black', linewidth=0.5, zorder=3)

    plt.ylim(ylims[i])
    plt.hlines(y=base[i], xmin=-0.5, xmax=4.5, color='r', zorder=5)

    plt.xlim(-0.5, 4.5)
    plt.xticks(xticks, labels=[f'{dut}' for dut in ['qkv', 'to_out', 'ff0', 'ff3', 'all']], fontsize=9)
    plt.tick_params(bottom=False, left=False)
    plt.xlabel('Smooth Module')
    plt.ylabel('MSE')
    plt.grid(True, axis='y', linestyle='--', color='lightgray', zorder=0)

plt.tight_layout()
plt.savefig(f'aaa.png')
plt.savefig(f'smooth_module.pdf')
plt.close()