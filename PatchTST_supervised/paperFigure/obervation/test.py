import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import torch

torch.manual_seed(0)
tensor = torch.randint(0, 100, (4, 4), dtype=torch.int8)
# print(tensor)
# exit()

# 定义分组条件
def classify(val):
    if val < 16:
        return 0  # 小于16
    elif val < 64:
        return 1  # [16, 64)
    else:
        return 2  # 大于64

classified = np.vectorize(classify)(tensor)

# 定义颜色映射
cmap = mcolors.ListedColormap([(120/255, 183/255, 201/255), (229/255, 139/255, 123/255), (70/255, 120/255, 142/255)])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# 绘制
plt.figure(figsize=(5, 5))
plt.imshow(classified, cmap=cmap, norm=norm)

# # 在方格上显示原始数值
for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
        plt.text(j, i, str(tensor[i, j].item()), ha="center", va="center", color="black")

# plt.xticks(range(tensor.shape[1]))
# plt.yticks(range(tensor.shape[0]))
plt.grid(which="both", color="gray", linewidth=0.5, linestyle="--")
# plt.title("INT8 Tensor Visualization")
plt.savefig('aaa.png')
