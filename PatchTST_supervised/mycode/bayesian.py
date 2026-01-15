import torch
import torch.nn as nn
import numpy as np
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

# -----------------------------
# 1. 三层 Linear 模型
# -----------------------------
class ThreeLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x, qconfig):
        # qconfig = [q1, q2, q3], 0: INT8, 1: MXINT8
        x = self.relu(self.quant_linear(self.fc1, x, qconfig[0]))
        x = self.relu(self.quant_linear(self.fc2, x, qconfig[1]))
        x = self.quant_linear(self.fc3, x, qconfig[2])
        return x

    def quant_linear(self, layer, x, qtype):
        """
        mock 量化算子：
        INT8   -> 直接线性
        MXINT8 -> 人为注入更小误差
        """
        y = layer(x)
        if qtype == 0:  # INT8
            noise = torch.randn_like(y) * 0.02
        else:           # MXINT8
            noise = torch.randn_like(y) * 0.005
        return y + noise


# -----------------------------
# 2. 构造数据
# -----------------------------
torch.manual_seed(0)
model = ThreeLinearNet()
model.eval()

x = torch.randn(16, 128)
y_gt = torch.randn(16, 10)  # fake ground truth

# -----------------------------
# 3. 定义搜索空间
# -----------------------------
space = [
    Categorical([0, 1], name="fc1"),
    Categorical([0, 1], name="fc2"),
    Categorical([0, 1], name="fc3"),
]

# -----------------------------
# 4. 目标函数
# -----------------------------
@use_named_args(space)
def objective(fc1, fc2, fc3):
    qconfig = [fc1, fc2, fc3]

    with torch.no_grad():
        y = model(x, qconfig)

    # fake 精度损失（MSE）
    mse = torch.mean((y - y_gt) ** 2).item()

    # fake 硬件代价：MXINT8 比 INT8 更贵
    hw_cost = 0.0
    for q in qconfig:
        hw_cost += 1.0 if q == 0 else 1.3

    # 综合目标（最小化）
    loss = mse + 0.05 * hw_cost

    print(f"Try {qconfig}, mse={mse:.4f}, cost={hw_cost:.2f}, loss={loss:.4f}")
    return loss


# -----------------------------
# 5. 运行贝叶斯优化
# -----------------------------
res = gp_minimize(
    objective,
    space,
    n_calls=20,
    n_initial_points=6,
    random_state=0
)

print("\n===== 搜索完成 =====")
print("Best loss:", res.fun)
print("Best config:", res.x)
print("Config meaning: [fc1, fc2, fc3] 0=INT8, 1=MXINT8")
