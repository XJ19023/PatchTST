import torch
import torch.nn as nn

# 先定义你的Transpose类（确保和原代码一致）
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

# 核心融合函数
def fuse_bn_transpose_with_linear(norm_attn, linear):
    """
    融合包含Transpose+BN1d的norm_attn模块和后续的Linear层
    :param norm_attn: nn.Sequential(Transpose, BN1d, Transpose)
    :param linear: 后续的nn.Linear层
    :return: 融合后的nn.Linear层
    """
    # 1. 提取BN层和参数（跳过Transpose层）
    bn = None
    for module in norm_attn:
        if isinstance(module, nn.BatchNorm1d):
            bn = module
            break
    assert bn is not None, "norm_attn中未找到BatchNorm1d层"
    
    # 2. 计算BN的融合缩放/平移因子（推理阶段用running_mean/running_var）
    eps = bn.eps
    running_mean = bn.running_mean  # [D]
    running_var = bn.running_var    # [D]
    gamma = bn.weight              # [D]
    beta = bn.bias                 # [D]
    
    # 计算scale和shift
    std = torch.sqrt(running_var + eps)
    scale = gamma / std            # [D]
    shift = beta - running_mean * scale  # [D]
    
    # 3. 提取Linear层参数
    W = linear.weight  # [H, D]
    b = linear.bias    # [H] (可能为None)
    
    # 4. 计算融合后的权重和偏置
    # 融合权重：W_fused = W * scale.unsqueeze(0) → [H,D] * [1,D] = [H,D]
    W_fused = W * scale.unsqueeze(0)
    
    # 融合偏置
    if b is not None:
        # shift @ W.T → [D] @ [D,H] = [H]，再加原偏置b
        b_fused = torch.matmul(shift, W.t()) + b
    else:
        b_fused = torch.matmul(shift, W.t())
    
    # 5. 创建融合后的Linear层
    fused_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None
    )
    # 赋值融合后的参数（关闭梯度，因为是推理阶段融合）
    with torch.no_grad():
        fused_linear.weight.copy_(W_fused)
        if fused_linear.bias is not None:
            fused_linear.bias.copy_(b_fused)
    
    return fused_linear

# ------------------- 用法示例 -------------------
# 1. 模拟你的模型模块
class ModelBlock(nn.Module):
    def __init__(self, d_model=128, d_hidden=256):
        super().__init__()
        self.norm_attn = nn.Sequential(
            Transpose(1,2),
            nn.BatchNorm1d(d_model),
            Transpose(1,2)
        )
        self.linear = nn.Linear(d_model, d_hidden, bias=True)
    
    def forward(self, x):
        x = self.norm_attn(x)
        x = self.linear(x)
        return x

# 2. 初始化模型并融合
model = ModelBlock(d_model=128, d_hidden=256)
# 切换到eval模式（BN用running_mean/running_var，必须！）
model.eval()

# 融合norm_attn和linear
fused_linear = fuse_bn_transpose_with_linear(model.norm_attn, model.linear)

# 3. 替换模型中的模块（移除norm_attn，只用融合后的linear）
model.fused_linear = fused_linear
# 重写forward（跳过原norm_attn和linear，直接用融合层）
def new_forward(self, x):
    return self.fused_linear(x)
ModelBlock.forward = new_forward

# 4. 验证融合正确性（输入示例）
x = torch.randn(2, 32, 128)  # [B=2, L=32, D=128]
with torch.no_grad():
    # 融合前输出
    model.eval()
    original_out = model.norm_attn(x)
    original_out = model.linear(original_out)
    # 融合后输出
    fused_out = model.fused_linear(x)
    # 检查误差（应小于1e-5）
    max_diff = torch.abs(original_out - fused_out).max()
    print(f"max error: {max_diff.item():.8f}")