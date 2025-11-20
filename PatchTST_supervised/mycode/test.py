# import torch

# # A: (2×3)
# A = torch.tensor([
#     [1., 2., 3.],
#     [4., 5., 6.]
# ])

# # B: (3×4)
# B = torch.tensor([
#     [1., 2., 3., 4.],
#     [5., 6., 7., 8.],
#     [9., 10., 11., 12.]
# ])

# # --- 方式 1: 使用广播生成逐元素乘积 ---
# # 逐元素乘法结果 shape = (2×3×4)
# mul_res = A[:, :, None] * B[None, :, :]

# # 沿着 k 维求和（k = 3），得到最终矩阵乘法结果 (2×4)
# C_manual = mul_res.sum(dim=1)

# # 对比 PyTorch matmul
# C_torch = A @ B

# print("A[i,k] * B[k,j] (shape=2, 3, 4):")
# print(mul_res)
# print("\nC:")
# print(mul_res.shape, C_manual.shape)
# print(C_manual)
# # print("\nPyTorch matmul:")
# # print(C_torch)

import torch
from mx.mx_ops import _shared_exponents


def matmul_decompose_general(A, B):
    """
    A: (..., M, K)
    B: (K, N)
    return:
        P: (..., M, K, N)  # all pairwise products
        C: (..., M, N)     # sum along K (matmul result)
    """

    # Expand A → (..., M, K, 1)
    A_exp = A.unsqueeze(-1)

    # Expand B → (1, ..., 1, K, N) 让它能广播
    # 在前面加与 A 相同数量的 batch 维
    expand_dims = A.dim() - 2  # A 的 batch 维数
    B_exp = B.view(*([1] * expand_dims), B.shape[0], B.shape[1])
    # B_exp shape = (..., K, N)

    # Step 1：pairwise 乘法 → (..., M, K, N)
    P = A_exp * B_exp


    # Step 2：对 K 维求和 → (..., M, N)
    # Y = P.sum(dim=-2)
    # 把 N 和 K 交换
    P_t = P.transpose(-1, -2)  # (..., K, N)
    print(P_t.shape)
    print(P_t)

    max_exp = _shared_exponents(P_t, dim=-1)
    print("max_exp:", max_exp)

    ones = torch.ones(P_t.shape[-1], device=P.device, dtype=P.dtype)  # shape: (N,)

    Y = torch.matmul(P_t, ones)  # (..., K)


    return Y

seed = 42
torch.manual_seed(seed)
A = torch.randn(2, 4)  # (batch1=2, batch2=3, M=4, K=5)
B = torch.randn(4, 3)       # (K=5, N=6)
# A = torch.arange(10).reshape(2, 5)  # (batch1=2, batch2=3, M=4, K=5)
# B = torch.arange(15).reshape(5, 3)       # (K=5, N=6)

Y = matmul_decompose_general(A, B)

# print("A @ B =\n", A @ B)
# print("Decomposed C =\n", Y)
