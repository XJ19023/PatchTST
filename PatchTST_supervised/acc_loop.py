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
from mx.mx_ops import _shared_exponents, _quantize_elemwise_core
def float32_to_binary(x: torch.Tensor):
    # 确保是 float32 类型
    x = x.to(torch.float32)
    # 重新解释底层比特为 uint32
    bits = x.view(torch.uint32).item()
    # 转成 32 位二进制字符串
    return format(bits, '032b')

def matmul_decompose_general(A, B, acc_bits):
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
    Pt = P.transpose(-1, -2)  # (..., K, N)
    # print(Pt.shape)
    max_exp = _shared_exponents(Pt, axes=[-1])
    # print('11', torch.log2(torch.abs(Pt)+1e-8).floor())
    # print(Pt)
    # print(max_exp)

    # file = open('debug.txt', 'w')
    # print('>>> org', file=file)
    # for i, item in enumerate(Pt.reshape(-1)):
    #     binary = float32_to_binary(item)
    #     exp = binary[1:9]
    #     exp = int(exp, 2)-127
    #     man = binary[9:]
    #     shift = '0' * (int(max_exp - exp))

    #     print(f"{exp:>4} {shift}1{man}", file=file)
    #     # print(f'{exp:>4} {0:010}{man}', file=file)
    #     if (i+1) % Pt.shape[-1] == 0:
    #         print(file=file)

    Pt = Pt / (2**max_exp)
    # print(max_exp)
    mbits = acc_bits + 1  # include sign bit and hidden bit
    max_norm = float(2**(mbits-1) - 1) / 2**(mbits-2)
    Pt = _quantize_elemwise_core(
                Pt, mbits, 0, max_norm, round='floor',
                allow_denorm=True, saturate_normals=True,
                custom_cuda=False)
    Pt = Pt * (2**max_exp)

    # print('>>> trc', file=file)
    # for i, item in enumerate(Pt.reshape(-1)):
    #     binary = float32_to_binary(item)
    #     exp = binary[1:9]
    #     man = binary[9:]
    #     print(f'{int(exp, 2)-127:>4} 1{man}', file=file)
    #     if (i+1) % Pt.shape[-1] == 0:
    #         print(file=file)

    ones = torch.ones(Pt.shape[-1], device=P.device, dtype=P.dtype)  # shape: (N,)

    Y = torch.matmul(Pt, ones)  # (..., K)
    return Y

seed = 42
torch.manual_seed(seed)
A = torch.randn(300, 42, 512)  # (batch1=2, batch2=3, M=4, K=5)
B = torch.randn(512, 512).transpose(0, 1)       # (K=5, N=6)
# A = torch.arange(10).reshape(2, 5)  # (batch1=2, batch2=3, M=4, K=5)
# B = torch.arange(15).reshape(5, 3)       # (K=5, N=6)

A = A.cuda()
B = B.cuda()

def matmul_split_B(A, B, acc_bits, chunk_size=2):
    """
    A: (..., M, K)
    B: (K, N)
    将 B 沿最后一维（列 N）分块，逐块计算 A @ B_chunk，避免 OOM。
    返回: (..., M, N)
    """
    K = A.shape[-1]
    assert B.shape[0] == K, "A and B shape mismatch"

    N = B.shape[1]
    outputs = []   # 保存每个块的结果

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B_chunk = B[:, start:end]        # shape: (K, chunk)

        # 逐块矩阵乘法
        # C_chunk = A @ B_chunk            # shape: (..., M, chunk)
        C_chunk = matmul_decompose_general(A, B_chunk, acc_bits)  # 使用分解方法

        outputs.append(C_chunk)

    # 沿最后一维拼接
    return torch.cat(outputs, dim=-1)

# print((A@B).equal(matmul_split_B(A, B, chunk_size=2)))

for acc_bits in range(14, 41):
    # Y = matmul_decompose_general(A, B, acc_bits)
    Y = matmul_split_B(A, B, acc_bits, chunk_size=128)
    # print(acc_bits, (A@B).sum())


    mse = torch.mean((Y - (A @ B))**2).item()
    print(f"MSE: {acc_bits} {mse:.20f}")
# print("A @ B =\n", A @ B)
# print("Decomposed C =\n", Y)
