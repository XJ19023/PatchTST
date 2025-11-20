import torch
from mx.mx_ops import _quantize_elemwise_core, _shared_exponents
def float32_to_binary(x: torch.Tensor):
    # 确保是 float32 类型
    x = x.to(torch.float32)
    # 重新解释底层比特为 uint32
    bits = x.view(torch.uint32).item()
    # 转成 32 位二进制字符串
    return format(bits, '032b')

seed = 42
torch.manual_seed(seed)
A = torch.randn(3, 4) / (2**50)
A = torch.tensor(A, dtype=torch.float32)
A_test = A.clone()
shared_exp = _shared_exponents(A, axes=[-1])
print('shared_exp before clamping:', shared_exp)
scale_emax = 2**(8-1) - 1
shared_exp[shared_exp > scale_emax] = float("NaN")
shared_exp[shared_exp < -scale_emax] = -scale_emax


A = A / (2**shared_exp)

A = _quantize_elemwise_core(
        A, 8, 0, 1.984375, round='nearest',
        allow_denorm=True, saturate_normals=True,
        custom_cuda=False)

A = A * (2**shared_exp)
# =========================
shared_exp = _shared_exponents(A_test, axes=[-1])
scale_emax = 2**(6-1) - 1
shared_exp[shared_exp > scale_emax] = float("NaN")
# shared_exp[shared_exp < -scale_emax] = -scale_emax
mask = shared_exp < -scale_emax
mask = mask.expand_as(A)
print(mask)

print(A.shape, shared_exp.shape)

A_test = A_test / (2**shared_exp)

A_test = _quantize_elemwise_core(
        A_test, 8, 0, 1.984375, round='nearest',
        allow_denorm=True, saturate_normals=True,
        custom_cuda=False)

A_test = A_test * (2**shared_exp)

A_test = A_test.masked_fill(mask, 0.0)
print(A_test.equal(A))
print(A_test)
print(A)
# file = open('debug.txt', 'w')
# print('>>> org', file=file)
# for i, item in enumerate(A.reshape(-1)):
#     binary = float32_to_binary(item)
#     exp = binary[1:9]
#     exp = int(exp, 2)-127
#     man = binary[9:]
#     shift = '0' * (int(shared_exp - exp))

#     print(f"{exp:>4} {shift}1{man}", file=file)
#     if (i+1) % A.shape[-1] == 0:
#         print(file=file)