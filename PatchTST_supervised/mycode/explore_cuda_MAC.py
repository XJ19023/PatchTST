import torch

'''
bf16_max = torch.finfo(torch.bfloat16).max
bf16_min = torch.finfo(torch.bfloat16).min

bf16_max = torch.tensor(bf16_max, dtype=torch.bfloat16)
bf16_min = torch.tensor(bf16_min, dtype=torch.bfloat16)

def bfloat16_to_binary(x: torch.Tensor):
    # 确保是 bfloat16 类型
    x = x.to(torch.bfloat16)
    # 重新解释底层比特为 uint16
    bits = x.view(torch.uint16).item()
    # 转成 16 位二进制字符串
    return format(bits, '016b')

binary = bfloat16_to_binary(bf16_max)
print("Value:", bf16_max.item())
# print(binary)
print(f"Binary: {binary[1:9]} {binary[9:]}")
'''
def float32_to_binary(x: torch.Tensor):
    # 确保是 float32 类型
    x = x.to(torch.float32)
    # 重新解释底层比特为 uint32
    bits = x.view(torch.uint32).item()
    # 转成 32 位二进制字符串
    return format(bits, '032b')

import struct
def binary_to_fp32(b: str) -> float:
    """
    将 32 位二进制字符串转换为 IEEE754 float32
    """
    if len(b) != 32 or any(c not in "01" for c in b):
        raise ValueError("必须输入长度为 32 的二进制字符串")

    # 将二进制字符串转换为 32 位整数
    int_val = int(b, 2)

    # 使用 struct 将 uint32 按 IEEE754 重新解释为 float32
    return struct.unpack('!f', struct.pack('!I', int_val))[0]


# 示例

# B = binary_to_fp32(f"0{127-24:08b}11111111111111111111111")
for i in range(1):
    A = binary_to_fp32(f"0{127:08b}00000000000000000000001")                  # 100000000000000000000001
    B = binary_to_fp32(f"0{127-25:08b}11111111111111111111111")               #                         0111111111111111111111111
    C = binary_to_fp32(f"0{127-25-10:08b}00000000000000000000000")             #                           100000000000000000000000
    Z = binary_to_fp32(f"0{0:08b}00000000000000000000000")             # 0
    

    tensorA = torch.tensor([A, Z, B, B, B, A, Z, Z, A, B, B, Z], dtype=torch.float32)
    tensorA = tensorA.cuda()
    tensorA = tensorA.unsqueeze(0)
    print(tensorA.shape)
    print(f' .sum(), {tensorA.sum():.16f}')

    tensorB = torch.tensor([1 for _ in range(tensorA.size(1))], dtype=torch.float32)
    tensorB = tensorB.cuda()
    tensorB = tensorB.unsqueeze(0)
    tensorB = tensorB.transpose(0, 1)

    tensorY = tensorA @ tensorB
    print(f'A and B, {tensorY[0, 0]:.16f}')

    tensorA = torch.tensor([A, A, A], dtype=torch.float32)
    tensorA = tensorA.cuda()
    tensorA = tensorA.unsqueeze(0)

    tensorB = torch.tensor([1 for _ in range(tensorA.size(1))], dtype=torch.float32)
    tensorB = tensorB.cuda()
    tensorB = tensorB.unsqueeze(0)
    tensorB = tensorB.transpose(0, 1)

    tensorY = tensorA @ tensorB
    print(f' only A, {tensorY[0, 0]:.16f}')

    
bf16_max = torch.finfo(torch.bfloat16).max
bf16_max = torch.tensor(bf16_max, dtype=torch.bfloat16)
# print(bf16_max+bf16_max)