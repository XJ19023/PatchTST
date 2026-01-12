
# import torch

# tensor = torch.load('act_scales/traffic_336_336.pt')

# for idx, (k, v) in enumerate(tensor.items()):
#     if idx == 5:
#         break
#     print(f'({v.min().item():.4f}, {v.max().item():2.4f}), {k}')
#     power = torch.log2(v + 1e-6).int()
#     residual = v / (2.0 ** power)
#     pr = 2 ** power * residual
#     print(pr.equal(v))
#     print(f'({pr.min().item():.4f}, {pr.max().item():2.4f}), {k}')
#     print(f'({power.min().item():.4f}, {power.max().item():2.4f}), {k}')
#     print(f'({residual.min().item():.4f}, {residual.max().item():2.4f}), {k}')
#     print('---')

import torch

tensor = torch.load('logs/weather_336_96/smooth_factors.pt')
for k, v in tensor.items():
    print(v[0].max().item(), v[1].max().item(), k)