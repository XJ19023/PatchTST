import torch
tensor = torch.load('act_scales/patchTST.pt')
# print(type(tensor))
file_handle = open('logs/act_patchTST.txt', 'w')
for k, v in tensor.items():
    print(f'{k:<25} {v.shape}', file=file_handle)
file_handle.close()