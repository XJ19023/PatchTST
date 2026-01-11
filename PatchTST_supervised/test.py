import torch

a = torch.randn(3, 4)
print(a)
print('-----')
print(a[0])
print('-----')
for item in a[1:]:
    print(item)