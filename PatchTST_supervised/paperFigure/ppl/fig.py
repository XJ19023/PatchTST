
import torch

weight = torch.tensor([7, 7, 8, 7, 8, 13, 14])

weight = weight / weight.sum()

ratio_wiki = torch.tensor([0.3965, 0.2063, 0.1846, 0.2582, 0.5406, 0.4836, 0.4628])
ratio_ptb = torch.tensor([0.4321, 0.4485, 0.3221, 0.5188, 0.2988, 0.3505, 0.3274])

# print(((weight * ratio_wiki).sum() + (weight * ratio_ptb).sum()) / 2)



w4a8_wiki = torch.tensor([5.4327, 5.7198,  6.6942, 7.2352, 10.0539, 5.0088, 8.7748])
w4a8_ptb = torch.tensor([22.3329, 26.2342, 11.2711, 12.7335, 17.3907, 31.0399, 15.0776])


shiftQ_wiki = torch.tensor([5.46, 5.74, 6.77, 7.25, 10.05, 5.04, 8.77])
shiftQ_ptb = torch.tensor([22.20, 26.22, 11.34, 12.76, 17.37, 31.02, 15.09])

# print(w4a8_wiki + w4a8_wiki * 0.01)
# print(w4a8_ptb + w4a8_ptb * 0.01)

wiki = (shiftQ_wiki - w4a8_wiki) / w4a8_wiki * 100
ptb = (shiftQ_ptb - w4a8_ptb) / w4a8_ptb * 100

print(f'w4a8:    {w4a8_wiki}')
print(f'loss 1%: {w4a8_wiki + w4a8_wiki * 0.01}')
print(f'choose:  {wiki}')
# print(ptb)