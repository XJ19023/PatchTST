import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parameter import Parameter


# self.bias = Parameter(torch.empty(out_features, **factory_kwargs))

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
 
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# print(example_targets)
# print(example_data.shape)
 
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
 
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.custom_para_fc1 = Parameter(torch.empty(50))
        self.custom_para_fc2 = Parameter(torch.empty(10))
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x) + self.custom_para_fc1)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) + self.custom_para_fc2
        # print(self.custom_para)
        return F.log_softmax(x, dim=1)

 
model = Net()
model.to("cuda")

# for name, para in model.named_parameters():
#     print(f'-name: {name} -para: {para.shape}')
# for name, module in model.named_modules():
#     print(f'name: {name}')
#     print(f'module: {module}')

with open('log/123.log', 'w') as f:
    f.writelines(f'before: {model.custom_para_fc2}')
# 冻结参数
for name, param in model.named_parameters():
    if 'weight' in name or 'bias' in name:
        param.requires_grad = False
optimizer = optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum)


import functools
def stat_input_hook(m, x, y, name):
    if isinstance(x, tuple):
        x = x[0]
    # print(f'{name}.wgt: {m.weight[0,0:3].tolist()}')
    # print(f'{name}.bias: {m.bias[0:3].tolist()}')
    # print(f'{name}.custom_para: {m.custom_para[0:3].tolist()}')
hooks = []
for name, m in model.named_modules():
    # if isinstance(m, nn.Linear):
    if name == 'fc1':
        hooks.append(
            m.register_forward_hook(functools.partial(stat_input_hook, name=name))
        )

from_check_point = False
if from_check_point:
    network_state_dict = torch.load('model.pth')
    model.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load('optimizer.pth')
    optimizer.load_state_dict(optimizer_state_dict)
 
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
 
 
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm((train_loader), total = len(train_loader))):
        data=data.to("cuda")
        target=target.to("cuda")
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:

            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
        # if batch_idx == 5:
        #     break

        loop.set_description(f'Epoch [{epoch}/{epoch}]')
        loop.set_postfix(loss = loss.item())
    torch.save(model.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')
 
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data=data.to("cuda")
            target=target.to("cuda")
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            # if batch_idx == 5:
            #     break
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
 
# test()  # 不加这个，后面画图就会报错：x and y must be the same size
for epoch in range(1, 4):
    train(epoch)
with open('log/123.log', 'a') as f:
    f.writelines(f'after: {model.custom_para_fc2}')
test()



'''
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
 
 
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = model(example_data)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
 
 
# ----------------------------------------------------------- #
 
continued_network = Net()
continued_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
 
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
 
# 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# 不然报错：x and y must be the same size
# 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
for i in range(4, 9):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()
 
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
'''
