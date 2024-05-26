import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 定义训练的设备
device = torch.device('cuda')
print(device)
train_data = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_loader = DataLoader(train_data,batch_size=64)
test_data_loader = DataLoader(test_data,batch_size=64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.model(x)
        return x


model = MyModel()
model.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


optmizer = torch.optim.SGD(model.parameters(),0.01)

#设置训练网络的参数
total_train_step = 0
total_test_step = 10
epoch = 10


for i in range(epoch):
    print('第{}轮'.format(i+1))
    for data in train_data_loader:
        imgs,targets = data
        imgs =imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs,targets)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        total_train_step = total_train_step+1
        print("训练次数：{},loss：{}".format(total_train_step,loss.item()))
