import torchvision
from torch.utils.data import DataLoader
from model import *
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)


train_data_loader = DataLoader(train_data,batch_size=64)
test_data_loader = DataLoader(test_data,batch_size=64)

myModel = MyModel()

#创建损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
optmizer = torch.optim.SGD(myModel.parameters(),0.01)

#设置训练网络的参数
total_train_step = 0
total_test_step = 10
epoch = 10

for i in range(epoch):
    print('第{}轮'.format(i+1))
    for data in train_data_loader:
        imgs,targets = data
        outputs = myModel(imgs)
        loss = loss_fn(outputs,targets)


        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        total_train_step = total_train_step+1
        print("训练次数：{},loss：{}".format(total_train_step,loss.item()))
