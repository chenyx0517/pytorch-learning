import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset',False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class MyMaxPool(nn.Module):
    def __init__(self):
        super(MyMaxPool,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool1(input)
        return output


mymaxpool  = MyMaxPool()

writer = SummaryWriter('./conv')

step = 0

for data in dataloader:
    imgs,target = data
    output = mymaxpool(imgs)
    writer.add_images('maxpool',imgs,step)
    writer.add_images('maxpool2',output,step)

    step = step+1

writer.close()