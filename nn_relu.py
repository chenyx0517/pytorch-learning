import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./daraset',train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


writer = SummaryWriter('conv')

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.Sigmoid = Sigmoid()

    def forward(self,input):
        output = self.Sigmoid(input)
        return output


mymodel = MyModel()
step = 1
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs,global_step=step)
    output = mymodel(imgs)
    writer.add_images("input2", output,global_step=step)
    step =step+1


writer.close()