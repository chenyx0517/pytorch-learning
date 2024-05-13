import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear,Sequential

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)


class MyModel(nn.Module):
    def __init__(self):
        super(self, MyModel).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(10240, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1()
        return x
