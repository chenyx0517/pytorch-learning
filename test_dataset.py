import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform, download=False)

print(test_set[0])

writer = SummaryWriter('logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_dataset', img, i)

writer.close()