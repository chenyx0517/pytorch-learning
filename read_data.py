from torch.utils.data import Dataset
from PIL import Image
import os

# 继承的是dataset这个类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        # 图片路径数组
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)




root_dir = 'data/train'
bees_label_dir = 'bees_image'
bees_dataset = MyData(root_dir, bees_label_dir)

ants_label_dir = 'ants_image'
ants_dataset = MyData(root_dir, ants_label_dir)

train_dataset = bees_dataset + ants_dataset

