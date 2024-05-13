from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

image= Image.open('data/train/ants_image/0013035.jpg')


writer = SummaryWriter('logs')
trans_tensor = transforms.ToTensor()
img_tensor = transforms.ToTensor()(image)
print(img_tensor[0][0][0])

writer.add_image('before',img_tensor)
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])


writer.add_image('norm',img_norm)


# resize
print(image.size)
trans_size = transforms.Resize((512,512))
img_resize = trans_size(image)
img_resize_tensor = trans_tensor(img_resize)
print(img_resize_tensor)


writer.add_image('img_resize_tensor',img_resize_tensor)
writer.close()