import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

img_path = "../train/cbb/train-cbb-0.jpg"

# transforms.ToTensor()
transform1 = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

##numpy.ndarray
img = cv2.imread(img_path)  # 读取图像 3x1080x1920(通道*高*宽)，数值[0, 255]
print("img = ", img)
img1 = transform1(img)  # 归一化到 3x1080x1920(通道*高*宽)，数值[0.0,1.0]
print("img1 = ", img1)

# 转化为numpy.ndarray并显示
img_1 = img1.numpy()*255
img_1 = img_1.astype('uint8')
img_1 = np.transpose(img_1, (1,2,0))
cv2.imshow('img_1', img_1)
cv2.waitKey()

# ##PIL
# img = Image.open(img_path).convert('RGB') # 读取图像
# img2 = transform1(img) # 归一化到 [0.0,1.0]
# print("img2 = ",img2)
#
# #转化为PILImage并显示
# img_2 = transforms.ToPILImage()(img2).convert('RGB')
# print("img_2 = ",img_2)
# img_2.show()
