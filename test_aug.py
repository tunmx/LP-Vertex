import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

from imgaug import augmenters as iaa  # 引入数据增强的包

# 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([  # 建立一个名为seq的实例，定义增强方法，用于增强
    iaa.Crop(px=(0, 16)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
    iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
    iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
    iaa.Rotate((-30, 30)),

    # 对一部分图像做仿射变换
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
        rotate=(-25, 25),  # 旋转±45度之间
        shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
        order=[0, 1],  # 使用最邻近差值或者双线性差值
        cval=0,  # 全白全黑填充
        mode=ia.ALL  # 定义填充图像外区域的方法
    )),
])

img = cv2.imread("/Users/tunm/Downloads/hand2.png")
for _ in range(10):
    img_aug = seq.augment_image(img)
    cv2.imshow("s", img_aug)
    cv2.waitKey(0)