import json
import os
import random

import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
from imgaug import augmenters as iaa  # 引入数据增强的包


def load_label(path: str) -> dict:
    with open(path, "r") as f:
        result = json.load(f)

    return result


# 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([  # 建立一个名为seq的实例，定义增强方法，用于增强
    sometimes(iaa.Crop(percent=(0, 0.1))),
    iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
    iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
    # iaa.Rotate((-30, 30)),
    iaa.Multiply((0.25, 1.75)),  # 改变亮度, 不影响bounding box
    iaa.ContrastNormalization((0.8, 1.2)),  # 对比度
    iaa.GammaContrast((0.8, 1.5), per_channel=True),    # 随机颜色变换
    # iaa.Sequential([
    #         iaa.Dropout(p=0.005),  # 随机删除像素点
    #     ]),


    # 对一部分图像做仿射变换
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.6), "y": (0.8, 1.6)},  # 图像缩放为80%到120%之间
        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # 平移±20%之间
        rotate=(-25, 25),  # 旋转±45度之间
        shear=(-25, 25),  # 剪切变换±16度，（矩形变平行四边形）
        order=[0, 1],  # 使用最邻近差值或者双线性差值
        cval=0,  # 全白全黑填充
        mode="constant"  # 定义填充图像外区域的方法
    )),
])

root = "/Users/tunm/datasets/oinbagCrawler_vertex/data/"
list_ = [os.path.join(root, item) for item in os.listdir(root) if
                             item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
random.shuffle(list_)
for path in list_:
    img = cv2.imread(path)
    label = load_label(path.replace(".jpg", ".json"))
    polyline = np.asarray(label['shapes'][0]['points'])
    kps = [
        ia.Keypoint(x=polyline[0][0], y=polyline[0][1]),
        ia.Keypoint(x=polyline[1][0], y=polyline[1][1]),
        ia.Keypoint(x=polyline[2][0], y=polyline[2][1]),
        ia.Keypoint(x=polyline[3][0], y=polyline[3][1]),
    ]

    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    print(kpsoi.keypoints)
    # ia.imshow(kpsoi.draw_on_image(image, size=7))

    size = 256

    show_list = list()
    for _ in range(9):
        aug_det = seq.to_deterministic()
        img_aug = aug_det.augment_image(img)
        kps_aug = aug_det.augment_keypoints(kpsoi)
        arr = kps_aug.get_coords_array()
        for x, y in arr.astype(np.int32):
            cv2.line(img_aug, (x, y), (x, y), (100, 100, 255), 9)
        # cv2.imshow("s", img_aug)
        # cv2.waitKey(0)
        img_aug = cv2.resize(img_aug, (size, size))
        show_list.append(img_aug)

    show_list = np.asarray(show_list)
    img_array = np.array(show_list * 9, dtype=np.uint8)
    write_img = np.zeros(shape=(size, (size + 10) * 9, 3), dtype=np.uint8)
    for j, item in enumerate(show_list):
        write_img[:, j * (size + 10): j * (size + 10) + size, :] = item

    cv2.imshow("w", write_img)
    cv2.waitKey(0)
