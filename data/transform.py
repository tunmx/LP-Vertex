import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
from imgaug import augmenters as iaa  # 引入数据增强的包


class Pipeline(object):

    def __init__(self):
        # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential([  # 建立一个名为seq的实例，定义增强方法，用于增强
            sometimes(iaa.Crop(percent=(0, 0.1))),
            iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
            iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
            # iaa.Rotate((-30, 30)),
            iaa.Multiply((0.25, 1.75)),  # 改变亮度, 不影响bounding box
            iaa.ContrastNormalization((0.8, 1.2)),  # 对比度
            iaa.GammaContrast((0.9, 1.2), per_channel=True),  # 随机颜色变换
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

            iaa.Resize((112, 112), ),
        ])

    def _transform_one(self, image: np.ndarray, points: np.ndarray) -> tuple:
        kps = [
            ia.Keypoint(x=points[0][0], y=points[0][1]),
            ia.Keypoint(x=points[1][0], y=points[1][1]),
            ia.Keypoint(x=points[2][0], y=points[2][1]),
            ia.Keypoint(x=points[3][0], y=points[3][1]),
        ]
        kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)
        aug_det = self.seq.to_deterministic()
        img_aug = aug_det.augment_image(image)
        kps_aug = aug_det.augment_keypoints(kpsoi)
        kps_out = kps_aug.get_coords_array()

        return img_aug, kps_out

    def __call__(self, image: np.ndarray, points: np.ndarray, *args, **kwargs) -> tuple:
        return self._transform_one(image, points)
