import cv2
import numpy as np


def decode_images(images_tensor):
    image_decode = images_tensor.detach().numpy() * 255
    image_decode = image_decode.transpose(0, 2, 3, 1)
    image_decode = image_decode.astype(np.uint8)
    image_decode = image_decode.copy()

    return image_decode


def decode_points(label_tensor, w, h):
    kps = label_tensor.detach().numpy().reshape(-1, 4, 2)
    kps[:, :, 0] *= w
    kps[:, :, 1] *= h

    return kps


colors = [(100, 100, 255), (10, 255, 100), (255, 100, 20), (100, 255, 255)]
def visual_images(images_tensor, label_tensor, w, h):
    images = decode_images(images_tensor)
    kps = decode_points(label_tensor, w, h)
    list_ = list()
    for idx, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, p in enumerate(kps[idx].astype(np.int32)):
            x, y = p
            cv2.line(img, (x, y), (x, y), colors[i], 3)
        list_.append(img)

    return list_
