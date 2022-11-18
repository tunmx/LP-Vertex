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


def visual_images(images_tensor, label_tensor, w, h):
    images = decode_images(images_tensor)
    kps = decode_points(label_tensor, w, h)
    for idx, img in enumerate(images):
        for x, y in kps[idx].astype(np.int32):
            cv2.line(img, (x, y), (x, y), (100, 100, 255), 3)

    return images
