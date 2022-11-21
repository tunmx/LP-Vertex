import cv2
import numpy as np


def images_to_square(images, pad=4, resize_pad=None):
    n, h, w, c = images.shape
    assert n == pad * pad
    new_list = list()
    for idx, img in enumerate(images):
        if resize_pad:
            img = cv2.resize(img, (resize_pad, resize_pad))
        new_list.append(img)

    array_images = np.asarray(new_list).reshape((4, 4, h, w, c))
    cols = list()
    for rows in array_images:
        lines = np.concatenate(rows, axis=1)
        cols.append(lines)
    square = np.concatenate(cols, axis=0)

    return square


def encode_images(image: np.ndarray):
    image_encode = image / 255.0
    if len(image_encode.shape) == 4:
        image_encode = image_encode.transpose(0, 3, 1, 2)
    else:
        image_encode = image_encode.transpose(2, 0, 1)
    image_encode = image_encode.astype(np.float32)

    return image_encode


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


colors = [(100, 100, 255), (10, 255, 100), (100, 190, 240), (100, 255, 255)]


def visual_images(images_tensor, label_tensor, w, h, swap=True, is_val=False):
    images = decode_images(images_tensor)
    kps = decode_points(label_tensor, w, h)
    list_ = list()
    for idx, img in enumerate(images):
        if swap:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, p in enumerate(kps[idx].astype(np.int32)):
            x, y = p
            if is_val:
                cv2.line(img, (x, y), (x, y), (127, 255, 0), 3)
            else:
                cv2.line(img, (x, y), (x, y), (0, 255, 255), 3)

            cv2.polylines(img, [kps[idx].astype(np.int32)], True, (255, 0, 0), 1, )
        list_.append(img)

    return list_
