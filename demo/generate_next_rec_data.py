import json
import random
import shutil

import cv2
import numpy as np
import breezevertex as bvt
import click
import os
from tqdm import tqdm
import time


def backend_matching(backend):
    table = dict(torch=bvt.BACKEND_TORCH, onnx=bvt.BACKEND_ONNXRUNTIME, mnn=bvt.BACKEND_MNN)

    return table[backend]


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    points = points.astype(np.float32)
    # print(points.shape, pts_std.shape)
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def export_data(net, save_dir, phase, data):
    crop_imgs = os.path.join(save_dir, phase, 'crop_imgs')
    w_path = os.path.join(phase, 'crop_imgs')
    os.makedirs(crop_imgs, exist_ok=True)
    label_path = os.path.join(save_dir, phase, 'rec.txt')
    lines = list()
    for idx, path in enumerate(tqdm(data)):
        image = cv2.imread(path)
        ori = image.copy()
        image = cv2.resize(image, INPUT_SIZE)
        t = time.time()
        kps = net(image)
        ut = time.time() - t
        h, w, _ = image.shape
        kps[:, 0] = kps[:, 0] / INPUT_SIZE[1] * h
        kps[:, 1] = kps[:, 1] / INPUT_SIZE[0] * w
        # for x, y in kps.astype(np.int32):
        #     cv2.line(image, (x, y), (x, y), (80, 240, 100), 3)
        # cv2.polylines(image, [kps.astype(np.int32)], True, (0, 0, 200), 1, )
        kps = kps.astype(float)
        height, weight, _ = ori.shape
        kps[:, 0] = kps[:, 0] / h * height
        kps[:, 1] = kps[:, 1] / w * weight
        dst_img = get_rotate_crop_image(ori, kps)
        bs = os.path.basename(path)
        name = bs.replace(".jpg", '')
        name = name.split('-')[-1]
        # print(name)
        line = f"{os.path.join(w_path, name+'.jpg')}\t{name}\n"
        lines.append(line)
        cv2.imwrite(os.path.join(crop_imgs, name+'.jpg'), dst_img)
    with open(label_path, 'w') as f:
        f.write("".join(lines))


@click.command(help='Exec inference flow.')
@click.argument('backend', type=click.Choice(['torch', 'onnx', 'mnn']))
@click.option('-config', '--config', type=click.Path(), default=None, )
@click.option('-model_path', '--model_path', type=click.Path(), default=None, )
@click.option('-data', '--data', type=click.Path(), )
@click.option('-save_dir', '--save_dir', default=None, type=click.Path(), )
@click.option('-input_shape', '--input_shape', default=None, multiple=True, nargs=2, type=int)
@click.option("-show", "--show", is_flag=True, type=click.BOOL, )
def inference(backend, config, model_path, data, save_dir, input_shape, show):
    net = None
    backend_tag = backend_matching(backend)
    if input_shape is not None:
        input_size = input_shape[0]
    else:
        input_size = (112, 112)
    INPUT_SIZE = input_size
    data_list = list()
    if os.path.isdir(data):
        data_list = [os.path.join(data, item) for item in os.listdir(data) if
                     item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
    else:
        data_list.append(data)
    if backend_tag == bvt.BACKEND_TORCH:
        assert config is not None, 'use torch need input config.'
        cfg = bvt.load_cfg(config)
        model_cfg = cfg.model
        if model_path is None:
            model_path = os.path.join(cfg.save_dir, 'best_model.pth')
            assert os.path.exists(model_path), 'The model was not matched.'
        if input_shape is None:
            input_size = tuple(cfg.data.pipeline.image_size)
        infer = bvt.multiple_backend(bvt.BACKEND_TORCH)
        net = infer(weights_path=model_path, input_shape=input_size, model_name=model_cfg.name,
                    model_option=model_cfg.option)
        print(net)

    elif backend_tag == bvt.BACKEND_ONNXRUNTIME:
        assert model_path, 'Place input onnx model path.'
        infer = bvt.multiple_backend(bvt.BACKEND_ONNXRUNTIME)
        net = infer(onnx_path=model_path, input_shape=input_size)
    else:
        return NotImplementedError('not implement backend.')

    assert net is not None, 'error'

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    random.shuffle(data_list)
    split_map = {'train': 0.92, 'val': 0.05, 'test': 0.03}
    phase_list = list()
    phase_rate = list()
    phase_data = list()
    for k, v in split_map.items():
        phase_list.append(k)
        phase_rate.append(v)

    total = len(data_list)
    for idx, rate in enumerate(phase_rate):
        pre = np.sum(phase_rate[:idx])
        end = pre + rate
        pre = int(pre * total)
        end = int(end * total)
        phase_data.append(data_list[pre: end])

    for idx, phase in enumerate(phase_list):
        export_data(net, save_dir, phase, phase_data[idx])



INPUT_SIZE = (112, 112)
if __name__ == '__main__':

    inference()
