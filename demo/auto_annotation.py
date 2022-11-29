"""
============================[-Benchmark-]==============================
[Model]         [Input]     [Loss]      [use-time@corei7]     [ONNX-size]
--
ResNet50        112x112     2.6e-4      23ms                    96.2M
MNetV2:0.5      112x112     3.6e-4      5ms                     96.2M
MNetV2:0.35     96x96       6.5e-4      1.7ms                    2.2M
MNetV2:0.25     96x96       6.4e-4      1.3ms                    1.6M
"""
import json
import shutil

import cv2
import numpy as np
import breezevertex as bvt
import click
import os
from tqdm import tqdm
import time

vertex_temp = {
    "version": "5.0.5",
    "flags": {},
    "shapes": [
        {
            "label": "plate",
            "points": [
                [
                    1127.5483870967741,
                    1238.0
                ],
                [
                    3104.9677419354834,
                    1192.8387096774193
                ],
                [
                    3182.387096774193,
                    2015.4193548387095
                ],
                [
                    1092.0645161290322,
                    2241.2258064516127
                ]
            ],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
    ],
    "imagePath": "xxx.jpg",
    "imageData": None,
    "imageHeight": 0,
    "imageWidth": 0
}

def backend_matching(backend):
    table = dict(torch=bvt.BACKEND_TORCH, onnx=bvt.BACKEND_ONNXRUNTIME, mnn=bvt.BACKEND_MNN)

    return table[backend]


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

    sum_t = 0.0
    for path in tqdm(data_list):
        image = cv2.imread(path)
        ori = image.copy()
        image = cv2.resize(image, input_size)
        t = time.time()
        kps = net(image)
        ut = time.time() - t
        sum_t += ut
        h, w, _ = image.shape
        kps[:, 0] = kps[:, 0] / input_size[1] * h
        kps[:, 1] = kps[:, 1] / input_size[0] * w
        # for x, y in kps.astype(np.int32):
        #     cv2.line(image, (x, y), (x, y), (80, 240, 100), 3)
        # cv2.polylines(image, [kps.astype(np.int32)], True, (0, 0, 200), 1, )
        kps = kps.astype(float)
        if show:
            cv2.imshow('image', image)
            cv2.waitKey(0)
        if save_dir:
            height, weight, _ = ori.shape
            kps[:, 0] = kps[:, 0] / h * height
            kps[:, 1] = kps[:, 1] / w * weight
            data_result = vertex_temp.copy()
            data_result['shapes'][0]['points'][0][0] = kps[0][0]
            data_result['shapes'][0]['points'][0][1] = kps[0][1]
            data_result['shapes'][0]['points'][1][0] = kps[1][0]
            data_result['shapes'][0]['points'][1][1] = kps[1][1]
            data_result['shapes'][0]['points'][2][0] = kps[2][0]
            data_result['shapes'][0]['points'][2][1] = kps[2][1]
            data_result['shapes'][0]['points'][3][0] = kps[3][0]
            data_result['shapes'][0]['points'][3][1] = kps[3][1]
            data_result['imagePath'] = os.path.basename(path)
            data_result['imageHeight'] = height
            data_result['imageWidth'] = weight

            basename = os.path.basename(path)
            image_dst_path = os.path.join(save_dir, basename)
            label_dst_path = os.path.join(save_dir, "".join(basename.split('.')[:-1])+'.json')
            # print(label_dst_path)
            shutil.copy(path, image_dst_path)
            with open(label_dst_path, 'w') as f:
                json.dump(data_result, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    inference()
