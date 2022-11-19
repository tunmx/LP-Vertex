import os
import click
import cv2
import numpy as np
import tqdm
from loguru import logger
from breezevertex.utils.cfg_tools import load_cfg
from breezevertex.model import build_model
from breezevertex.utils.image_tools import encode_images, images_to_square
import time
import torch

__all__ = ['test']


@click.command(help='Test')
@click.argument('config_path', type=click.Path(exists=True))
@click.option('-model_path', '--model_path', default=None, type=click.Path())
@click.option('-data', '--data', default=None, type=click.Path())
@click.option("-show", "--show", is_flag=True, type=click.BOOL, )
@click.option("-square", "--square", is_flag=True, type=click.BOOL, )
def test(config_path, model_path, data, show, square):
    cfg = load_cfg(config_path)
    print(cfg)
    # build training model
    model_cfg = cfg.model
    if model_path is None:
        model_path = os.path.join(cfg.save_dir, 'best_model.pth')
        assert os.path.exists(model_path), 'The model was not matched.'
    net = build_model(model_cfg.name, **model_cfg.option)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # load data
    path_list = list()
    data_cfg = cfg.data
    if data is None:
        val_cfg = data_cfg.val
        img_path = val_cfg.option.img_path
        if isinstance(img_path, str):
            path_list.append(img_path)
        elif isinstance(img_path, list):
            for idx, p in enumerate(img_path):
                list_ = [os.path.join(p, item) for item in tqdm.tqdm(os.listdir(p)) if
                         item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
                path_list += list_
    else:
        if os.path.isdir(data):
            list_ = [os.path.join(data, item) for item in tqdm.tqdm(os.listdir(data)) if
                     item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
            path_list += list_
        else:
            path_list.append(data)

    # create folder
    save_dir = os.path.join(cfg.save_dir, 'test', str(int(time.time())))
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"predict result to {save_dir} ...")
    h, w = data_cfg.pipeline.image_size
    show_list = list()
    for idx, path in enumerate(tqdm.tqdm(path_list)):
        image = cv2.imread(path)
        image = cv2.resize(image, tuple(data_cfg.pipeline.image_size))
        tensor = encode_images(image)
        tensor = np.expand_dims(tensor, 0)
        tensor = torch.Tensor(tensor)
        with torch.no_grad():
            output = net(tensor)
        output = output.detach().numpy()
        kps = output.reshape(4, 2)
        kps[:, 0] *= w
        kps[:, 1] *= h

        for x, y in kps.astype(np.int32):
            cv2.line(image, (x, y), (x, y), (100, 100, 255), 3)

        if square:

            show_list.append(image)

            if len(show_list) == 16:
                show_list = np.asarray(show_list)
                pad = images_to_square(show_list)
                show_list = list()

                cv2.imwrite(os.path.join(save_dir, str(idx) + '.jpg'), pad)

                if show:
                    cv2.imshow("w", pad)
                    cv2.waitKey(0)
        else:
            cv2.imwrite(os.path.join(save_dir, os.path.basename(path)), image)
            if show:
                cv2.imshow("w", image)
                cv2.waitKey(0)


if __name__ == '__main__':
    test()
