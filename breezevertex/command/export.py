import os
import click
from loguru import logger
from breezevertex.utils.cfg_tools import load_cfg
from breezevertex.model import build_model
import torch
import onnxsim
import onnx

__all__ = ['export']


def export_onnx(net, model_save, input_shape: tuple):
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[0], input_shape[1])
    )
    torch.onnx.export(
        net,
        dummy_input,
        model_save,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )
    logger.info("finished exporting onnx.")
    logger.info("start simplifying onnx.")
    input_data = {"data": dummy_input.detach().cpu().numpy()}
    model_sim, flag = onnxsim.simplify(model_save, input_data=input_data)
    if flag:
        onnx.save(model_sim, model_save)
        logger.info("simplify onnx successfully")
    else:
        logger.error("simplify onnx failed")
    logger.info(f"export onnx model to {model_save}")


@click.command(help='export model : onnx or ..')
@click.argument('model', type=click.Choice(['onnx', 'mnn', 'ncnn', ]))
@click.option('-config', '--config', type=click.Path(exists=True))
@click.option('-model_path', '--model_path', default=None, type=click.Path())
@click.option('-model_save', '--model_save', default=None, type=click.Path())
@click.option('-input_shape', '--input_shape', default=None, multiple=True, nargs=2, type=int)
def export(model, config, model_path, model_save, input_shape):
    logger.info("export")
    cfg = load_cfg(config)
    print(cfg)
    if input_shape:
        input_shape = input_shape[0]
    else:
        input_shape = tuple(cfg.data.pipeline.image_size)
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

    if model_save is None:
        model_save = os.path.join(cfg.save_dir, 'model.onnx')

    if model == 'onnx':
        export_onnx(net, model_save, input_shape)


if __name__ == '__main__':
    export()
