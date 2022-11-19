import cv2
import numpy as np
import onnxruntime as ort
from .base_inference import VertexInferenceABC
from loguru import logger
from breezevertex.utils.image_tools import encode_images, decode_points

class VertexinferenceOrt(VertexInferenceABC):

    def __init__(self, onnx_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = ort.InferenceSession(onnx_path, None)
        self.input_config = self.session.get_inputs()[0]
        self.output_config = self.session.get_outputs()[0]


    def _predict(self, data: np.ndarray) -> np.ndarray:
        result = self.session.run([self.output_config.name], {self.input_config.name: data})

        return result[0]

    def _postprocess(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[0] == 1
        data = np.asarray(data).reshape(-1, 4, 2)
        data[:, :, 0] *= self.input_shape[1]
        data[:, :, 1] *= self.input_shape[0]

        return data[0]


    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        assert len(
            bgr.shape) == 3, "The dimensions of the input image object do not match. The input supports a single image."
        image_resize = cv2.resize(bgr, self.input_shape)
        encode = encode_images(image_resize)
        encode = encode.astype(np.float32)
        input_tensor = np.expand_dims(encode, 0)

        return input_tensor

