from abc import ABCMeta, abstractmethod


class VertexInferenceABC(metaclass=ABCMeta):

    def __init__(self, input_shape: tuple = (112, 112), *args, **kwargs):
        self.input_shape = input_shape

    @abstractmethod
    def _predict(self, data):
        pass

    @abstractmethod
    def _postprocess(self, data):
        pass

    @abstractmethod
    def _preprocess(self, image):
        pass

    def __call__(self, image):
        flow = self._preprocess(image)
        flow = self._predict(flow)
        result = self._postprocess(flow)

        return result
