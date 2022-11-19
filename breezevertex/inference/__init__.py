BACKEND_TORCH = 0
BACKEND_ONNXRUNTIME = 1
BACKEND_MNN = 2


def multiple_backend(backend):
    if backend == BACKEND_TORCH:
        from .torch_inference import VertexInferenceTorch
        return VertexInferenceTorch
    elif backend == BACKEND_ONNXRUNTIME:
        from .ort_inference import VertexinferenceOrt
        return VertexinferenceOrt
    elif backend == BACKEND_MNN:
        return NotImplementedError("todo")
    else:
        return NotImplementedError("Unknown backend")
