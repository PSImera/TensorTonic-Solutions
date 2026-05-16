import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    N, H, W, C = x.shape
    x = x.reshape(N, H // 2, 2, W // 2, 2, C)
    return x.max(axis=(2, 4))
