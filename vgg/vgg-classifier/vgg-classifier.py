import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    x = features
    B = x.shape[0]

    x = x.reshape(B, -1)
    x = np.maximum(0, x @ W1 + b1)
    x = np.maximum(0, x @ W2 + b2)
    return x @ W3 + b3