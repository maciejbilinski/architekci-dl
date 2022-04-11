import numpy as np
import layers.layer


class ReLU(layers.layer.Layer):
    def __init__(self):
        self.saved = None

    # matrix multiplication
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.saved = x
        return np.maximum(0, x)

    def backward_pass(self, upstream: np.ndarray) -> np.ndarray:
        if self.saved is None:
            raise ValueError('You have to call forward_pass first')

        out = np.empty(upstream.shape)
        for i in range(self.saved.shape[0]):
            for j in range(self.saved.shape[1]):
                if self.saved[i, j] > 0:
                    out[i, j] = upstream[i, j]
                else:
                    out[i, j] = 0
        self.saved = None
        return out
