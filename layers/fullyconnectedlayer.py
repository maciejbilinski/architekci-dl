import numpy as np
import layers.weightlayer
import optimization.optimizer


class FullyConnectedLayer(layers.weightlayer.WeightLayer):
    # init values of weight matrix are taken from the standard normal distribution
    def __init__(self, shape, algorithm: optimization.optimizer.Optimizer):
        super().__init__(algorithm)
        self.W = np.random.normal(0, 1, shape)
        self.saved = None

    # matrix multiplication
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.saved = x
        return x @ self.W

    def backward_pass(self, upstream: np.ndarray) -> np.ndarray:
        if self.saved is None:
            raise ValueError('You have to call forward_pass first')
        self.gradient = self.saved.T @ upstream
        self.saved = None
        return upstream @ self.W.T
