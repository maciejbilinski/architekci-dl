import numpy as np
import abc
import optimization.optimizer
import layers.layer


class WeightLayer(layers.layer.Layer, abc.ABC):
    def __init__(self, algorithm: optimization.optimizer.Optimizer):
        self.W = None
        self.gradient = None
        self.optimizer = algorithm

    def get_weights(self) -> np.ndarray:
        return self.W

    def update_weights(self) -> None:
        self.optimizer.update_weights(self)
