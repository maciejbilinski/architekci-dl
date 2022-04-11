import abc

import numpy as np

import layers.layer


class LossFunction(layers.layer.Layer, abc.ABC):
    @abc.abstractmethod
    def setup(self, training_y: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def get_loss(self, forward_value: np.ndarray) -> float:
        pass
