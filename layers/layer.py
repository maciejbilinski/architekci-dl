import numpy as np
import abc


class Layer(abc.ABC):
    # return scores and save x if needed
    @abc.abstractmethod
    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        pass

    # set self.gradient attribute as gradient on weights matrix and return gradient on x variable
    @abc.abstractmethod
    def backward_pass(self, upstream: np.ndarray) -> np.ndarray:
        pass
