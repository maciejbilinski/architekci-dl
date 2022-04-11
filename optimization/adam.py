import optimization.optimizer
import layers.layer
import numpy as np


class Adam(optimization.optimizer.Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.moment1 = 0
        self.moment2 = 0
        self.t = 0

    def update_weights(self, layer: layers.layer.Layer) -> None:
        if layer.gradient is None:
            raise ValueError('You have to call backward_pass first')

        self.t += 1
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * layer.gradient
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * (layer.gradient ** 2)
        moment1_unbias = self.moment1 / (1 - self.beta1 ** self.t)
        moment2_unbias = self.moment2 / (1 - self.beta2 ** self.t)
        layer.W -= self.learning_rate * (moment1_unbias / (np.sqrt(moment2_unbias) + 1e-7))

    def reset(self) -> None:
        self.moment1 = 0
        self.moment2 = 0
        self.t = 0
