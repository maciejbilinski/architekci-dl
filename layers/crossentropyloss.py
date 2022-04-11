import numpy as np
import layers.lossfunction


class CrossEntropyLoss(layers.lossfunction.LossFunction):
    def setup(self, training_y: np.ndarray) -> None:
        self.y = np.zeros((training_y.shape[0], np.max(training_y) + 1))
        for i in range(training_y.shape[0]):
            self.y[i, training_y[i]] = 1

    def __init__(self):
        self.y = None
        self.saved = None

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        self.saved = x
        for i in range(x.shape[0]):
            _a = np.e ** x[i]
            _a_sum = _a.sum()
            for j in range(x.shape[1]):
                self.saved[i, j] = _a[j] / _a_sum
        return self.saved

    def backward_pass(self, upstream: np.ndarray = None) -> np.ndarray:
        if self.saved is None:
            raise ValueError('You have to call forward_pass first')
        if self.y is None or self.y.shape != self.saved.shape:
            raise ValueError('You have to call setup with property training_y shape')

        out = self.saved - self.y
        self.saved = None
        return out

    def get_loss(self, forward_value: np.ndarray) -> float:
        L_i = np.empty(forward_value.shape[0])
        for i in range(self.y.shape[0]):
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1:
                    L_i[i] = forward_value[i, j]
                    if L_i[i] < 0.0000001:
                        L_i[i] = 0.0000001
                    break
        return np.sum(-np.log(L_i)) / self.y.shape[0]
