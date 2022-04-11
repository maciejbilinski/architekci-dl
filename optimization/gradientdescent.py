import optimization.optimizer
import layers.layer


class GradientDescent(optimization.optimizer.Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update_weights(self, layer: layers.layer.Layer) -> None:
        if layer.gradient is None:
            raise ValueError('You have to call backward_pass first')

        layer.W -= self.learning_rate * layer.gradient
