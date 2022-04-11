import abc

import numpy as np

import layers
import data.datamanager
import layers.lossfunction
import layers.layer
import layers.weightlayer


# TODO: mini-batch
class Network(abc.ABC):
    def __init__(self, manager: data.datamanager.DataManager, loss_fn: layers.lossfunction.LossFunction):
        self.training_x, self.training_y, self.test_x, self.test_y = manager.get_training_test_set_in_equal_portion()

        self.layers: list[layers.layer.Layer] = []
        self.loss_fn = loss_fn
        self.loss_fn.setup(self.training_y)

    def train(self, iterations: int, reset_optimizers: bool = False) -> float:
        if reset_optimizers:
            for layer in self.layers:
                if isinstance(layer, layers.weightlayer.WeightLayer):
                    layer.optimizer.reset()

        forward = None
        for i in range(iterations):
            forward = self.training_x
            for layer in self.layers:
                forward = layer.forward_pass(forward)
            forward = self.loss_fn.forward_pass(forward)

            backward = self.loss_fn.backward_pass()
            for layer in reversed(self.layers):
                backward = layer.backward_pass(backward)
                if isinstance(layer, layers.weightlayer.WeightLayer):
                    layer.update_weights()

        return self.loss_fn.get_loss(forward)

    def check_accuracy(self):
        scores = self.test_x
        for layer in self.layers:
            scores = layer.forward_pass(scores)

        correct = 0
        options = np.zeros(np.max(self.test_y) + 1)
        for i in range(scores.shape[0]):
            selected_class = np.argmax(scores[i, :])
            options[selected_class] += 1
            if selected_class == self.test_y[i]:
                correct = correct + 1

        return correct / self.test_x.shape[0], options
