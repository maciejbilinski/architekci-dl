import abc
import layers.layer


class Optimizer(abc.ABC):
    def update_weights(self, layer: layers.layer.Layer) -> None:
        pass

    def reset(self) -> None:
        pass
