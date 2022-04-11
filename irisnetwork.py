import data.csvreader
import data.datamanager
import layers.crossentropyloss
import layers.fullyconnectedlayer
import layers.relu
import network
import optimization.adam
import optimization.gradientdescent


class IrisNetwork(network.Network):
    def __init__(self):
        iris = data.csvreader.CSVReader('data/IRIS.csv')
        iris_np = iris.get_numpy()

        manager = data.datamanager.DataManager(iris_np)
        x, y = manager.separate_x_y()
        x = manager.convert_x_to_floats()
        convention, y = manager.convert_classes_to_numbers()
        # classes = [x[1] for x in convention]
        x = manager.add_bias_trick()

        optimizer = optimization.adam.Adam

        super().__init__(manager, layers.crossentropyloss.CrossEntropyLoss())

        hidden_layer_length = 10
        self.layers.append(layers.fullyconnectedlayer.FullyConnectedLayer((self.training_x.shape[1], 100), optimizer()))
        self.layers.append(layers.relu.ReLU())
        self.layers.append(layers.fullyconnectedlayer.FullyConnectedLayer((100, 200), optimizer()))
        self.layers.append(layers.relu.ReLU())
        self.layers.append(layers.fullyconnectedlayer.FullyConnectedLayer((200, manager.get_classes_length()), optimizer()))
