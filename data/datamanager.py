import numpy as np


# first row should have only strings with column labels
# last column should be the class of instance
# https://stackoverflow.com/a/4602224
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class DataManager:
    def __init__(self, data, save_original=True):
        error_msg = 'data parameter should be 2d numpy array'
        if type(data) is not np.ndarray:
            print(error_msg)
            pass

        shape = data.shape
        if len(shape) != 2:
            print(error_msg)
            pass
        elif shape[0] < 2 or shape[1] < 2:
            print('the minimum size of array is (2, 2)')
            pass

        self.data = data
        self.x = None
        self.y = None
        self.convention = None

    def get_attributes_length(self):
        return self.data.shape[1] - 1  # last column is not attribute

    def get_attributes(self):
        return self.data[0, :self.get_attributes_length()]

    def get_class_column_name(self):
        return self.data[0, self.get_attributes_length()]

    # unique classes
    def get_classes(self):
        return np.unique(self.data[1:, self.get_attributes_length()])

    def get_classes_length(self):
        return len(self.get_classes())

    def separate_x_y(self):
        self.x = self.data[1:, :self.get_attributes_length()].astype(object)
        self.y = self.data[1:, self.get_attributes_length()].astype(object)
        return self.x, self.y

    # if columns_range is None then all columns will be converted
    # columns_range is list of column indexes
    def convert_x_to_floats(self, columns_range=None):
        if self.x is None:
            print('Firstly you have to separate x and y')
            return False

        if columns_range is None:
            self.x = self.x.astype(float)
            return self.x

        for i in columns_range:
            self.x[:, i] = self.x[:, i].astype(float)
        return self.x

    # classes will be converted from string to simple numbers
    def convert_classes_to_numbers(self):
        if self.y is None:
            print('Firstly you have to separate x and y')
            return False

        classes = self.get_classes()
        self.convention = []
        for i in range(len(classes)):
            self.convention.append((i, classes[i]))
        for i in range(len(self.y)):
            for j in range(len(self.convention)):
                if self.y[i] == self.convention[j][1]:
                    self.y[i] = self.convention[j][0]
                    break

        return self.convention, self.y

    def add_bias_trick(self):
        if self.x is None:
            print('Firstly you have to separate x and y')
            return False

        self.x = np.hstack((self.x, np.ones((self.x.shape[0], 1))))
        return self.x

    def get_training_test_set_randomly(self, test_size=0.2):
        if self.x is None:
            print('Firstly you have to separate x and y')
            return False

        x, y = unison_shuffled_copies(self.x, self.y)
        size = int(self.x.shape[0]*(1-test_size))
        return x[:size, :], y[:size], x[size:, :], y[size:]

    def sort_data_by_class(self):
        if self.x is None:
            print('Firstly you have to separate x and y')
            return False
        z = [(x, y) for y, x in sorted(zip(self.y, self.x), key=lambda pair: pair[0])]
        self.x = np.array(z[0][0])
        self.y = np.array(z[0][1])

        for i in range(1, len(z)):
            self.x = np.vstack((self.x, z[i][0]))
            self.y = np.hstack((self.y, z[i][1]))
        return self.x, self.y

    # if you have the same amount of instances of each class, you may want stay that ratio in training and test sets
    # sorted means that data is sorted by class column
    def get_training_test_set_in_equal_portion(self, test_size=0.2, is_data_sorted=True):
        if not is_data_sorted:
            self.sort_data_by_class()

        classes_length = self.get_classes_length()
        x = np.split(self.x, classes_length)
        y = np.split(self.y, classes_length)

        training_x = np.empty((0, x[0].shape[1]))
        training_y = np.empty(0)
        test_x = np.empty((0, x[0].shape[1]))
        test_y = np.empty(0)
        for i in range(classes_length):
            x[i], y[i] = unison_shuffled_copies(x[i], y[i])
            size = int(x[0].shape[0] * (1 - test_size))
            training_x = np.vstack((training_x, x[i][:size, :]))
            training_y = np.hstack((training_y, y[i][:size]))
            test_x = np.vstack((test_x, x[i][size:, :]))
            test_y = np.hstack((test_y, y[i][size:]))
        return training_x, training_y, test_x, test_y
