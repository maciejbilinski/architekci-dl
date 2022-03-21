import numpy as np


class CSVReader:
    def __init__(self, file, separator=','):
        f = open(file, 'r')
        lines = f.readlines()
        f.close()

        self.data = [line.replace('\n', '').split(separator) for line in lines]

    def get_data(self):
        return self.data

    def get_numpy(self):
        return np.array(self.data)
