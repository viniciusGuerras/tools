import numpy as np


class Optimizer_SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights -= (self.learning_rate * layer.dweights)
        layer.biases -= (self.learning_rate * layer.dbiases)

#TODO adagrad oprimizer

#TODO adadelta oprimizer

#TODO adam oprimizer