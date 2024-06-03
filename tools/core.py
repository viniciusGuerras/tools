import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.activations = []

    def addLayer(self, layer, activation=None):
        self.layers.append(layer)
        if activation is None:
            self.activations.append(None)
        else:
            self.activations.append(activation)

    def addLoss(self, loss):
        self.lossFunction = loss

    def addOptimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, X):
        self.core = X
        for i in range(len(self.layers)):
            self.layers[i].forward(self.core)
            self.core = self.layers[i].output
            if self.activations[i] is not None:
                self.activations[i].forward(self.core)
                self.core = self.activations[i].output

    def backward(self, y_true):
        loss_gradient = self.lossFunction.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            if self.activations[i] is not None:
                self.activations[i].backward(loss_gradient)
                loss_gradient = self.activations[i].dinputs
            self.layers[i].backward(loss_gradient)
            loss_gradient = self.layers[i].dinputs
    
    def train(self, epochs, X, y):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.lossFunction.forward(y, self.core)

            self.backward(y)
            for layer in self.layers:
                self.optimizer.update_params(layer)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
            