import numpy as np
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn import datasets
digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons));
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
class Activation_Softmax:
    def forward(self, inputs):
        self.exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.exponents / np.sum(self.exponents, axis=1, keepdims=True)
        return self.output

class Sum_Of_Squared_Residuals:
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        self.output = 0.5 * np.sum(((y_true - y_pred) ** 2))
        return self.output
    
    def backward(self):
        self.dinputs = self.y_pred - self.y_true
        return self.dinputs

class Cross_Entropy:
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        m = y_pred.shape[0]
        self.soft = Activation_Softmax()
        p = self.soft.forward(y_pred)
        log_likelihood = -np.log(p[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss
    def backward(self):
        m = self.y_pred.shape[0]
        grad = self.soft.forward(self.y_pred)
        grad[range(m), self.y_true] -= 1
        grad = grad/m
        return grad


class Optimizer_SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights -= (self.learning_rate * layer.dweights)
        layer.biases -= (self.learning_rate * layer.dbiases)


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
            


model = Model()
model.addLayer(Dense_layer(64, 512), Activation_Relu())
model.addLayer(Dense_layer(512, 512), Activation_Relu())
model.addLayer(Dense_layer(512, 10))
model.addLoss(Cross_Entropy())
model.addOptimizer(Optimizer_SGD(0.001))
model.train(200, X_train, y_train)

model.forward(X_test)
predictions = np.argmax(model.core, axis=1)
for i in range(len(predictions)):
    print(X_test.index[i])
    print(f"Predicted: {predictions[i]}, True: {y_test[i]}")
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on the testing set: {accuracy}")