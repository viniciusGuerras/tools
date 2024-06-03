import numpy as np


class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    def backward(self, dvalues):
        self.dinput = (self.output * (1 - self.output) ) * dvalues
        return self.dinput
    

class Softmax:
    def forward(self, inputs):
        self.exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.exponents / np.sum(self.exponents, axis=1, keepdims=True)
        return self.output
    #need to implement this
    def backward(self, dvalues):
        pass


class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Leaky_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, 0.01 * inputs)
        
    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs > 0, dvalues, 0.01 * dvalues)

