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

    def backward(self, dvalues):
        for index, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
                single_output = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
                
class Tanh:
    def forward(self, inputs):
        self.output = (np.exp(inputs) - np.exp(-inputs))/(np.exp(inputs) + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        self.dinput = (1 - (self.output ** 2)) * dvalues
        return self.dinput
    
class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Leaky_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, 0.01 * inputs)
        return self.output
    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs > 0, dvalues, 0.01 * dvalues)
        return self.dinputs
