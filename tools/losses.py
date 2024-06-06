import numpy as np
from tools import activations


class Mean_Squared_Error:
    def forward(self, y_true, y_pred):
        
        self.y_true = y_true.reshape(y_true.shape[0], 1)
        self.y_pred = y_pred.reshape(y_pred.shape[0], 1)

        self.m = self.y_true.shape[0]
        self.output = (1/self.m) * np.sum((y_true - y_pred) ** 2)
        return self.output
    
    def backward(self):
        self.dinput = (-2/self.m) * (self.y_true - self.y_pred)
        return self.dinput
    

class Sum_Of_Squared_Residuals:

    def forward(self, y_true, y_pred):

        self.y_true = y_true.reshape(y_true.shape[0], 1)
        self.y_pred = y_pred.reshape(y_pred.shape[0], 1)

        self.output = (1/2) * np.sum((y_true - y_pred) ** 2)
        return self.output

    def backward(self):
        self.dinput = -(self.y_true - self.y_pred)
        return self.dinput
    
    
class Binary_Cross_Entropy:
    def forward(self, y_true, y_pred):
        self.y_pred = y_pred
        self.y_true = y_true
        self.n = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.output = -1 / self.n * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return self.output
    def backward(self):
        epsilon = 1e-7
        y_pred_clipped = np.clip(self.y_pred, epsilon, 1 - epsilon)
        self.dinputs = -(self.y_true/ y_pred_clipped) + ((1 - self.y_true)/ (1 - y_pred_clipped))
        return self.dinputs
    

class Cross_Entropy:	
	def forward(self, y_true, y_pred_prob):
		self.y_true = y_true
		self.y_pred = y_pred_prob
		y_pred_prob = np.clip(y_pred_prob, 1e-15, 1 - 1e-15)
		self.loss = -np.sum(y_true * np.log(y_pred_prob))
		return self.loss
	
	def backward(self):
		self.dinputs = -self.y_true/self.y_pred
		self.dinputs = self.dinputs / self.y_true.shape[0]
		return self.dinputs
