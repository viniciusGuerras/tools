import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("regression/linear_regression/datasets/train.csv")

#reember to normalize the data
data['x'] = (data['x'] - data['x'].min()) / (data['x'].max() - data['x'].min())
data['y'] = (data['y'] - data['y'].min()) / (data['y'].max() - data['y'].min())

X_train = data['x'].values.reshape(-1, 1)  
y_train = data['y'].values.reshape(-1, 1)  

class Linear_regression:

    def __init__(self):
        self.weight = 0.1 * np.random.randn(1)
        self.bias = np.zeros(1)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weight) + self.bias
        return self.output
        
    def backward(self, dvalue):
        self.dweight = np.sum(self.inputs * dvalue)
        self.dbias = np.mean(dvalue)
        return self.dweight, self.dbias

class Sum_Of_Squared_Residuals:

    def forward(self, y_true, y_pred):

        self.y_true = y_true.reshape(y_true.shape[0], 1)
        self.y_pred = y_pred.reshape(y_pred.shape[0], 1)

        self.output = (1/2) * np.sum((y_true - y_pred) ** 2)
        return self.output

    def backward(self):
        self.dinput = -(self.y_true - self.y_pred)
        return self.dinput
    
regression = Linear_regression()
loss = Sum_Of_Squared_Residuals()
epochs = 300
learning_rate = 0.001

for epoch in range(epochs):

    regression.forward(X_train)
    loss_value = loss.forward(y_train, regression.output)
    ssr = loss.backward()
    
    dweight, dbias = regression.backward(ssr)
    
    regression.weight -= learning_rate * dweight
    regression.bias -= learning_rate * dbias
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss_value}")


plt.scatter(X_train, y_train)
plt.plot(X_train, regression.forward(X_train), color='red')
plt.show()