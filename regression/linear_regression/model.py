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

#increase of loss with the increase of the dataset, changing for mean squared error should solve
"""class Sum_Of_Squared_Residuals:

    def forward(self, y_true, y_pred):

        self.y_true = y_true.reshape(y_true.shape[0], 1)
        self.y_pred = y_pred.reshape(y_pred.shape[0], 1)

        self.output = (1/2) * np.sum((y_true - y_pred) ** 2)
        return self.output

    def backward(self):
        self.dinput = -(self.y_true - self.y_pred)
        return self.dinput"""
    
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
    
regression = Linear_regression()
loss = Mean_Squared_Error()
epochs = 1000
learning_rate = 0.0015

for epoch in range(epochs):

    regression.forward(X_train)
    loss_value = loss.forward(y_train, regression.output)
    ssr = loss.backward()
    
    dweight, dbias = regression.backward(ssr)
    
    regression.weight -= learning_rate * dweight
    regression.bias -= learning_rate * dbias
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value}")


test_data = pd.read_csv('regression/linear_regression/datasets/test.csv')
test_data['x'] = (test_data['x'] - test_data['x'].min()) / (test_data['x'].max() - test_data['x'].min())
test_data['y'] = (test_data['y'] - test_data['y'].min()) / (test_data['y'].max() - test_data['y'].min())

X_test = test_data['x'].values.reshape(-1, 1)  
y_test = test_data['y'].values.reshape(-1, 1)  

regression.forward(X_test)

print("test loss:", loss.forward(y_test, regression.output))

plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, regression.output, color='red', label='Regression Output')
plt.show()