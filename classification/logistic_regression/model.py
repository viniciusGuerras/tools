import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("classification/logistic_regression/dataset/logistic_regression_dataset.csv")

X = df.drop("Medical_Condition", axis=1)
y = df['Medical_Condition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2)

class Logistic_regression:
    def __init__(self, n_features):
        self.weight = 0.1 * np.random.randn(n_features)
        self.bias = np.zeros(1)
        self.dweight = None 

    def forward(self, inputs):
        inputs_array = np.array(inputs)
        self.inputs = inputs_array.reshape(-1, inputs.shape[1])
        self.output = np.dot(self.inputs, self.weight) + self.bias
        return self.output

    def backward(self, dvalue):
        self.dweight = np.array(self.dweight).reshape(-1, 1)
        self.dweight = np.dot(self.inputs.T, dvalue)
        self.dbias = np.mean(dvalue)
        return self.dweight, self.dbias

class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    def backward(self, dvalues):
        self.dinput = (self.output * (1 - self.output) ) * dvalues
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


logistic_regression = Logistic_regression(X_train.shape[1])
sigmoid = Sigmoid()
loss = Binary_Cross_Entropy()
epochs = 1000
learning_rate = 0.0001  

for i in range(epochs):
    logistic_regression.forward(X_train)
    sigmoid.forward(logistic_regression.output)
    loss.forward(y_train, sigmoid.output)
    loss.backward()
    sigmoid.backward(loss.dinputs)  
    logistic_regression.backward(sigmoid.dinput)
    
    logistic_regression.weight -= (logistic_regression.dweight * learning_rate)
    logistic_regression.bias -= (logistic_regression.dbias * learning_rate)


logistic_regression.forward(X_test)
sigmoid.forward(logistic_regression.output)

y_pred_prob = sigmoid.output
y_pred = np.where(y_pred_prob >= 0.84, 1, 0)

for pred, true in zip(y_pred, y_test):
    print(f"true:{true}, predicted:{pred}")

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)