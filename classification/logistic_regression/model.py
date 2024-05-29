import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""df = pd.read_csv("classification/logistic_regression/dataset/framingham.csv")

df = df.fillna(df.mean())

columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 
           'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']

df = pd.DataFrame(df, columns=columns)

X = df.drop("TenYearCHD", axis=1)
y = df['TenYearCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized_df = pd.DataFrame(X_scaled, columns=X.columns)
 
y = df['TenYearCHD']  """

class Logistic_regression:
    def __init__(self, n_features):
        self.weight = 0.1 * np.random.randn(n_features)
        self.bias = np.zeros(1)

    def forward(self, inputs):
        inputs_array = np.array(inputs)
        self.inputs = inputs_array.reshape(-1, inputs.shape[1])
        self.output = np.dot(self.inputs, self.weight) + self.bias
        return self.output

    def backward(self, dvalue):
        dvalue = np.array(dvalue).reshape(-1, 1)
        self.dweight = np.sum(self.inputs * dvalue, axis=0)
        self.dbias = np.mean(dvalue)
        return self.dweight, self.dbias

class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    def backward(self, dvalues):
        self.dinput = (self.output * (1 - self.output) )* dvalues
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
        self.dinputs = (y_pred_clipped - self.y_true) / (self.n * (y_pred_clipped * (1 - y_pred_clipped)))
        return self.dinputs
    