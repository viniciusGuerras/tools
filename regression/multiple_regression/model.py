import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



data = pd.read_csv("regression/multiple_regression/datasets/CarPrice_Assignment.csv")

data.drop(columns=['CarName'], inplace=True)

replacement_mapping = {
    "fuelsystem": {"mpfi": 1, "2bbl": 2, "mfi": 3, "1bbl": 4, "idi": 5, "spdi": 6, "4bbl": 7, "spfi": 8},
    "fueltype": {"gas": 1, "diesel": 2},
    "aspiration": {"std": 1, "turbo": 2},
    "enginelocation": {"front": 1, "rear": 2},
    "doornumber": {"two": 2, "four": 4},
    "carbody": {"sedan": 1, "hatchback": 2, "wagon": 3, "hardtop": 4, "convertible": 5},
    "drivewheel": {"rwd": 1, "fwd": 2, "4wd": 3},
    "enginetype": {"dohc": 1, "ohcv": 2, "ohc": 3, "l": 4, "rotor": 5, "dohcv": 6, "ohcf": 7},
    "cylindernumber": {"four": 4, "six": 6, "five": 5, "three": 3, "eight": 8, "two": 2, "twelve": 12}
}


for column, mapping in replacement_mapping.items():
    data[column] = data[column].map(mapping)


list_of_x_cols = ['symboling','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation',
                  'wheelbase','carlength','carwidth','carheight','curbweight','enginetype','cylindernumber',
                  'enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm',
                  'citympg','highwaympg']
y_col = 'price'


data.fillna(data.mean(), inplace=True) 

scaler = StandardScaler()
data[list_of_x_cols] = scaler.fit_transform(data[list_of_x_cols]) 

X_train, X_test, y_train, y_test = train_test_split(data[list_of_x_cols], data[y_col],test_size=0.2)

class Linear_regression:
    def __init__(self, n_features):
        self.weight = 0.1 * np.random.randn(n_features)
        self.bias = np.zeros(1)

    def forward(self, inputs):
        inputs_array = np.array(inputs)
        self.inputs = inputs_array.reshape(-1, inputs.shape[1])
        self.output = np.dot(self.inputs, self.weight) + self.bias
        return self.output

    def backward(self, dvalue):
        self.dweight = np.sum(self.inputs * dvalue, axis=0)
        self.dbias = np.mean(dvalue)
        return self.dweight, self.dbias


class Mean_Squared_Error:
    def forward(self, y_true, y_pred):
        self.y_true = np.array(y_true).reshape(-1, 1)
        self.y_pred = np.array(y_pred).reshape(-1, 1)

        self.m = self.y_true.shape[0]
        self.output = (1/self.m) * np.sum((self.y_true - self.y_pred) ** 2)
        return self.output

    def backward(self):
        self.dinput = -2 * (self.y_true - self.y_pred) / self.m
        return self.dinput

losses = []
regression = Linear_regression(X_train.shape[1])
loss = Mean_Squared_Error()
learning_rate = 0.095
epochs = 2500
for i in range(epochs):
    y_pred = regression.forward(X_train)
    current_loss = loss.forward(y_train, y_pred)
    losses.append(current_loss)
    

    loss.backward()
    regression.backward(loss.dinput)

    regression.weight -= learning_rate * regression.dweight
    regression.bias -= learning_rate * regression.dbias
    

y_train_pred = regression.forward(X_train)
y_test_pred = regression.forward(X_test)


plt.figure(figsize=(12, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Train data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Perfect fit')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Train Data)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_test_pred, color='green', label='Test data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect fit')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices (Test Data)')
plt.legend()
plt.show()