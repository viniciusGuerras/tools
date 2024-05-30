import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.style.use('dark_background')

dataframe = pd.read_csv("both/decision_tree/dataset/drug200.csv")

dataframe['Drug'] = dataframe['Drug'].replace({"drugA" : 1, "drugB" : 2,
                           "drugC" : 3, "drugX" : 4,
                           "drugY" : 5})
dataframe['Sex'] = dataframe['Sex'].replace({"F": 1, "M": 0})

dataframe['BP'] = dataframe['BP'].replace({"HIGH": 1, "NORMAL": 0.5, "LOW": 0})

dataframe['Cholesterol'] = dataframe['Cholesterol'].replace({"HIGH": 1, "NORMAL": 0.5})

dataframe['Age'] = (dataframe['Age'] - dataframe['Age'].min()) / (dataframe['Age'].max() - dataframe['Age'].min())
dataframe['Na_to_K'] = (dataframe['Na_to_K'] - dataframe['Na_to_K'].min()) / (dataframe['Na_to_K'].max() - dataframe['Na_to_K'].min())

def gini_impurity(y):
    p = y.value_counts() / y.shape[0]
    gini = 1 - np.sum(p**2)
    return gini

def split_dataset(df, feature, threshold):
    left_split = df[df[feature] <= threshold]
    right_split = df[df[feature] > threshold]
    return left_split, right_split

def best_split(df, target):
    best_gini = float("inf")
    best_feature = None
    best_threshold = None
    features = df.columns.drop(target)

    for feature in features:
        thresholds = df[feature].unique()
        for threshold in thresholds:
            left_split, right_split = split_dataset(df, feature, threshold)
            
            gini_left = gini_impurity(left_split[target])
            gini_right = gini_impurity(right_split[target])
            weighted_gini = (len(left_split) * gini_left + len(right_split) * gini_right) / len(df)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold
                
    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def build_tree(self, df, target, depth=0, max_depth=10):
        y = df[target]

        if depth == max_depth or len(y.unique()) == 1:
            return Node(value=y.mode()[0])

        feature, threshold = best_split(df, target)
        if feature is None:
            return Node(value=y.mode()[0])

        left_split, right_split = split_dataset(df, feature, threshold)
        left_node = self.build_tree(left_split, target, depth + 1, max_depth)
        right_node = self.build_tree(right_split, target, depth + 1, max_depth)
    
        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def predict(self, x):
        if self.value is not None:
            return self.value
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

# Example usage:
tree = Node()
decision_tree = tree.build_tree(dataframe, 'Drug')
prediction = decision_tree.predict(dataframe.iloc[0])
print("Predicted class:", prediction)
