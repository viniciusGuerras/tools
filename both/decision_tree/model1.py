import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

plt.style.use('dark_background')

#data clean-up
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
    #calculate impurity (1 - summation of p^2)
    p = y.value_counts() / y.shape[0]
    gini = 1 - np.sum(p**2)
    return gini

def split_dataset(df, feature, threshold):
    #separate the dataset feature in two based on a threshold
    left_split = df[df[feature] <= threshold]
    right_split = df[df[feature] > threshold]
    return left_split, right_split

def best_split(df, target):
    #finds the best split
    
    #initialize variables
    best_gini = float("inf")
    best_feature = None
    best_threshold = None
    features = df.columns.drop(target)

    #for feature in the dataset(without y)
    for feature in features:
        thresholds = df[feature].unique()
        #for class inside the feature column
        for threshold in thresholds:
            #split it in two based in the treshold
            left_split, right_split = split_dataset(df, feature, threshold)
            
            #check the purity left
            gini_left = gini_impurity(left_split[target])
            #check the purity right
            gini_right = gini_impurity(right_split[target])
            #check the weighted purity
            weighted_gini = (len(left_split) * gini_left + len(right_split) * gini_right) / len(df)
            
            #goes on until the best_gini is found
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold
                
    #return the feature and threshold
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

        #check for depth (preventing from overfitting) or if y is a sinngle class (then all will be the same)
        if depth == max_depth or len(y.unique()) == 1:
            return Node(value=y.mode()[0])

        #find the best feature and threshold for the split
        feature, threshold = best_split(df, target)
        if feature is None:
            return Node(value=y.mode()[0])

        #split on the best gini feature and threshold
        left_split, right_split = split_dataset(df, feature, threshold)
        
        #recursively do the left branch of the root
        left_node = self.build_tree(left_split, target, depth + 1, max_depth)
        
        #recursively do the right branch of the root
        right_node = self.build_tree(right_split, target, depth + 1, max_depth)
    
        #return root at the end and Nodes elsewhere
        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def predict(self, x):
        #check if its a leaf Node
        if self.value is not None:
            return self.value
        #else, uses the feature of x to see if beats the threshold and go left or right from there
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

# Example usage:
tree = Node()
decision_tree = tree.build_tree(dataframe, 'Drug')
prediction = decision_tree.predict(dataframe.iloc[1].drop("Drug"))
print("Predicted class:", prediction)
