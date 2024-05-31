import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

folder1 = 'both/random_forests/data_set/pizza_not_pizza/pizza'
folder2 = 'both/random_forests/data_set/pizza_not_pizza/not_pizza'

imgs1 = []
imgs2 = []

image_files = os.listdir(folder1)
image_files = [file for file in image_files if file.endswith(".jpg") or file.endswith(".png")]
for image_file in image_files:
    image_path = os.path.join(folder1, image_file)
    img = np.array(Image.open(image_path))    
    imgs1.append(img.flatten())
    
image_files = os.listdir(folder2)
image_files = [file for file in image_files if file.endswith(".jpg") or file.endswith(".png")]
for image_file in image_files:
    image_path = os.path.join(folder2, image_file)
    img = np.array(Image.open(image_path))
    imgs2.append(img.flatten())

df = pd.DataFrame()

df['pizzas'] = imgs1
df['not_pizzas'] = imgs2

print(df)
y = np.array([0 , 1])


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


def get_random_patch(image, patch_size):
    height, width = image.shape[:2]
    print(image.shape)
    patch_x = np.random.randint(0, width - patch_size[1])
    patch_y = np.random.randint(0, height - patch_size[0])
    patch = image[patch_y:patch_y+patch_size[0], patch_x:patch_x+patch_size[1]]
    return patch



print("\n")
print(get_random_patch(imgs1, 100))
class Random_Forest:
    def __init__(self, n_base_learners, max_depth, min_samples_leaf):
        pass    