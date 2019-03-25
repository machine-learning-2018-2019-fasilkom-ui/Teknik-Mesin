from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import random

print("Done importing")

def split(df, label):
    X = data.drop(columns=label)
    y = data[label]
    return X, y

class Tree:
    def __init__(self, is_leaf, label, threshold, gain):
        self.branches = {}
        self.is_leaf = is_leaf
        self.label = label
        self.threshold = threshold
        self.gain = gain

    def add(self, branch, subtree):
        self.branches[branch] = subtree

    def traverse(self, inputs):
        return self.traverseHelper(self, inputs)

    def traverseHelper(self, curr_node, inputs):
        if curr_node.is_leaf:
            return curr_node.label
        else:
            value = inputs.get(curr_node.label)
            temp = None
            if type(value) is str:
                temp = curr_node.branches.get(value)
            else:
                threshold = curr_node.threshold
                temp2 = "<="+str(threshold) if value <= threshold else ">"+str(threshold)
                temp = curr_node.branches.get(temp2)
            return self.traverseHelper(temp, inputs)
    
    def cetak(self):
        queue = [self]
        while len(queue) > 0:
            curr = queue.pop(0)
            print(curr, curr.branches)
            for x in curr.branches:
                queue.append(curr.branches[x])

    def __repr__(self):
        return self.label + " : " + str(round(self.gain,4))
            
class DecisionTreeClassifier:
    
    def __init__(self):
        self.T = None
        self.target = None
        
    def log2(self, k):
        return 0 if k == 0 else np.log2(k)
        
    def info(self, T):
        m = len(T)
        labels = T[self.target].value_counts()
        result = 0
        for ci in labels:
            pi =  ci / m
            result += (pi * self.log2(pi))
        return -result

    def info_x(self, T, subsets):
        m = len(T)
        result = 0
        for Ti in subsets:
            mi = len(Ti)
            result += ((mi/m)*(self.info(Ti)))
        return result

    def split_info(self, T, subsets):
        m = len(T)
        result = 0
        for Ti in subsets:
            mi = len(Ti)
            result += ((mi/m)*self.log2(mi/m))
        return -result

    def gain(self, T, subsets):
        return self.info(T) - self.info_x(T, subsets)

    def gain_ratio(self, T, subsets):
        a = self.gain(T, subsets)
        b = self.split_info(T, subsets)
        if abs(a - b) <= 10**-10 or b == 0:
            return 0
        return a/b

    def best_threshold(self, T, attribute):
        # v = list(T[attribute].quantile([.2, .4, .6, .8]))
        v = [T[attribute].mean()]
        max_value = -1
        max_thres = None
        for t in v:
            gr = self.gain_ratio(T, [T[T[attribute] <= t], T[T[attribute] > t]])
            if gr > max_value:
                max_value = gr
                max_thres = t
        return max_thres
        
    def least_attribute_gains(self, columns, top_count=10):
        gains = self.split_attribute(self.T, X.columns, is_find_top=True)
        top = sorted(gains)[-1:-min([top_count, len(columns)]):-1]
        for x in top:
            gains.pop(x)
        return list(gains.values())
        
        
    def split_attribute(self, T, attributes, is_find_top=False):
        best_value = -1
        best_attrs = None
        best_thres = None
        all_gains = {}
        splitted = {}
        for attribute in attributes:
            threshold = None
            subsets = {}
            if (T[attribute].dtype in ["int64", "float64"]):
                threshold = self.best_threshold(T, attribute)
                subsets["<="+str(threshold)] = (T[T[attribute] <= threshold])
                subsets[">"+str(threshold)] = (T[T[attribute] > threshold])
            else:
                for vk in self.T[attribute].unique():
                    subsets[vk] = T[T[attribute] == vk]
            gr = self.gain_ratio(T, list(subsets.values()))
            if gr > best_value:
                best_value = gr
                best_attrs = attribute
                best_thres = threshold
                splitted = subsets
            all_gains[round(gr*10000)] = attribute
        if is_find_top:
            return all_gains
        return best_attrs, best_thres, splitted, best_value
    
    def is_pure(self, T):
        values = T[self.target].unique()
        if len(values) == 1:
            return True
        return False
    
    def same_class(self, T):
        return T[self.target].unique()[0]
    
    def plurality_value(self, T):
        values = T[self.target].value_counts()
        max_value = values.max()
        values_that_max = (values == max_value).index
        return values_that_max[random.randrange(len(values_that_max))]
        
    def fit(self, X, y):
        self.T = pd.concat([X, y], axis=1)
        self.target = y.name
        dropped = self.least_attribute_gains(X.columns)
        print(self.T.shape)
        self.T = self.T.drop(columns=dropped)
        print(self.T.shape)
        # self.tree = self.dtl(self.T.copy(), list(X.columns), self.T.copy())
        
    def dtl(self, examples, attributes, parent_examples):
        print(len(attributes))
        if len(examples) == 0:
            return Tree(True, self.plurality_value(parent_examples), None, 0)
        elif self.is_pure(examples):
            return Tree(True, self.same_class(examples), None, 0)
        elif len(attributes) == 0:
            return Tree(True, self.plurality_value(examples), None, 0)
        else:
            best_attr, best_thres, splitted, gr = self.split_attribute(examples, attributes)
            attributes.remove(best_attr)
            tree = Tree(False, best_attr, best_thres, gr)
            for branch in splitted.keys():
                tree.add(branch, self.dtl(splitted[branch], attributes, examples))
            return tree

    def predict(self, X):
        predictions = []
        for i in X.index:
            temp = X.loc[i].to_dict()
            predictions.append(self.tree.traverse(temp))
        return predictions
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

data = pd.read_csv('data_all_3.csv')
X, y = split(data, "Target")
clf = DecisionTreeClassifier()
clf.fit(X, y)
res = 0
for v in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = tree.DecisionTreeClassifier()
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    res += clf.score(X_test, y_test)
print(res/10)