import numpy as np
import pandas as pd
import random

data = pd.read_csv('tennis.csv')
data["Day"] = data["Day"].astype(str)

class Tree:
    def __init__(self, is_leaf, label, threshold=None):
        self.branches = {}
        self.is_leaf = is_leaf
        self.label = label
        self.threshold = threshold

    def add(self, branch, subtree):
        self.branches[branch] = subtree

    def traverse(self, inputs):
        return self.traverseHelper(self, inputs)

    def traverseHelper(self, curr_node, inputs):
        if curr_node.is_leaf:
            return curr_node.label
        else:
            value = inputs.get(curr_node.label)
            functions = list(curr_node.branches.keys())
            for function in functions:
                if function(value):
                    temp = curr_node.branches.get(function)
                    return self.traverseHelper(temp, inputs)
            
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

    def get_threshold(self, T, attribute):
        column = T[attribute].sort_values()
        v = list(column)
        thresholds = []
        if (column.dtype == "float64"):
            for i in range(len(v)-1):
                t = (v[i] + v[i+1])//2
                thresholds.append(t)
        elif (column.dtype == "int64"):
            thresholds = v
        return sorted(list(set(thresholds)))

    def best_threshold(self, T, attribute):
        thresholds = self.get_threshold(T, attribute)
        max_value = -1
        max_thres = None
        for threshold in thresholds:
            gr = self.gain_ratio(T, [T[T[attribute] <= threshold], T[T[attribute] > threshold]])
            if gr > max_value:
                max_value = gr
                max_thres = threshold
        return max_thres
        
    def split_attribute(self, T, attributes):
        best_value = -1
        best_attrs = None
        best_thres = None
        splitted = []
        for attribute in attributes:
            threshold = None
            subsets = []
            if (T[attribute].dtype in ["int64", "float64"]):
                threshold = self.best_threshold(T, attribute)
                subsets.append(T[T[attribute] <= threshold])
                subsets.append(T[T[attribute] > threshold])
            else:
                for vk in T[attribute].unique():
                    subsets.append(T[T[attribute] == vk])
            gr = self.gain_ratio(T, subsets)
            if gr > best_value:
                best_value = gr
                best_attrs = attribute
                best_thres = threshold
                splitted = subsets
        return best_attrs, best_thres, splitted
    
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
        self.tree = self.dtl(self.T.copy(), list(X.columns), self.T.copy())
        
    def dtl(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return Tree(True, self.plurality_value(parent_examples))
        elif self.is_pure(examples):
            return Tree(True, self.same_class(examples))
        elif len(attributes) == 0:
            return Tree(True, self.plurality_value(examples))
        else:
            best_attr, best_thres, splitted = self.split_attribute(examples, attributes)
            attributes.remove(best_attr)
            tree = Tree(False, best_attr, best_thres)
            for child in splitted:
                value = None
                if best_thres is None:
                    value = lambda x : x == child[best_attr].unique()[0]
                else:
                    maks = max(child[best_attr])
                    if maks <= best_thres:
                        value = lambda x : x <= best_thres
                    else:
                        value = lambda x : x > best_thres
                tree.add(value, self.dtl(child, attributes, examples))
            return tree
            
    def predict(self, inputs):
        return self.tree.traverse(inputs)
        
X = data.drop(["Decision", "Day"], axis=1)
y = data["Decision"]
clf = DecisionTreeClassifier()
clf.fit(X, y)
inputs = {
    "Day":1,
    "Outlook":"Sunny",
    "Temp.":85,
    "Humidity":85,
    "Wind":"Weak",
}
print(clf.predict(inputs))