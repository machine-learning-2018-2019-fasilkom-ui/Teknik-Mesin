import random
import numpy as np
import pandas as pd

def split(df, label):
    X = data.drop(columns=label)
    y = data[label]
    return X, y

class Tree:
    def __init__(self, is_leaf, label, threshold, gain, distribution):
        self.branches = {}
        self.is_leaf = is_leaf
        self.label = label
        self.threshold = threshold
        self.gain = gain
        self.distribution = distribution

    def add(self, branch, subtree):
        self.branches[branch] = subtree

    def traverse(self, inputs):
        return self.traverse_helper(self, inputs)

    def traverse_helper(self, curr_node, inputs):
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
            return self.traverse_helper(temp, inputs)
    
    def tree_struct(self):
        self.tree_struct_help(self, 0)

    def tree_struct_help(self, curr, level, thr=None):
        print("|\t" * level + "+", "" if thr is None else "["+str(thr)+"]", curr)
        for key in curr.branches:
            subtree = curr.branches.get(key)
            self.tree_struct_help(subtree, level + 1, key)

    def __repr__(self):
        if self.is_leaf:
            return self.label + " (LEAF)"
        return self.label

class DecisionTreeClassifier:

    def __init__(self, max_features=0):
        self.T = None
        self.target = None
        self.max_features = max_features
        self.tree = None
    
    def print_tree_struct(self):
        return self.tree.tree_struct()

    def log2(self, k):
        return 0 if k == 0 else np.log2(k)

    def info(self, T):
        m = len(T)
        labels = T[self.target].value_counts()
        result = 0
        for ci in labels:
            pi = ci / m
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
        return 0 if abs(a - b) <= 10**-10 or b == 0 else a/b

    def best_threshold(self, T, attribute):
        v = [T[attribute].mean()]
        max_value, max_thres = -1, None
        for t in v:
            gr = self.gain_ratio(T, [T[T[attribute] <= t], T[T[attribute] > t]])
            if gr > max_value:
                max_value, max_thres = gr, t
        return max_thres

    def split_attribute(self, T, attributes):
        best_value, best_attrs, best_thres, splitted = -1, None, None, {}
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
                best_value, best_attrs, best_thres, splitted = gr, attribute, threshold, subsets
        return best_attrs, best_thres, splitted, best_value

    def is_pure(self, T):
        values = T[self.target].unique()
        return len(values) == 1

    def same_class(self, T):
        return T[self.target].unique()[0]

    def plurality_value(self, T):
        values = T[self.target].value_counts()
        max_value = values.max()
        values_that_max = (values == max_value).index
        return values_that_max[random.randrange(len(values_that_max))]

    def get_distribution(self, T):
        return dict(T[self.target].value_counts())

    def get_fn(self, dist):
        maks, f, N = -1, 0, 0
        majority = []
        for value in dist:
            if dist[value] > maks:
                maks = dist[value]
            else:
                f += dist[value]
            if dist[value] == maks:
                majority.append(value)
            N += dist[value]
        f = f/N
        return f, N, majority

    def e(self, f, N, z=0.69):
        return (f + z**2/(2*N) + z*np.sqrt(f/N - f**2/N + z**2/(4*N**2)))/(1 + z**2/N)

    def pruning(self, curr_tree):
        e = 0
        for subtree in list(curr_tree.branches.values()):
            if self.pruning(subtree).is_leaf:
                f, N, majority = self.get_fn(subtree.distribution)
                e += (self.e(f, N))
        f_curr, N_curr, majority_curr = self.get_fn(curr_tree.distribution)
        e_curr = self.e(f_curr, N_curr)
        if e_curr < e:
            curr_tree.is_leaf = True
            curr_tree.label = majority_curr[random.randrange(len(majority_curr))]
            curr_tree.branches = {}
        return curr_tree

    def fit(self, X, y):
        self.T = pd.concat([X, y], axis=1)
        self.target = y.name
        self.tree = self.dtl(self.T.copy(), list(X.columns), self.T.copy())
        self.pruning(self.tree)

    def dtl(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return Tree(True, self.plurality_value(parent_examples), None, -1, self.get_distribution(parent_examples))
        elif self.is_pure(examples):
            return Tree(True, self.same_class(examples), None, -2, self.get_distribution(examples))
        elif len(attributes) == self.max_features:
            return Tree(True, self.plurality_value(examples), None, -3, self.get_distribution(examples))
        else:
            best_attr, best_thres, splitted, gr = self.split_attribute(examples, attributes)
            attributes = list(attributes)
            attributes.remove(best_attr)
            tree = Tree(False, best_attr, best_thres, gr, self.get_distribution(examples))
            for branch in splitted.keys():
                tree.add(branch, self.dtl(splitted[branch], attributes, examples))
            return tree

    def predict(self, X):
        predictions = []
        for i in X.index:
            temp = X.loc[i].to_dict()
            predictions.append(self.tree.traverse(temp))
        return predictions
