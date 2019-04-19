import numpy as np
import math
import random
import operator


# --------------------------------------------------------------------------------
#                          C4.5 Decision Tree Classifier
# --------------------------------------------------------------------------------

class Func:
    def __init__(self, rel, label):
        self.rel = rel
        self.label = label

    def func(self, x):
        if self.rel == "==":
            return self.label == x
        elif self.rel == ">":
            return x > self.label
        return x <= self.label

    def __repr__(self):
        return str(self.rel) + str(self.label)


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
            selected = None
            for branch in curr_node.branches:
                if branch.func(value):
                    selected = curr_node.branches.get(branch)
                    break
            return self.traverse_helper(selected, inputs)

    def tree_struct(self):
        self.tree_struct_help(self, 0)

    def tree_struct_help(self, curr, level, thr=None):
        print("|\t" * level + "+", "" if thr is None else "[" + str(thr) + "]", curr, curr.distribution)
        for key in curr.branches:
            subtree = curr.branches.get(key)
            self.tree_struct_help(subtree, level + 1, key)

    def __repr__(self):
        if self.is_leaf:
            return str(self.label) + " (LEAF)"
        return str(self.label)


class DT:
    def __init__(self):
        self.tree = None
        self.X = None
        self.y = None
        self.row_idxs = None
        self.col_idxs = None
        self.domain = {}

    def log2(self, k):
        return 0 if k == 0 else math.log2(k)

    def value_counts(self, T_index):
        values = {}
        for i in T_index:
            curr_value = self.y[i]
            freq = values[curr_value] if curr_value in values else 0
            values[curr_value] = freq + 1
        return values

    def get_by_value(self, T_index, col_idx, value):
        idxs = []
        for i in T_index:
            xi = self.X[i, col_idx]
            if value.func(xi):
                idxs.append(i)
        return idxs

    def info(self, T_index):
        m, result = len(T_index), 0
        labels = list(self.value_counts(T_index).values())
        for ci in labels:
            pi = ci / m
            result += (pi * self.log2(pi))
        return -result

    def info_x(self, T_index, subsets):
        m, result = len(T_index), 0
        for Ti_index in subsets:
            mi = len(Ti_index)
            result += ((mi / m) * (self.info(Ti_index)))
        return result

    def split_info(self, T_index, subsets):
        m, result = len(T_index), 0
        for Ti_index in subsets:
            mi = len(Ti_index)
            result += ((mi / m) * self.log2(mi / m))
        return -result

    def gain(self, T_index, subsets):
        return self.info(T_index) - self.info_x(T_index, subsets)

    def gain_ratio(self, T_index, subsets):
        a = self.gain(T_index, subsets)
        b = self.split_info(T_index, subsets)
        return 0 if abs(a - b) <= 10 ** -10 or b == 0 else a / b

    def majority_class(self, T_index):
        values = self.value_counts(T_index)
        if len(values) == 1:
            return list(values.keys())[0], True
        maks = max(list(values.values()))
        maj_class = []
        for value in values:
            if values[value] == maks:
                maj_class.append(value)
        return maj_class[random.randrange(len(maj_class))], False

    def get_distribution(self, T_index):
        return self.value_counts(T_index)

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
        f = f / N
        return f, N, majority

    def e(self, f, N, z=0.69):
        return (f + z ** 2 / (2 * N) + z * np.sqrt(f / N - f ** 2 / N + z ** 2 / (4 * N ** 2))) / (1 + z ** 2 / N)

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

    def is_categorical(self, col_idx):
        return type(self.X[0, col_idx]) is str

    def best_threshold(self, T_index, col_idx):
        candidates = sorted(self.X[T_index, col_idx])
        n = len(candidates)
        result = [candidates[int(_ * n - .5)] for _ in [.3, .5, .7]]
        result.append(np.mean(self.X[T_index, col_idx]))
        best = None
        maks = -1
        for res in result:
            func1, func2 = Func("<=", res), Func(">", res)
            subsets1 = self.get_by_value(T_index, col_idx, func1)
            subsets2 = self.get_by_value(T_index, col_idx, func2)
            gain = self.gain(T_index, [subsets1, subsets2])
            if gain > maks:
                best = res
                maks = gain
        return best

    def split_attribute(self, T_index, col_idxs, is_find_top=False):
        best_value, best_attrs, best_thres, splitted = -1, None, None, {}
        gains = {}
        for col_idx in col_idxs:
            threshold = None
            subsets = {}
            if col_idx in self.domain:
                for vk in self.domain[col_idx]:
                    func = Func("==", vk)
                    subsets[func] = self.get_by_value(T_index, col_idx, func)
            else:
                threshold = self.best_threshold(T_index, col_idx)
                func1, func2 = Func("<=", threshold), Func(">", threshold)
                subsets[func1] = self.get_by_value(T_index, col_idx, func1)
                subsets[func2] = self.get_by_value(T_index, col_idx, func2)
            gr = self.gain_ratio(T_index, list(subsets.values()))
            gains[round(gr * 10000)] = col_idx
            if gr > best_value:
                best_value, best_attrs, best_thres, splitted = gr, col_idx, threshold, subsets
        if is_find_top:
            return gains
        return best_attrs, best_thres, splitted, best_value

    def dtl(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            maj_class, is_pure = self.majority_class(parent_examples)
            dist = self.get_distribution(parent_examples)
            return Tree(True, maj_class, None, -1, dist)
        maj_class, is_pure = self.majority_class(examples)
        dist = self.get_distribution(examples)
        if is_pure:
            return Tree(True, maj_class, None, -2, dist)
        elif len(attributes) == 0:
            return Tree(True, maj_class, None, -3, dist)
        else:
            best_attr, best_thres, splitted, gr = self.split_attribute(examples, attributes)
            attributes = list(attributes)
            attributes.remove(best_attr)
            tree = Tree(False, best_attr, best_thres, gr, dist)
            for branch in splitted.keys():
                tree.add(branch, self.dtl(splitted[branch], attributes, examples))
            return tree

    def initialize(self, X, y):
        self.X = X
        self.y = y
        if (self.row_idxs is None) and (self.col_idxs is None):
            self.row_idxs = [i for i in range(len(X))]
            self.col_idxs = [j for j in range(len(X[0]))]
            for col_idx in self.col_idxs:
                if self.is_categorical(col_idx):
                    self.domain[col_idx] = set()
                    for row_idx in self.row_idxs:
                        self.domain[col_idx].add(self.X[row_idx, col_idx])

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.initialize(X, y)
        self.tree = self.dtl(self.row_idxs, list(self.col_idxs), self.row_idxs)
        self.tree = self.pruning(self.tree)

    def print_tree_struct(self):
        return self.tree.tree_struct()

    def to_dict(self, xi):
        result = {}
        for i in self.col_idxs:
            result[i] = xi[i]
        return result

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for xi in X:
            inputsi = self.to_dict(xi)
            predictions.append(self.tree.traverse(inputsi))
        return predictions


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
#                               K-Nearest Neighbors
# --------------------------------------------------------------------------------

class KNN:

    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k

    def euclideanDistance(self, point1, point2):
        diff = point1 - point2
        return math.sqrt(sum(diff ** 2))

    def get_neighbors(self, test_point):
        distances = []
        for i in range(len(self.X)):
            train_point = self.X[i]
            distance = self.euclideanDistance(test_point, train_point)
            distances.append((i, distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors_index = []
        for i in range(self.k):
            neighbors_index.append(distances[i][0])
        return neighbors_index

    def get_majority(self, neighbors_index):
        class_votes = {}
        for i in neighbors_index:
            label = self.y[i]
            class_votes[label] = class_votes.get(label, 0) + 1
        sorted_votes = list(class_votes.items())
        sorted_votes.sort(key=operator.itemgetter(1))
        return sorted_votes[-1][0]

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predicted = []
        for xi in X:
            neighbors_index = self.get_neighbors(xi)
            predicted.append(self.get_majority(neighbors_index))
        return predicted


# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------
#                          Random Forest Classifier
# --------------------------------------------------------------------------------

class RandomForest:

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.X = None
        self.y = None
        self.forest = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.forest = self.build_forest()

    def build_forest(self):
        forest = []
        m, n = self.X.shape
        k = m
        l = int(math.sqrt(n))
        for i in range(self.n_estimators):
            dt = DT()
            dt.row_idxs = np.random.choice(m, k)
            dt.col_idxs = list(set(np.random.choice(n, l)))
            dt.fit(self.X, self.y)
            forest.append(dt)
        return forest

    def predict(self, X):
        X = np.array(X)
        y_pred = []
        for xi in X:
            predictions = {}
            for dt in self.forest:
                predicted = dt.predict(xi.reshape(1, -1))[0]
                predictions[predicted] = predictions.get(predicted, 0) + 1
            sorted_votes = list(predictions.items())
            sorted_votes.sort(key=operator.itemgetter(1))
            y_pred.append(sorted_votes[-1][0])
        return y_pred

# --------------------------------------------------------------------------------
