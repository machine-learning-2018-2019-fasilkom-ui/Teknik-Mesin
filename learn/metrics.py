import numpy as np


def confusion_matrix(y_true, y_pred, show_label=False):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    domain = sorted(list(set(y_true)))
    c = len(domain)
    index = {}
    for i in range(c):
        index[domain[i]] = i
    matrix = np.zeros((c, c))
    m = len(y_true)
    for i in range(m):
        rows = index[y_true[i]]
        cols = index[y_pred[i]]
        matrix[rows, cols] += 1
    if show_label:
        return matrix, domain
    return matrix


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    all_sum = np.sum(matrix)
    diag_sum = np.sum(np.diag(matrix))
    return diag_sum / all_sum
