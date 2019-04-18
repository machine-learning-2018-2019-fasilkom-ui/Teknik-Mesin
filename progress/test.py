import pandas as pd
import time
from learn.classifier import DT, KNN, RandomForest
from learn.decomposition import PCA
from learn.metrics import confusion_matrix, accuracy
from learn.model_selection import train_test_split

df = pd.read_csv('data\genres.csv')

col_label = 'Target'
X = df.drop(columns=col_label)
y = df[col_label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

algos = {
    'knn' : KNN(4),
    'dt' : DT(),
    'rf' : RandomForest(n_estimators=100),
}

clf = algos['knn']
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#
# print("-"*30)
#
# clf = algos['dt']
#
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
#
# print("-"*30)
#
# clf = algos['rf']
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
