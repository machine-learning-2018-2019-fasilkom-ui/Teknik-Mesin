from tree.DecisionTree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('iris.csv')
X = data.drop(columns="species")
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.print_tree_struct()
y_predict = clf.predict(X_test)
accuracy_score(y_test, y_predict)
confusion_matrix(y_test, y_predict)