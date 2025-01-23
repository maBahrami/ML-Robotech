import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

dataset = pd.read_csv(r"Week3\HW\Datasets\Q1_admission_result.csv")
#print(dataset.head())

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print("\n\n------------------ Logistic Regression --------------------\n")
clf = LogisticRegression()
clf.fit(x_train, y_train)

print(clf.coef_)
print(clf.intercept_)
print(clf.n_iter_)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_pred, y_test) * 100

print(f"acc of Logistic Regressino: {acc}")

print("\n\n------------------ SGD Classification --------------------\n")
clf = SGDClassifier(loss="log_loss")
clf.fit(x_train, y_train)

print(clf.coef_)
print(clf.intercept_)
print(clf.n_iter_)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_pred, y_test) * 100

print(f"acc of SGD Classification: {acc}")

