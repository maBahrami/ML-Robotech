import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


dataset = pd.read_csv(r"Week5\HW\reference\Dataset\Q2\mnist_train.csv", header=None)
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

x = x / 255
#print(x.max(axis=1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)



#clf = KNeighborsClassifier(n_neighbors=11)
clf = SGDClassifier(loss="log_loss")
#clf = LogisticRegression()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_pred, y_test) * 100
print(f"accuracy: {acc}")