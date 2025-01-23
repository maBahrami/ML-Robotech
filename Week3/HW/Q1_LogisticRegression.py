import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"Week3\HW\Datasets\Q1_admission_result.csv")
#print(dataset.head())

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

"""
plt.subplot(2, 1, 1)
plt.hist(x_train.iloc[:, 0])
"""
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""
plt.subplot(2, 1, 2)
plt.hist(x_train[:, 0])
plt.show()
"""

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
clf = SGDClassifier(loss="log_loss", random_state=123)
clf.fit(x_train, y_train)

print(clf.coef_)
print(clf.intercept_)
print(clf.n_iter_)

y_pred = clf.predict(x_test)
acc = accuracy_score(y_pred, y_test) * 100

print(f"acc of SGD Classification: {acc}\n\n\n")

#grade = int(input("enter the grade: "))

for i in range(45, 65, 1):
    
    grade = pd.DataFrame(np.array([[i]]), columns=['score'])
    grade = sc.transform(grade)

    output = clf.predict(grade)
    print(f"grade: {i}, admission: {output}, probabilities: {(clf.predict_proba(grade))[0]}\n")


"""
                ** The effect of normalizing on SGD  **

The drastic improvement in accuracy from 40% to 100% after normalization suggests that
 your feature's scale was significantly 
 affecting the optimization process of SGDClassifier. Here’s why:

1️⃣ SGDClassifier is Sensitive to Feature Scale
SGD (Stochastic Gradient Descent) relies on gradient updates, and if your feature values
are on a very large or very small scale, the updates can become inefficient.

Without normalization, large feature values can lead to large gradient steps, 
causing unstable learning and poor convergence. With normalization, feature values 
are brought to a standard scale (e.g., mean = 0, std = 1), 
ensuring smoother and more effective updates.

2️⃣ Decision Boundary Distortion
When features are not normalized, the decision boundary might not be properly 
aligned with the actual data distribution. This could lead to misclassification.

Normalization ensures that the classifier learns a well-defined decision boundary, 
improving classification performance.

"""