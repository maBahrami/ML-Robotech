import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import glob
from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data = []
labels = []

for i, item in enumerate(glob.glob(r"Week5\HW\reference\Dataset\Q1\train\*\*")):
    img = cv2.imread(item)
  
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    img = img / 255
    data.append(img)

    label = item.split("\\")[-1].split(".")[0]
    labels.append(label)

    if i % 100 == 0:
        print(f"[INFO]  {i}/2000 processed")

data = np.array(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1234)

# ------------------------ KNN --------------------------
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test) * 100
print(f"acuracy of KNN: {acc}")

# ------------------------ SVM --------------------------
clf = SGDClassifier()
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test) * 100
print(f"acuracy of SVM: {acc}")

# ----------------- Logistic Regression --------------------------
clf = LogisticRegression(max_iter=3000)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = clf.score(x_test, y_test) * 100
print(f"acuracy of LR: {acc}")

precision = precision_score(y_test, y_pred, pos_label="cat") * 100
recall = recall_score(y_test, y_pred, pos_label="cat") * 100

print(f"precision score of LR: {precision}")
print(f"recall of LR: {recall}")

dump(clf, "Q1_model.z")