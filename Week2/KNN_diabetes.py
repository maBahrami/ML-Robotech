import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv(r"Week2\Week2-Datasets\diabetes.csv")

# print(dataset.head())

zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


#print(dataset["SkinThickness"])
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.nan)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.nan, mean)
#print(dataset["SkinThickness"])


x = dataset.iloc[:, :8]
y = dataset.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_pred, y_test) * 100

print(acc)









