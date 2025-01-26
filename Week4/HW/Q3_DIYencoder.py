import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


cols = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
        "deg-malig", "breast", "breast-quad", "irradiat"]
df = pd.read_csv(r"Week4\HW\Datasets\breast-cancer.data", header=None, names=cols)
#print(df.head())
#print(df.shape)

df = df.loc[(df.iloc[:, 5])!="?"]
#print(df.shape)
df = df.loc[(df.iloc[:, 8])!="?"]
#print(df.shape)

train, test = train_test_split(df, test_size=0.2, random_state=123)


def encOneHot_DIY(featureName):
    categories = train[featureName].value_counts().keys().tolist()
    #print(categories)
    #print(train[featureName].value_counts().tolist())
    a = np.zeros([len(train), len(categories)]) # train
    b = np.zeros([len(test), len(categories)])  # test
    #print(a.shape)

    j = 0
    for i in train[featureName]:
        if i in categories:
            index = categories.index(i)
            a[j, index] = 1
        else:
            continue
        j += 1

    j = 0
    for i in test[featureName]:
        if i in categories:
            index = categories.index(i)
            b[j, index] = 1
        else:
            continue
        j += 1

    #print(a)
    #print(train[featureName])

    return a, b


def allOneHot_DIY():
    features = ["menopause", "tumor-size", "inv-nodes", "node-caps",
                "deg-malig", "breast", "breast-quad", "irradiat"]

    trainX, testX = encOneHot_DIY("age")

    for i in features:
        a, b = encOneHot_DIY(i)

        trainX = np.hstack([trainX, a])
        #print(i, trainOneHot.shape)
        #print(encOneHot.get_feature_names_out([i]))
        testX = np.hstack([testX, b])

    return trainX, testX


# ----------------------------------------------------------------
trainX, testX = allOneHot_DIY()

trainY = train["class"]
testY = test["class"]


# ------------------------ Logistic Regression ------------------

clf = LogisticRegression()
clf.fit(trainX, trainY)

y_pred = clf.predict(testX)

acc = accuracy_score(y_pred, testY)
print(acc)
#print(clf.coef_)

