import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import SGDClassifier
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


def allOneHot(train, test):
    features = ["menopause", "tumor-size", "inv-nodes", "node-caps",
                "deg-malig", "breast", "breast-quad", "irradiat"]

    enc = OneHotEncoder(sparse_output=False)
    trainX = enc.fit_transform(np.array(train["age"]).reshape(-1, 1))
    #print(encOneHot.get_feature_names_out(["age"]))
    #print(trainOneHot.shape)
    testX = enc.transform(np.array(test["age"]).reshape(-1, 1))

    for i in features:
        trainX = np.hstack([trainX, enc.fit_transform(np.array(train[i]).reshape(-1, 1))])
        #print(i, trainOneHot.shape)
        #print(encOneHot.get_feature_names_out([i]))
        testX = np.hstack([testX, enc.transform(np.array(test[i]).reshape(-1, 1))])

    return trainX, testX


def allOrdinal(train, test):
    features = ["menopause", "tumor-size", "inv-nodes", "node-caps",
                "deg-malig", "breast", "breast-quad", "irradiat"]

    enc = OrdinalEncoder()
    trainX = enc.fit_transform(np.array(train["age"]).reshape(-1, 1))
    #print(encOneHot.get_feature_names_out(["age"]))
    #print(trainOneHot.shape)
    testX = enc.transform(np.array(test["age"]).reshape(-1, 1))

    for i in features:
        trainX = np.hstack([trainX, enc.fit_transform(np.array(train[i]).reshape(-1, 1))])
        #print(i, trainOneHot.shape)
        #print(encOneHot.get_feature_names_out([i]))
        testX = np.hstack([testX, enc.transform(np.array(test[i]).reshape(-1, 1))])

    return trainX, testX





def OneHot_Ordinal(train, test):

    ordinalFeatures = ["tumor-size", "inv-nodes", "deg-malig"]
    nominalFeatures = ["menopause", "node-caps", "breast", "breast-quad", "irradiat"]
    

    enc = OrdinalEncoder()
    trainX = enc.fit_transform(np.array(train["age"]).reshape(-1, 1))
    #print(enc.get_feature_names_out(["age"]))
    #print(trainOneHot.shape)
    testX = enc.transform(np.array(test["age"]).reshape(-1, 1))
    #print(testX[0:10])

    for i in ordinalFeatures:
        trainX = np.hstack([trainX, enc.fit_transform(np.array(train[i]).reshape(-1, 1))])
        #print(i, trainOneHot.shape)
        #print(enc.get_feature_names_out([i]))
        testX = np.hstack([testX, enc.transform(np.array(test[i]).reshape(-1, 1))])
        #print(testX[0:10])

    #print(testX.shape)

    enc2 = OneHotEncoder(sparse_output=False)
    for i in nominalFeatures:
        trainX = np.hstack([trainX, enc2.fit_transform(np.array(train[i]).reshape(-1, 1))])
        #print(i, trainOneHot.shape)
        #print(encOneHot.get_feature_names_out([i]))
        testX = np.hstack([testX, enc2.transform(np.array(test[i]).reshape(-1, 1))])
        #print(testX[0:10])

    return trainX, testX








# ----------------------------------------------------------------
#trainX, testX = allOrdinal(train, test)

#trainX, testX = allOneHot(train, test)

trainX, testX = OneHot_Ordinal(train, test)

#print(testX.shape)

trainY = train["class"]
testY = test["class"]

# -------------------------- SGD Classifier ---------------------
clf = SGDClassifier(loss="log_loss", tol=1e-5, random_state=123)
clf.fit(trainX, trainY)

y_pred = clf.predict(testX)
acc = accuracy_score(y_pred, testY)
#print(acc)

# ------------------------ Logistic Regression ------------------
clf = LogisticRegression()
clf.fit(trainX, trainY)

y_pred = clf.predict(testX)

acc = accuracy_score(y_pred, testY)
print(acc)
print(clf.coef_)

