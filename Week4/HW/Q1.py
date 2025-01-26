import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import SGDClassifier


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


features = ["menopause", "tumor-size", "inv-nodes", "node-caps",
            "deg-malig", "breast", "breast-quad", "irradiat"]

encOneHot = OneHotEncoder(sparse_output=False)
trainOneHot = encOneHot.fit_transform(np.array(train["age"]).reshape(-1, 1))
#print(encOneHot.get_feature_names_out(["age"]))
#print(trainOneHot.shape)
testOneHot = encOneHot.transform(np.array(test["age"]).reshape(-1, 1))

for i in features:
    trainOneHot = np.hstack([trainOneHot, encOneHot.fit_transform(np.array(train[i]).reshape(-1, 1))])
    #print(i, trainOneHot.shape)
    #print(encOneHot.get_feature_names_out([i]))
    testOneHot = np.hstack([testOneHot, encOneHot.transform(np.array(test[i]).reshape(-1, 1))])

#print(trainOneHot.shape)