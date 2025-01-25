import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_percentage_error


def load_house_attributes(inputPath):
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df['zipcode'].value_counts().tolist()

    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    return train, test


def preprocess_house_attribute(train, test):
    continuous = ["bedrooms", "bathrooms", "area"]
    sc = StandardScaler()
    trainContinuous = sc.fit_transform(train[continuous])
    testContinuous = sc.fit_transform(test[continuous])

    encoder = OneHotEncoder(sparse_output=False)
    trainCategorical = encoder.fit_transform(np.array(train["zipcode"]).reshape(-1, 1))
    testCategorical = encoder.fit_transform(np.array(test["zipcode"]).reshape(-1, 1))

    #encoder = LabelBinarizer()
    #trainCategorical = encoder.fit_transform(train["zipcode"])
    #testCategorical = encoder.fit_transform(test["zipcode"])

    trainX = np.hstack([trainContinuous, trainCategorical])
    testX = np.hstack([testContinuous, testCategorical])

    maxPrice = train["price"].max()
    trainY = train["price"] / maxPrice
    testY = test["price"] / maxPrice

    return trainX, testX, trainY, testY



train, test = load_house_attributes(r"Week4\Datasets\HousesInfo.txt")

trainX, testX, trainY, testY = preprocess_house_attribute(train, test)


# ------------------------------- Linear Regression -------------------------
model = LinearRegression()
model.fit(trainX, trainY)

preds = model.predict(testX)

# Mean Absolute Percentage Error
diff = preds - testY
percentDiff = (diff/testY)
absPercentDiff = np.abs(percentDiff)
MAPE = np.mean(absPercentDiff)

print(MAPE)

mape = mean_absolute_percentage_error(testY, preds)
print(mape)


# ----------------------------- SGD Regressor ------------------------------
model2 = SGDRegressor(tol=0.000001, random_state=42)
model2.fit(trainX, trainY)

preds = model2.predict(testX)

mape = mean_absolute_percentage_error(testY, preds)
print(mape)
