import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras import models

EPOCHS = 100
batchSize = 32


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



def neural_network():
    net = models.Sequential([
                                #layers.Flatten(input_shape=(32, 32, 3)),
                                layers.Dense(20, activation="relu"),
                                layers.Dense(8, activation="relu"),
                                layers.Dense(1, activation="linear")
                            ])

    net.summary()

    net.compile(optimizer="SGD",
                loss="mse")

    H = net.fit(x_train, y_train, batch_size=batchSize, epochs=EPOCHS, validation_data=(x_test, y_test))

    loss = net.evaluate(x_test, y_test)
    print(f"loss: {loss}")

    #net.save("housePrice_NN.h5")

    # net = models.load_model("fire_detection_NeuralNetwork.h5")

    return net




train, test = load_house_attributes(r"Week4\Datasets\HousesInfo.txt")

x_train, x_test, y_train, y_test = preprocess_house_attribute(train, test)

model = neural_network()



preds = model.predict(x_test)
diff = preds.flatten() - y_test
percentDiff = np.abs((diff/y_test)*100)
mean = np.mean(percentDiff)
std = np.std(percentDiff)
print(f"mean: {mean}, std:{std}")

