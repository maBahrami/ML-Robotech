import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers


def load_data():

    data_list = []
    labels = []

    le = LabelEncoder()

    for i, address in enumerate(glob.glob(r"Week5\reference\Datasets\fire_dataset\*\*")):
        #print(address)
        img = cv2.imread(address)
        #print(img.shape)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = img.flatten()
        #print(img.shape)
        data_list.append(img)

        label = address.split("\\")[-1].split(".")[0]
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO] {i}/{1000} processed.")


    data_list = np.array(data_list)

    x_train, x_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2, random_state=123)

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_data()


net = models.Sequential([
                            layers.Dense(20, activation="relu"),
                            layers.Dense(8, activation="relu"),
                            layers.Dense(2, activation="softmax")
                         ])

net.compile(optimizer="SGD",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

H = net.fit(x_train, y_train, batch_size=32, epochs=23, validation_data=(x_test, y_test))



