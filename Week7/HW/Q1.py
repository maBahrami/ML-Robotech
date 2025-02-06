import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


def load_data():
    dataset = pd.read_csv(r"Week7\HW\reference\Q1\fruit_data_with_colors.txt", sep='\t')
    #print(dataset.head())
    x = dataset.iloc[:, 3:]
    y = dataset.iloc[:, 1]
    #print(x)
    #print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return x_train, x_test, y_train, y_test


def training():
    net = models.Sequential([layers.Dense(16, activation="relu"),
                             layers.Dense(8, activation="relu"),
                             layers.Dense(4, activation="softmax")])
    
    net.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    
    H = net.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(x_test, y_test))
    
    return H

def show_plots():
    plt.style.use("ggplot")
    plt.plot(H.history["accuracy"], label = "train accuracy")
    plt.plot(H.history["val_accuracy"], label = "test accuracy")
    plt.plot(H.history["loss"], label = "train loss")
    plt.plot(H.history["val_loss"], label = "test loss")
    plt.xlabel("epochs")
    plt.ylabel("accuracy/loss")
    plt.title(f"batch size: {BATCH_SIZE}, epochs: {EPOCH}")
    plt.legend()
    plt.show()


BATCH_SIZE = 5
EPOCH = 500

x_train, x_test, y_train, y_test = load_data()
print(x_train.shape)

H = training()

show_plots()
