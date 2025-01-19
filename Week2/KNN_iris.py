import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss


def load_data():
    dataset = pd.read_csv(r"Week2\Week2-Datasets\iris.data", header=None,
                        names=["sepal length", "sepal width",
                                "petal length", "petal width", "label"])

    #print(dataset.head(5))

    rows, cols = dataset.shape
    #print(f"the number of samples is {rows} and the number of features is {cols-1}")

    data = dataset.iloc[:, :4]
    label = dataset.iloc[:, 4]
    #print(label.head(5))

    # splitinf data to test and train
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    #print(y_test.shape)

    return x_train, x_test, y_train, y_test


def training():
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train)

    return clf


def results():
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"accuracy: {acc}")
    loss = zero_one_loss(y_test, y_pred) *100
    print(f"loss: {loss}")




x_train, x_test, y_train, y_test = load_data()

clf = training()

results()




