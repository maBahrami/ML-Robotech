import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data():
    dataset = pd.read_csv(r"Week2\HW\Datasets\Q1_fruit_data_with_colors.txt", sep="\t")
    #print(dataset.head())
    #print(dataset.info())

    x = dataset.iloc[:, 3:8]
    y = dataset.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    #print(y_train.shape)
    #print(x_test.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def training(kNum):
    clf = KNeighborsClassifier(n_neighbors=kNum)
    clf.fit(x_train, y_train)
    return clf


def results():
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_pred, y_test) * 100
    return acc



x_train, x_test, y_train, y_test = load_data()

# for different values of K in KNN
for i in range(1, 20, 2):
    
    clf = training(i)

    acc = round(results(), 2)

    print(f"{i} neighbors >> accuracy: {acc}")






