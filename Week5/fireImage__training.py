import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_data():

    data_list = []
    labels = []

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

    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_data()


clf = KNeighborsClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

#dump(clf, "fireDetector_model.z")

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, pos_label="fire") * 100
recall = recall_score(y_test, y_pred, pos_label="fire") * 100
f_score = f1_score(y_test, y_pred, pos_label="fire") * 100

print(f"accuracy: {accuracy}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f_score: {f_score}")










