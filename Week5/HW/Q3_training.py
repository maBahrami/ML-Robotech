import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from mtcnn import MTCNN
from sklearn.metrics import accuracy_score, recall_score, precision_score
from joblib import dump

detector = MTCNN()

data = []
labels = []

def faceDetector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out["box"]

        return img[y:y+h, x:x+w]
    
    except:
        pass


for i, item in enumerate(glob.glob(r"Week5\HW\reference\Dataset\Q3\smile_dataset\*\*")):
    img = cv2.imread(item)
    face = faceDetector(img)

    if face is None:    continue

    face = cv2.resize(face, (32, 32))
    face = face.flatten()
    face = face / 255

    data.append(face)

    label = item.split("\\")[-2]
    labels.append(label)

    if i % 100 == 0: print(f"[INFO]: {i}/3686 processed")


data = np.array(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = SGDClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(f"accuracy: {acc}")


dump(clf, "smileDetection_Model.z")