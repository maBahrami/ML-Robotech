import cv2
import numpy as np
import glob
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data = []
labels = []

for i, item in enumerate(glob.glob(r"Week5\HW\reference\Dataset\Q1\test\*\*")):
    img = cv2.imread(item)
  
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    img = img / 255
    data.append(img)

    label = item.split("\\")[-1].split(".")[0].split(" ")[0].lower()
    labels.append(label)

    if i % 100 == 0:
        print(f"[INFO]  {i}/2000 processed")

data = np.array(data)

clf = load(r"Week5\HW\Q1_model.z")

out = clf.predict(data)
print(out)
print(labels)

acc = accuracy_score(out, labels) * 100
print(acc)