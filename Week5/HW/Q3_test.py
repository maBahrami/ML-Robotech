import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import glob
from mtcnn import MTCNN
from joblib import load

detector = MTCNN()

clf = load(r"Week5\HW\Q3_model.z")

def faceDetector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out["box"]
    
        return x, y, w, h

    except:
        pass


for i, item in enumerate(glob.glob(r"Week5/HW/reference/Dataset/Q3/test_data/*/*")):
    img = cv2.imread(item)
    F = faceDetector(img)
    if F is None:
        continue
    else:
        x, y, w, h = F
        face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (32, 32))
    face = face.flatten()
    face = face / 255

    out = clf.predict(np.array([face]))[0]
    print(out)

    cv2.putText(img, out, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()






