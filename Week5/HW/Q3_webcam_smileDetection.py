import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
from mtcnn import MTCNN
from joblib import load


def faceDetector(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out["box"]
    
        return x, y, w, h

    except:
        pass

detector = MTCNN()

clf = load(r"Week5\HW\Q3_model.z")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, img = cap.read()

    if not ret:
        print("Failed to grab frame")
        break


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
    #print(out)

    if out == "neg":
        cv2.putText(img, out, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        cv2.putText(img, out, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)      

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
