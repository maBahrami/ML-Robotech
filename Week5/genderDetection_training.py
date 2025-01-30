import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from mtcnn import MTCNN

detector = MTCNN()

data = []
labels = []

def faceDetector(img):
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

out = detector.detect_faces(rgb_img)[0]
x, y, w, h = out["box"]



for item in glob.glob(r"Week5\reference\Datasets\Gender\*\*"):
    img = cv2.imread(item)
    face = faceDetector(img)