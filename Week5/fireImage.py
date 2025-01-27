import cv2
from sklearn.model_selection import train_test_split
import glob

data_list = []

for address in glob.glob(r"Week5\reference\Datasets\fire_dataset\*\*"):
    #print(address)
    img = cv2.imread(address)
    #print(img.shape)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.flatten()
    print(img.shape)
