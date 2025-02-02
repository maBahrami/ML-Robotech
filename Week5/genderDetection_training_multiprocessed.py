import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from mtcnn import MTCNN
from sklearn.metrics import accuracy_score
from joblib import dump
from multiprocessing import Pool  # Import Pool for multiprocessing

# Process a single image file:
def process_file(item):
    # Each process creates its own detector instance.
    detector = MTCNN()
    
    # Read the image from disk
    img = cv2.imread(item)
    if img is None:
        return None  # Skip if image loading fails

    try:
        # Convert image from BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detect faces; use the first detected face
        out = detector.detect_faces(rgb_img)[0]
        x, y, w, h = out["box"]
        face = img[y:y+h, x:x+w]
    except Exception as e:
        return None  # Skip if detection fails

    try:
        # Resize the face to 32x32 pixels
        face = cv2.resize(face, (32, 32))
    except Exception as e:
        return None

    # Flatten and normalize the image
    face = face.flatten() / 255.0

    # Extract label from file path (assuming structure .../Gender/<label>/<image>)
    label = item.split("\\")[-2]
    return (face, label)

if __name__ == '__main__':
    # Get list of all image file paths
    file_list = glob.glob(r"Week5\reference\Datasets\Gender\*\*")

    data = []
    labels = []

    # Use multiprocessing Pool to process images in parallel
    with Pool() as pool:
        # Map process_file function to all image paths in parallel
        results = pool.map(process_file, file_list)

    # Iterate through results and add valid data to lists
    for i, result in enumerate(results):
        if result is None:
            continue
        face, label = result
        data.append(face)
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO]: {i}/{len(file_list)} processed")

    # Convert data to a numpy array
    data = np.array(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create and train the SGDClassifier
    clf = SGDClassifier()
    clf.fit(x_train, y_train)

    # Evaluate the classifier
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"accuracy: {acc}")

    # Save the trained model
    dump(clf, "genderDetection_Model.z")
