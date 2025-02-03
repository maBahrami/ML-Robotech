import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
#print(digits.data.shape)

#img = digits.data[0]
#img = np.reshape(img, (8, 8))
#img = cv2.resize(img, (64, 64))

#cv2.imshow("image", img)
#cv2.waitKey(0)


kmeans = KMeans(n_clusters=10, random_state=123)
clusters = kmeans.fit_predict(digits.data)
print(clusters.shape)
centers = kmeans.cluster_centers_
print(centers.shape)

for i, item in enumerate(centers):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(item, (8, 8)))

plt.show()

