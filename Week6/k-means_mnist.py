import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"Week5\HW\reference\Dataset\Q2\mnist_train.csv", header=None)
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

print(x.shape)

kmeans = KMeans(n_clusters=10, random_state=123)
clusters = kmeans.fit_predict(x)
print(clusters.shape)
centers = kmeans.cluster_centers_
print(centers.shape)

for i, item in enumerate(centers):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(item, (28, 28)))

plt.show()



