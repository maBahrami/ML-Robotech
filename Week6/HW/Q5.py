import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"Week6\HW\reference\q5-Mall_Customers.csv")
x = dataset.iloc[:, 3:]

kmeans = KMeans(n_clusters=5, random_state=123)
clusters = kmeans.fit_predict(x)
print(clusters.shape)
centers = kmeans.cluster_centers_
print(centers.shape)


plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=clusters, cmap="viridis")

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Cluster of Customers")
plt.show()
