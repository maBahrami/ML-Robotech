import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, 
                       cluster_std=0.7, random_state=0)

#plt.scatter(X[:, 0], X[:, 1], s=50)
#plt.show()


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)
print(y_kmeans)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis")

centers = kmeans.cluster_centers_
print(centers)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.show()