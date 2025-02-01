import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, 
                       cluster_std=0.6, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

print(y_kmeans)


