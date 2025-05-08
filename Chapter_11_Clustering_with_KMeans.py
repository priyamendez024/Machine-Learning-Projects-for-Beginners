
from sklearn.cluster import KMeans

# K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)
kmeans_labels = kmeans.predict(X_test)

# Visualize or evaluate clusters
print(f"Cluster Centers: {kmeans.cluster_centers_}")
