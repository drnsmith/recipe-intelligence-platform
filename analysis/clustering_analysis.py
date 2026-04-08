from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

def cluster_data(embeddings, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def evaluate_clustering(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    return score

if __name__ == "__main__":
    reduced_embeddings_path = 'reduced_embeddings.npy'
    embeddings = np.load(reduced_embeddings_path)
    cluster_labels = cluster_data(embeddings)
    silhouette_avg = evaluate_clustering(embeddings, cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg}")
    np.save('cluster_labels.npy', cluster_labels)
    print("Cluster labels saved.")

