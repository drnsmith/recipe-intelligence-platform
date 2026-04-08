import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(tsne_embeddings, cluster_labels):
    plt.figure(figsize=(10, 8))
    for label in set(cluster_labels):
        cluster_points = tsne_embeddings[cluster_labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
    plt.title("t-SNE Clustering of Recipes")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    tsne_path = 'tsne_embeddings.npy'
    clusters_path = 'cluster_labels.npy'
    tsne_embeddings = np.load(tsne_path)
    cluster_labels = np.load(clusters_path)
    plot_tsne(tsne_embeddings, cluster_labels)
