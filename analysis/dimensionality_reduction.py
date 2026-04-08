from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# Load embeddings
def load_embeddings(file_path):
    df = pd.read_pickle(file_path)
    return np.array(df['bert_embeddings'].tolist())

# Apply PCA
def apply_pca(embeddings, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

# Apply t-SNE
def apply_tsne(embeddings, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)
    return tsne_embeddings

if __name__ == "__main__":
    file_path = 'bert_embeddings.pkl'
    embeddings = load_embeddings(file_path)
    reduced_embeddings = apply_pca(embeddings)
    tsne_embeddings = apply_tsne(reduced_embeddings)
    np.save('reduced_embeddings.npy', reduced_embeddings)
    np.save('tsne_embeddings.npy', tsne_embeddings)
    print("Reduced embeddings and t-SNE embeddings saved.")
