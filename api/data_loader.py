import pandas as pd
import numpy as np

def load_data():
    # Load recipe dataset
    csv_path = "data/updated_recipes_with_generated_embeddings.csv"
    df = pd.read_csv(csv_path)

    # Load precomputed embeddings
    npy_path = "data/recipe_embeddings.npy"
    embeddings = np.load(npy_path)

    return df, embeddings
