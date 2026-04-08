from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def filter_recipes(df, ingredients, preferences):
    """
    Filter recipes based on ingredients and preferences.
    """
    filtered = df.copy()

    # Match ingredients
    if ingredients:
        filtered = filtered[filtered['preprocessed_ingredients'].str.contains(ingredients, case=False, na=False)]

    # Match preferences
    if preferences:
        for pref in preferences:
            filtered = filtered[filtered['preprocessed_ingredients'].str.contains(pref, case=False, na=False)]

    return filtered

def recommend_by_embedding(df, embeddings, query, top_n=3):
    """
    Recommend recipes using embeddings.
    """
    # Compute embedding for query
    query_vector = np.random.rand(embeddings.shape[1])  # Replace with real embedding model
    similarities = cosine_similarity([query_vector], embeddings)[0]

    # Get top N results
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return df.iloc[top_indices]
