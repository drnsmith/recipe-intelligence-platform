"""
Semantic recipe search with MMR diversity reranking.
Loads FAISS index and recipe dataframe, returns ranked results.
"""

import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RecipeSearchEngine:
    """
    Semantic recipe search engine backed by FAISS.

    Supports:
    - Text query search
    - Optional difficulty filter (Easy / Medium / Hard / Very Hard)
    - MMR diversity reranking to avoid near-duplicate results
    """

    def __init__(self, index_path: str, recipes_path: str, embeddings_path: str):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(index_path)
        self.recipes = pd.read_csv(recipes_path)
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        logger.info(f"Search engine ready — {self.index.ntotal:,} recipes indexed")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query text to normalised embedding vector."""
        vec = self.model.encode([query], normalize_embeddings=True)
        return vec.astype(np.float32)

    def _mmr_rerank(
        self,
        query_vec: np.ndarray,
        candidate_indices: list,
        k: int,
        lambda_param: float = 0.6
    ) -> list:
        """
        Maximal Marginal Relevance reranking.
        Balances relevance to query against diversity among results.

        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
        """
        candidate_vecs = self.embeddings[candidate_indices]
        selected = []
        remaining = list(range(len(candidate_indices)))

        while len(selected) < k and remaining:
            if not selected:
                # First pick: highest relevance to query
                scores = candidate_vecs[remaining] @ query_vec.T
                best = remaining[int(np.argmax(scores))]
            else:
                selected_vecs = candidate_vecs[selected]
                relevance = (candidate_vecs[remaining] @ query_vec.T).flatten()
                # Similarity to already-selected items
                sim_to_selected = (candidate_vecs[remaining] @ selected_vecs.T).max(axis=1)
                mmr_scores = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
                best = remaining[int(np.argmax(mmr_scores))]

            selected.append(best)
            remaining.remove(best)

        return [candidate_indices[i] for i in selected]

    def search(
        self,
        query: str,
        k: int = 5,
        difficulty_filter: Optional[str] = None,
        fetch_k: int = 50,
        use_mmr: bool = True
    ) -> list[dict]:
        """
        Search for recipes semantically similar to the query.

        Args:
            query: Natural language search string
            k: Number of results to return
            difficulty_filter: Optional — 'Easy', 'Medium', 'Hard', 'Very Hard'
            fetch_k: Candidates to retrieve before reranking (must be >= k)
            use_mmr: Whether to apply MMR diversity reranking

        Returns:
            List of recipe dicts with title, ingredients, directions, difficulty, score
        """
        query_vec = self._encode_query(query)
        _, indices = self.index.search(query_vec, fetch_k)
        candidate_indices = indices[0].tolist()

        # Remove invalid indices (-1 can appear with IVF if nprobe is low)
        candidate_indices = [i for i in candidate_indices if i >= 0]

        # Difficulty filter
        if difficulty_filter and "difficulty" in self.recipes.columns:
            candidate_indices = [
                i for i in candidate_indices
                if self.recipes.iloc[i].get("difficulty", "") == difficulty_filter
            ]

        if not candidate_indices:
            return []

        if use_mmr:
            top_indices = self._mmr_rerank(query_vec, candidate_indices, k)
        else:
            top_indices = candidate_indices[:k]

        results = []
        for idx in top_indices:
            row = self.recipes.iloc[idx]
            score = float(self.embeddings[idx] @ query_vec.T)
            results.append({
                "id": int(idx),
                "title": row.get("title", ""),
                "ingredients": row.get("ingredients", ""),
                "directions": row.get("directions", ""),
                "difficulty": row.get("difficulty", "Unknown"),
                "similarity_score": round(score, 4)
            })

        return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Search recipes")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--index", default="data/recipes.index")
    parser.add_argument("--recipes", default="data/sample_recipes.csv")
    parser.add_argument("--embeddings", default="data/embeddings.npy")
    args = parser.parse_args()

    engine = RecipeSearchEngine(args.index, args.recipes, args.embeddings)
    results = engine.search(args.query, k=args.k, difficulty_filter=args.difficulty)
    print(json.dumps(results, indent=2))
