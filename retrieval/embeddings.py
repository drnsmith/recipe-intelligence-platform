"""
Sentence-transformer encoding pipeline for RecipeNLG corpus.
Encodes recipe text (title + ingredients + instructions) into 384-dim vectors.
Model: sentence-transformers/all-MiniLM-L6-v2
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 256


def load_recipes(csv_path: str) -> pd.DataFrame:
    """Load and validate recipe CSV."""
    df = pd.read_csv(csv_path)
    required = {"title", "ingredients", "directions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logger.info(f"Loaded {len(df):,} recipes from {csv_path}")
    return df


def build_recipe_text(row: pd.Series) -> str:
    """Combine title, ingredients and directions into a single string for encoding."""
    title = str(row.get("title", ""))
    ingredients = str(row.get("ingredients", ""))
    directions = str(row.get("directions", ""))
    return f"{title}. Ingredients: {ingredients}. Instructions: {directions}"


def compute_embeddings(
    csv_path: str,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_n: int = None
) -> np.ndarray:
    """
    Compute sentence-transformer embeddings for all recipes.

    Args:
        csv_path: Path to RecipeNLG CSV file
        output_path: Path to save .npy embeddings file
        batch_size: Encoding batch size (reduce if OOM)
        sample_n: If set, only encode first N recipes (for development)

    Returns:
        numpy array of shape (n_recipes, 384)
    """
    df = load_recipes(csv_path)

    if sample_n:
        df = df.head(sample_n)
        logger.info(f"Using sample of {sample_n:,} recipes")

    logger.info("Building recipe text representations...")
    texts = df.apply(build_recipe_text, axis=1).tolist()

    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    logger.info(f"Encoding {len(texts):,} recipes (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine similarity via dot product
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info(f"Saved embeddings: {output_path} — shape {embeddings.shape}")

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute recipe embeddings")
    parser.add_argument("--data", default="data/sample_recipes.csv")
    parser.add_argument("--output", default="data/embeddings.npy")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sample", type=int, default=None,
                        help="Encode only first N recipes (dev mode)")
    args = parser.parse_args()

    compute_embeddings(
        csv_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        sample_n=args.sample
    )
