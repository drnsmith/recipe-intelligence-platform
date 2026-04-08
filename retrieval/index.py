"""
FAISS index builder for recipe embeddings.
Uses IVF index for large corpora (>100k recipes), flat index for samples.
"""

import numpy as np
import faiss
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
IVF_THRESHOLD = 100_000  # use IVF above this size
N_CELLS = 1024  # IVF cells — sqrt(n) rule of thumb for 1M vectors


def build_index(embeddings_path: str, index_path: str) -> faiss.Index:
    """
    Build and save a FAISS index from precomputed embeddings.

    Uses FlatIP (exact inner product) for small corpora,
    IVF with flat quantiser for large corpora.
    Embeddings must be L2-normalised (inner product == cosine similarity).

    Args:
        embeddings_path: Path to .npy embeddings file
        index_path: Path to save .index file

    Returns:
        Built FAISS index
    """
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path).astype(np.float32)
    n, dim = embeddings.shape
    logger.info(f"Embeddings shape: {n:,} x {dim}")

    if n > IVF_THRESHOLD:
        logger.info(f"Building IVF index with {N_CELLS} cells...")
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, N_CELLS, faiss.METRIC_INNER_PRODUCT)
        logger.info("Training IVF index...")
        index.train(embeddings)
    else:
        logger.info("Building flat index (exact search)...")
        index = faiss.IndexFlatIP(dim)

    logger.info("Adding vectors to index...")
    index.add(embeddings)
    logger.info(f"Index contains {index.ntotal:,} vectors")

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved index to {index_path}")

    return index


def load_index(index_path: str) -> faiss.Index:
    """Load a saved FAISS index."""
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    logger.info(f"Index loaded — {index.ntotal:,} vectors")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--embeddings", default="data/embeddings.npy")
    parser.add_argument("--index", default="data/recipes.index")
    args = parser.parse_args()

    build_index(args.embeddings, args.index)
