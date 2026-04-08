"""
Full preprocessing pipeline for RecipeNLG / full_dataset.csv

Steps:
    1. Load and clean raw data
    2. Parse JSON columns (ingredients, directions, NER)
    3. Engineer complexity features
    4. Assign difficulty labels via quantile binning
    5. Save processed dataset ready for embedding

Usage:
    python data/preprocess.py --input data/full_dataset.csv --output data/processed_recipes.csv
    python data/preprocess.py --input data/full_dataset.csv --output data/processed_recipes.csv --sample 10000
"""

import pandas as pd
import numpy as np
import ast
import re
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── Cooking technique vocabulary ─────────────────────────────────────────────
TECHNIQUES = {
    # Basic
    "boil", "simmer", "steam", "poach", "blanch",
    "fry", "saute", "stir-fry", "deep-fry", "pan-fry",
    "bake", "roast", "broil", "grill", "toast",
    "microwave", "slow-cook", "pressure-cook",
    # Intermediate
    "braise", "deglaze", "reduce", "caramelise", "caramelize",
    "marinate", "cure", "pickle", "ferment", "smoke",
    "fold", "whisk", "beat", "cream", "knead",
    "strain", "puree", "blend", "emulsify",
    # Advanced
    "flambe", "flambé", "temper", "clarify", "render",
    "julienne", "brunoise", "chiffonade", "debone",
    "truss", "baste", "glaze", "proof", "laminate",
    "confit", "sous-vide", "brine", "cure", "sear",
    "dredge", "bread", "stuff", "wrap", "roll",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_parse_list(val) -> list:
    """Parse a JSON/repr list string into a Python list. Returns [] on failure."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def clean_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove non-ASCII."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text


def count_techniques(directions: list) -> int:
    """Count distinct cooking techniques mentioned across all steps."""
    full_text = " ".join(clean_text(s) for s in directions)
    words = set(re.findall(r'\b\w+\b', full_text))
    return len(words & TECHNIQUES)


def avg_step_length(directions: list) -> float:
    """Average word count per direction step."""
    if not directions:
        return 0.0
    lengths = [len(str(s).split()) for s in directions]
    return float(np.mean(lengths))


def ingredient_count(ingredients: list) -> int:
    """Number of ingredients."""
    return len(ingredients)


def unique_ner_count(ner: list) -> int:
    """Number of unique named ingredient entities."""
    return len(set(clean_text(str(n)) for n in ner if n))


def direction_count(directions: list) -> int:
    """Number of preparation steps."""
    return len(directions)


def has_advanced_technique(directions: list) -> int:
    """Binary flag — 1 if any advanced technique is mentioned."""
    advanced = {
        "flambe", "flambé", "temper", "clarify", "confit",
        "sous-vide", "brine", "laminate", "julienne", "brunoise",
        "chiffonade", "debone", "truss", "emulsify", "render"
    }
    full_text = " ".join(clean_text(s) for s in directions)
    words = set(re.findall(r'\b\w+\b', full_text))
    return int(bool(words & advanced))


# ── Complexity scoring ────────────────────────────────────────────────────────

def compute_complexity_score(df: pd.DataFrame) -> pd.Series:
    """
    Weighted complexity score from normalised feature components.

    Weights (sum to 1.0):
        0.30  technique_count       — skill diversity
        0.25  avg_step_length       — instruction complexity
        0.20  ingredient_count      — prep complexity
        0.15  direction_count       — effort
        0.10  has_advanced_technique — expertise signal
    """
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    score = (
        0.30 * norm(df["technique_count"]) +
        0.25 * norm(df["avg_step_length"]) +
        0.20 * norm(df["ingredient_count"]) +
        0.15 * norm(df["direction_count"]) +
        0.10 * df["has_advanced_technique"].astype(float)
    )
    return score


DIFFICULTY_LABELS = ["Easy", "Medium", "Hard", "Very Hard"]

def assign_difficulty(scores: pd.Series) -> pd.Series:
    """Bin complexity scores into 4 difficulty levels using corpus quantiles."""
    bins = pd.qcut(scores, q=4, labels=DIFFICULTY_LABELS, duplicates="drop")
    return bins


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    sample_n: int = None,
    chunksize: int = 50_000
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {input_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    if sample_n:
        df = pd.read_csv(input_path, nrows=sample_n)
        logger.info(f"Loaded sample: {len(df):,} rows")
    else:
        # Read in chunks for memory efficiency on 2GB file
        chunks = []
        total = 0
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            chunks.append(chunk)
            total += len(chunk)
            logger.info(f"  loaded {total:,} rows...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded full dataset: {len(df):,} rows")

    # ── Clean ─────────────────────────────────────────────────────────────────
    logger.info("Cleaning...")

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Parse JSON list columns
    logger.info("Parsing JSON columns...")
    df["ingredients"] = df["ingredients"].apply(safe_parse_list)
    df["directions"]  = df["directions"].apply(safe_parse_list)
    df["NER"]         = df["NER"].apply(safe_parse_list)

    # Drop rows missing essential fields
    before = len(df)
    df = df[df["title"].notna() & df["title"].str.strip().ne("")]
    df = df[df["directions"].map(len) > 0]
    df = df[df["ingredients"].map(len) > 0]
    after = len(df)
    logger.info(f"Dropped {before - after:,} incomplete rows — {after:,} remaining")

    # Clean title
    df["title"] = df["title"].apply(clean_text).str.title()

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Engineering features...")

    df["ingredient_count"]       = df["ingredients"].apply(ingredient_count)
    df["direction_count"]        = df["directions"].apply(direction_count)
    df["avg_step_length"]        = df["directions"].apply(avg_step_length)
    df["technique_count"]        = df["directions"].apply(count_techniques)
    df["unique_ner_count"]       = df["NER"].apply(unique_ner_count)
    df["has_advanced_technique"] = df["directions"].apply(has_advanced_technique)

    # ── Complexity score and difficulty label ─────────────────────────────────
    logger.info("Computing complexity scores...")
    df["complexity_score"] = compute_complexity_score(df)

    logger.info("Assigning difficulty labels...")
    df["difficulty"] = assign_difficulty(df["complexity_score"])

    # Log label distribution
    dist = df["difficulty"].value_counts().sort_index()
    logger.info("Difficulty distribution:")
    for label, count in dist.items():
        logger.info(f"  {label:10s}: {count:>8,}  ({100*count/len(df):.1f}%)")

    # ── Build search text ─────────────────────────────────────────────────────
    logger.info("Building search text field...")
    df["search_text"] = (
        df["title"] + ". Ingredients: " +
        df["ingredients"].apply(lambda x: ", ".join(str(i) for i in x)) +
        ". Instructions: " +
        df["directions"].apply(lambda x: " ".join(str(s) for s in x))
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    # Keep only columns needed downstream
    keep_cols = [
        "title", "ingredients", "directions", "NER",
        "link", "source",
        "ingredient_count", "direction_count", "avg_step_length",
        "technique_count", "unique_ner_count", "has_advanced_technique",
        "complexity_score", "difficulty", "search_text"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} processed recipes to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecipeNLG preprocessing pipeline")
    parser.add_argument("--input",  default="data/full_dataset.csv",
                        help="Path to raw CSV")
    parser.add_argument("--output", default="data/processed_recipes.csv",
                        help="Path to save processed CSV")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N rows (dev mode)")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        sample_n=args.sample
    )
