"""
Difficulty labelling using Gaussian Mixture Model clustering.

Rather than forcing equal-sized bins, GMM finds natural groupings
in the feature space and assigns difficulty based on cluster centroids.
This produces an honest, data-driven difficulty distribution.

Usage:
    python data/difficulty_labeller.py --input data/processed_sample.csv --output data/labelled_recipes.csv
    python data/difficulty_labeller.py --input data/processed_recipes.csv --output data/labelled_recipes.csv
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# Features used for clustering
FEATURE_COLS = [
    "ingredient_count",
    "direction_count",
    "avg_step_length",
    "technique_count",
    "unique_ner_count",
    "has_advanced_technique",
]

DIFFICULTY_LABELS = ["Easy", "Medium", "Hard", "Very Hard"]
N_COMPONENTS = 4


def fit_gmm(features: np.ndarray, n_components: int = N_COMPONENTS) -> GaussianMixture:
    """
    Fit a Gaussian Mixture Model on the standardised feature matrix.
    Tries multiple random seeds and returns the best fit by BIC.
    """
    logger.info(f"Fitting GMM with {n_components} components...")
    best_gmm = None
    best_bic = np.inf

    for seed in range(5):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=3,
            random_state=seed,
            max_iter=200
        )
        gmm.fit(features)
        bic = gmm.bic(features)
        logger.info(f"  seed={seed}  BIC={bic:.1f}  converged={gmm.converged_}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    logger.info(f"Best BIC: {best_bic:.1f}")
    return best_gmm


def order_clusters_by_difficulty(
    gmm: GaussianMixture,
    scaler: StandardScaler
) -> dict:
    """
    Order clusters from easiest to hardest by ranking cluster centroids
    on a weighted complexity score in the original feature space.

    Returns mapping: {gmm_cluster_id -> difficulty_label}
    """
    # Inverse transform centroids back to original feature space
    centroids = scaler.inverse_transform(gmm.means_)
    centroid_df = pd.DataFrame(centroids, columns=FEATURE_COLS)

    logger.info("Cluster centroids (original feature space):")
    for i, row in centroid_df.iterrows():
        logger.info(
            f"  Cluster {i}: "
            f"ingredients={row['ingredient_count']:.1f}  "
            f"steps={row['direction_count']:.1f}  "
            f"step_len={row['avg_step_length']:.1f}  "
            f"techniques={row['technique_count']:.1f}  "
            f"advanced={row['has_advanced_technique']:.2f}"
        )

    # Weighted score for ordering — same weights as preprocessing pipeline
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    score = (
        0.30 * norm(centroid_df["technique_count"]) +
        0.25 * norm(centroid_df["avg_step_length"]) +
        0.20 * norm(centroid_df["ingredient_count"]) +
        0.15 * norm(centroid_df["direction_count"]) +
        0.10 * centroid_df["has_advanced_technique"]
    )

    # Sort clusters by score ascending — lowest = Easy, highest = Very Hard
    order = score.argsort().values
    mapping = {int(cluster_id): DIFFICULTY_LABELS[rank] for rank, cluster_id in enumerate(order)}

    logger.info("Cluster → difficulty mapping:")
    for cluster_id, label in sorted(mapping.items()):
        logger.info(f"  Cluster {cluster_id} → {label}  (score={score[cluster_id]:.3f})")

    return mapping


def plot_clusters(
    features_2d: np.ndarray,
    labels: pd.Series,
    output_dir: Path
) -> None:
    """Save a 2D PCA visualisation of the clusters."""
    colours = {
        "Easy": "#2ecc71",
        "Medium": "#f39c12",
        "Hard": "#e74c3c",
        "Very Hard": "#8e44ad"
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for label in DIFFICULTY_LABELS:
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=colours[label],
            label=label,
            alpha=0.4,
            s=8,
            edgecolors='none'
        )
    ax.set_title("Recipe Difficulty Clusters (PCA projection)", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Difficulty", markerscale=3)
    plt.tight_layout()
    out = output_dir / "difficulty_clusters.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved cluster plot to {out}")


def bic_curve(features: np.ndarray, output_dir: Path) -> None:
    """
    Plot BIC scores for n_components 2-8 to validate choice of 4 clusters.
    Saves to output_dir/bic_curve.png
    """
    logger.info("Computing BIC curve to validate n_components=4...")
    bics = []
    ks = range(2, 9)
    for k in ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            random_state=42
        )
        gmm.fit(features)
        bics.append(gmm.bic(features))
        logger.info(f"  k={k}  BIC={bics[-1]:.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(ks), bics, marker='o', linewidth=2, color="#2c3e50")
    ax.axvline(x=4, color="#e74c3c", linestyle="--", label="n=4 (chosen)")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("BIC")
    ax.set_title("GMM BIC scores by number of components")
    ax.legend()
    plt.tight_layout()
    out = output_dir / "bic_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved BIC curve to {out}")


def run_labeller(
    input_path: str,
    output_path: str,
    plot: bool = True,
    validate_k: bool = True
) -> pd.DataFrame:

    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} recipes")

    # ── Prepare feature matrix ────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. "
            f"Run preprocess.py first."
        )

    X = df[FEATURE_COLS].fillna(0).values.astype(np.float64)
    logger.info(f"Feature matrix: {X.shape}")

    # ── Standardise ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Optional: BIC curve to validate k=4 ──────────────────────────────────
    if validate_k:
        bic_curve(X_scaled, plots_dir)

    # ── Fit GMM ───────────────────────────────────────────────────────────────
    gmm = fit_gmm(X_scaled)
    cluster_ids = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    # ── Order clusters by difficulty ──────────────────────────────────────────
    cluster_to_label = order_clusters_by_difficulty(gmm, scaler)
    df["difficulty"]         = pd.Categorical(
        [cluster_to_label[c] for c in cluster_ids],
        categories=DIFFICULTY_LABELS,
        ordered=True
    )
    df["difficulty_cluster"] = cluster_ids
    df["difficulty_confidence"] = probabilities.max(axis=1).round(4)

    # ── Log distribution ──────────────────────────────────────────────────────
    dist = df["difficulty"].value_counts().reindex(DIFFICULTY_LABELS)
    logger.info("Difficulty distribution:")
    for label, count in dist.items():
        logger.info(f"  {label:10s}: {count:>8,}  ({100*count/len(df):.1f}%)")

    # ── Plot clusters ─────────────────────────────────────────────────────────
    if plot:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        logger.info(
            f"PCA explained variance: "
            f"{pca.explained_variance_ratio_[0]:.1%} + "
            f"{pca.explained_variance_ratio_[1]:.1%}"
        )
        plot_clusters(X_2d, df["difficulty"], plots_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} labelled recipes to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMM-based difficulty labelling")
    parser.add_argument("--input",       default="data/processed_sample.csv")
    parser.add_argument("--output",      default="data/labelled_recipes.csv")
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip BIC curve (faster)")
    args = parser.parse_args()

    run_labeller(
        input_path=args.input,
        output_path=args.output,
        plot=not args.no_plot,
        validate_k=not args.no_validate
    )
