import json
import os
from pathlib import Path

import pandas as pd

from src.utils.config import METRICS_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_directories


FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_model_comparison() -> Path:
    comparison_path = METRICS_DIR / "model_comparison.json"
    rows = _load_metrics(comparison_path)
    df = pd.DataFrame(rows)

    metrics = ["roc_auc", "f1", "accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(df["model"], df[metric], color=["#e10600", "#1f77b4", "#2ca02c"][: len(df)])
        ax.set_title(metric.upper())
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=20)
        for x, val in enumerate(df[metric]):
            ax.text(x, float(val) + 0.015, f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle("Model Comparison on Test Set")
    fig.tight_layout()
    output_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_target_distribution() -> Path:
    features_path = PROCESSED_DIR / "f1_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature table: {features_path}")
    df = pd.read_csv(features_path)

    counts = df["target_points"].value_counts().sort_index()
    labels = ["No Points (0)", "Points (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#7f7f7f", "#e10600"])
    ax.set_title("Target Distribution: Points Finish")
    ax.set_ylabel("Number of Driver-Race Rows")
    for x, val in enumerate(values):
        ax.text(x, val + max(values) * 0.01, f"{val}", ha="center", fontsize=10)
    fig.tight_layout()

    output_path = FIGURES_DIR / "target_distribution.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_points_rate_by_season() -> Path:
    features_path = PROCESSED_DIR / "f1_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature table: {features_path}")
    df = pd.read_csv(features_path)

    season_rate = (
        df.groupby("season", as_index=False)["target_points"].mean().rename(
            columns={"target_points": "points_rate"}
        )
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(season_rate["season"], season_rate["points_rate"], color="#e10600", linewidth=2)
    ax.set_title("Points Finish Rate by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = FIGURES_DIR / "points_rate_by_season.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    ensure_directories()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_comp = plot_model_comparison()
    target_dist = plot_target_distribution()
    season_rate = plot_points_rate_by_season()

    print(f"Saved figure: {model_comp}")
    print(f"Saved figure: {target_dist}")
    print(f"Saved figure: {season_rate}")


if __name__ == "__main__":
    main()
