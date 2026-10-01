"""
scripts/generate_plots.py
==========================
Generates publication-quality visualizations for the research report and portfolio.

Generated Plots:
  1. results/plots/apfd_distribution.png       — Boxplot / violin of APFD per strategy
  2. results/plots/ttff_comparison.png          — Time-to-first-failure (TTFF) speedup
  3. results/plots/feature_importance.png       — RF & XGBoost feature importance ranking
  4. results/plots/apfd_over_commits.png        — Cumulative APFD trajectory across test commits
  5. results/plots/flaky_detection_summary.png  — Flaky score distribution and quarantine effect

Usage:
  python scripts/generate_plots.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Styling
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

PALETTE = {
    "random": "#95a5a6",
    "recently_failed_first": "#3498db",
    "coverage_based": "#9b59b6",
    "random_forest": "#2ecc71",
    "xgboost": "#e67e22",
}


def plot_apfd_distribution(per_commit_df: pd.DataFrame, out_path: Path) -> None:
    """APFD distribution comparison across all strategies."""
    plt.figure(figsize=(9, 5.5))
    df_clean = per_commit_df.dropna(subset=["apfd"])

    order = ["random", "recently_failed_first", "coverage_based", "random_forest", "xgboost"]
    order = [s for s in order if s in df_clean["strategy"].unique()]

    sns.boxplot(
        data=df_clean,
        x="strategy",
        y="apfd",
        order=order,
        palette=PALETTE,
        boxprops=dict(alpha=0.75),
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "red",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )

    # Reference baseline line at 0.50
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Random Expectation (0.50)")

    plt.title("Average Percentage of Faults Detected (APFD) Distribution by Strategy")
    plt.xlabel("Prioritization Strategy")
    plt.ylabel("APFD Score (Higher is Better)")
    plt.ylim(0.0, 1.05)
    plt.xticks(
        range(len(order)),
        [s.replace("_", " ").title() for s in order],
        rotation=15,
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log.info("Saved: %s", out_path)


def plot_ttff_comparison(full_results_df: pd.DataFrame, out_path: Path) -> None:
    """Mean Time to First Failure (TTFF) comparison."""
    plt.figure(figsize=(8, 5))
    df_clean = full_results_df.copy()

    order = ["random", "recently_failed_first", "coverage_based", "random_forest", "xgboost"]
    order = [s for s in order if s in df_clean["strategy"].unique()]
    df_clean = df_clean.set_index("strategy").reindex(order).reset_index()

    colors = [PALETTE.get(s, "#34495e") for s in df_clean["strategy"]]

    bars = plt.bar(
        [s.replace("_", " ").title() for s in df_clean["strategy"]],
        df_clean["ttff_s_mean"],
        yerr=df_clean["ttff_s_std"].fillna(0),
        capsize=5,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1,
    )

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title("Time to First Failure (TTFF) — CI Feedback Latency")
    plt.xlabel("Prioritization Strategy")
    plt.ylabel("Mean Cumulative Test Execution Time (s)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log.info("Saved: %s", out_path)


def plot_feature_importance(imp_json_path: Path, out_path: Path) -> None:
    """Plot feature importance ranking from JSON."""
    if not imp_json_path.exists():
        log.warning("Feature importance JSON not found at %s", imp_json_path)
        return

    with open(imp_json_path) as f:
        data = json.load(f)

    rf_imp = pd.Series(data.get("random_forest", {})).sort_values(ascending=True)
    xgb_imp = pd.Series(data.get("xgboost", {})).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    # Random Forest
    axes[0].barh(
        rf_imp.index, rf_imp.values, color=PALETTE["random_forest"], alpha=0.85, edgecolor="black"
    )
    axes[0].set_title("Random Forest Gini Importance")
    axes[0].set_xlabel("Relative Importance")

    # XGBoost
    axes[1].barh(
        xgb_imp.index, xgb_imp.values, color=PALETTE["xgboost"], alpha=0.85, edgecolor="black"
    )
    axes[1].set_title("XGBoost Gain Importance")
    axes[1].set_xlabel("Relative Importance")

    fig.suptitle(
        "Feature Importance in ML Test Prioritization Models", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log.info("Saved: %s", out_path)


def plot_apfd_trajectory(per_commit_df: pd.DataFrame, out_path: Path) -> None:
    """Plot cumulative APFD across test commit sequence."""
    plt.figure(figsize=(10, 5.5))
    df = per_commit_df.dropna(subset=["apfd"]).copy()

    for strategy, grp in df.groupby("strategy"):
        sorted_grp = grp.sort_values("commit_ts").reset_index(drop=True)
        rolling_apfd = sorted_grp["apfd"].expanding().mean()
        plt.plot(
            range(1, len(rolling_apfd) + 1),
            rolling_apfd,
            label=strategy.replace("_", " ").title(),
            color=PALETTE.get(strategy, "#333333"),
            linewidth=2.2,
            marker="o",
            markersize=3,
            alpha=0.9,
        )

    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Random (0.50)")
    plt.title("Cumulative Mean APFD Across Test Commits")
    plt.xlabel("Commit Index in Evaluation Set")
    plt.ylabel("Cumulative Mean APFD")
    plt.ylim(0.3, 0.9)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log.info("Saved: %s", out_path)


def main() -> None:
    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    tables_dir = results_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)

    full_res_path = tables_dir / "full_results.csv"
    per_commit_path = tables_dir / "per_commit_full.csv"
    imp_path = Path("experiments/models/feature_importances.json")

    if not full_res_path.exists() or not per_commit_path.exists():
        log.error("Results tables missing. Run scripts/run_experiment.py first.")
        return

    full_res = pd.read_csv(full_res_path)
    per_commit = pd.read_csv(per_commit_path)

    plot_apfd_distribution(per_commit, plots_dir / "apfd_distribution.png")
    plot_ttff_comparison(full_res, plots_dir / "ttff_comparison.png")
    plot_feature_importance(imp_path, plots_dir / "feature_importance.png")
    plot_apfd_trajectory(per_commit, plots_dir / "apfd_over_commits.png")

    log.info("All plots generated successfully in %s", plots_dir)


if __name__ == "__main__":
    main()
