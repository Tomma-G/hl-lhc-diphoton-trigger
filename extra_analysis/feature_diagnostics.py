#!/usr/bin/env python3
"""
Diagnostics script for all trained models.

The script:
- loops over NN, HistGradientBoosting, and XGBoost result folders
- loads the engineered dataset for each model
- plots correlation matrices
- plots signal-versus-background feature distributions
- plots validation and test ROC curves
- plots classifier score distributions

Designed to match the submission-ready classifier outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# increase font sizes for better readability in reports
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_BASE_DIR = PROJECT_ROOT / "final_outputs"


# plotting utilities

def start_plot(figsize=(7, 5)) -> None:
    """Create a matplotlib figure with consistent dimensions."""
    plt.figure(figsize=figsize)


def finish_plot(path: Path) -> None:
    """Apply final layout settings, save the figure, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_correlation_matrix(df, feature_names, out_dir: Path, model_name):
    """Plot the full feature correlation matrix."""
    corr = df[feature_names].corr()

    start_plot(figsize=(14, 12))
    im = plt.imshow(corr, interpolation="nearest", aspect="auto", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel("Feature")
    plt.ylabel("Feature")
    finish_plot(out_dir / "feature_correlation_matrix.png")


def plot_signal_background_features(df, feature_names, out_dir: Path, model_name):
    """Plot signal and background feature distributions for all features."""
    signal = df[df["label"] == 1]
    background = df[df["label"] == 0]

    n_features = len(feature_names)
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        s = signal[feature].dropna().to_numpy(dtype=float)
        b = background[feature].dropna().to_numpy(dtype=float)

        all_vals = np.concatenate([s, b])
        all_vals = all_vals[np.isfinite(all_vals)]

        if all_vals.size == 0:
            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel("Density")
            continue

        vmin = np.min(all_vals)
        vmax = np.max(all_vals)

        if np.isclose(vmin, vmax):
            pad = 0.5 if np.isclose(vmin, 0.0) else 0.05 * abs(vmin)
            hist_range = (vmin - pad, vmax + pad)
        else:
            q_low, q_high = np.percentile(all_vals, [0.5, 99.5])

            if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
                hist_range = (vmin, vmax)
            else:
                hist_range = (q_low, q_high)

                if vmin >= 0 and hist_range[0] > 0:
                    hist_range = (0.0, hist_range[1])

                width = hist_range[1] - hist_range[0]
                pad = 0.03 * width
                hist_range = (hist_range[0] - pad, hist_range[1] + pad)

        ax.hist(
            b,
            bins=50,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Background",
        )
        ax.hist(
            s,
            bins=50,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Signal",
        )

        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(loc="best", frameon=True, framealpha=0.9)

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(out_dir / "signal_background_features.png", dpi=150)
    plt.close()


def plot_selected_feature_distributions(df, selected_features, out_dir: Path, model_name):
    """Plot standalone signal/background distributions for selected features."""
    signal = df[df["label"] == 1]
    background = df[df["label"] == 0]

    for feature in selected_features:
        if feature not in df.columns:
            print(f"skipping {feature} for {model_name} (feature not found)")
            continue

        s = signal[feature].dropna().to_numpy(dtype=float)
        b = background[feature].dropna().to_numpy(dtype=float)

        all_vals = np.concatenate([s, b])
        all_vals = all_vals[np.isfinite(all_vals)]

        if all_vals.size == 0:
            print(f"skipping {feature} for {model_name} (no finite values)")
            continue

        vmin = np.min(all_vals)
        vmax = np.max(all_vals)

        if np.isclose(vmin, vmax):
            pad = 0.5 if np.isclose(vmin, 0.0) else 0.05 * abs(vmin)
            hist_range = (vmin - pad, vmax + pad)
        else:
            q_low, q_high = np.percentile(all_vals, [0.5, 99.5])

            if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low >= q_high:
                hist_range = (vmin, vmax)
            else:
                hist_range = (q_low, q_high)

                if vmin >= 0 and hist_range[0] > 0:
                    hist_range = (0.0, hist_range[1])

                width = hist_range[1] - hist_range[0]
                pad = 0.03 * width
                hist_range = (hist_range[0] - pad, hist_range[1] + pad)

        start_plot(figsize=(7, 5))

        if feature == "iso_ratio":
            bins = 50
        elif feature == "mean_d0_sig":
            bins = 800
        elif feature == "sumpt_r0_0p05":
            bins = 100
        else:
            bins = 400

        plt.hist(
            b,
            bins=bins,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Background",
        )
        plt.hist(
            s,
            bins=bins,
            range=hist_range,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Signal",
        )

        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend(loc="best", frameon=True, framealpha=0.9)

        if feature == "iso_ratio":
            plt.xlim(0, 1.5)
        elif feature == "mean_d0_sig":
            plt.xlim(0, 10)
        elif feature == "sumpt_r0_0p05":
            plt.xlim(0, 20)

        finish_plot(out_dir / f"{feature}_signal_background.png")


def plot_roc_curves(results_dir: Path, out_dir: Path, model_name):
    """Plot validation and test ROC curves in fake-rate versus acceptance form."""
    start_plot()

    for split in ["val", "test"]:
        path = results_dir / f"roc_{split}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        xcol = "fake_rate" if "fake_rate" in df.columns else "fpr"
        ycol = "acceptance" if "acceptance" in df.columns else "tpr"

        auc = np.trapz(df[ycol].to_numpy(dtype=float), df[xcol].to_numpy(dtype=float))
        label = f"{split.capitalize()} (AUC = {auc:.3f})"
        plt.plot(df[xcol], df[ycol], linewidth=2.0, label=label)

    plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Random classifier")
    plt.xlabel("Jet fake rate")
    plt.ylabel("Photon acceptance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    finish_plot(out_dir / "roc_curve.png")


def plot_classifier_scores(results_dir: Path, out_dir: Path, model_name):
    """Plot classifier score distributions for validation and test samples."""
    val_path = results_dir / "scores_val.csv"
    test_path = results_dir / "scores_test.csv"

    if not (val_path.exists() and test_path.exists()):
        return

    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    start_plot(figsize=(8, 6))

    for df, split, linestyle in [
        (df_val, "Validation", "-"),
        (df_test, "Test", "--"),
    ]:
        sig = df[df["label"] == 1]["score"].to_numpy(dtype=float)
        bkg = df[df["label"] == 0]["score"].to_numpy(dtype=float)

        plt.hist(
            bkg,
            bins=50,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=1.8,
            linestyle=linestyle,
            label=f"{split} background",
        )
        plt.hist(
            sig,
            bins=50,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=1.8,
            linestyle=linestyle,
            label=f"{split} signal",
        )

    plt.xlabel("Classifier score")
    plt.ylabel("Density")
    plt.xlim(0.0, 1.0)
    plt.legend(loc="best", frameon=True, framealpha=0.9)
    finish_plot(out_dir / "score_distributions.png")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run diagnostic plots for all trained classifier outputs."
    )
    parser.add_argument(
        "--results-base-dir",
        type=Path,
        default=DEFAULT_RESULTS_BASE_DIR,
        help="base directory containing nn_classifier, treehgb_classifier, and xgboost_classifier folders",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="single_run",
        help="run subfolder name to inspect inside each model directory",
    )
    return parser.parse_args()


def main():
    """Run diagnostics for all classifier result folders."""
    args = parse_args()

    results_base_dir = args.results_base_dir
    run_tag = args.run_tag

    models = [
        "nn_classifier",
        "treehgb_classifier",
        "xgboost_classifier",
    ]

    for model in models:
        print(f"\nprocessing: {model}")

        results_dir = results_base_dir / model / run_tag
        dataset_path = results_dir / "engineered_dataset.csv"

        if not dataset_path.exists():
            print(f"skipping {model} (no engineered_dataset.csv found)")
            continue

        df = pd.read_csv(dataset_path)

        if "label" not in df.columns:
            print(f"skipping {model} (no label column)")
            continue

        feature_names = [c for c in df.columns if c != "label"]

        selected_features = [
            "iso_ratio",
            "sumpt_r0_0p05",
            "mean_d0_sig",
        ]

        out_dir = results_dir / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_correlation_matrix(df, feature_names, out_dir, model)
        plot_signal_background_features(df, feature_names, out_dir, model)
        plot_selected_feature_distributions(df, selected_features, out_dir, model)
        plot_roc_curves(results_dir, out_dir, model)
        plot_classifier_scores(results_dir, out_dir, model)

        print(f"saved diagnostics -> {out_dir}")


if __name__ == "__main__":
    main()