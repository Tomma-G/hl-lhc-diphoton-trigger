#!/usr/bin/env python3
"""
Combined ROC plotting script for the main classifier outputs.

The script:
- loads ROC files for the baseline, isolation cut, neural network,
  histogram-based gradient boosting, and XGBoost classifiers
- handles either fake_rate/acceptance or fpr/tpr column naming
- plots all curves on a single figure
- saves the combined ROC comparison plot

Designed to match the submission-ready classifier outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = PROJECT_ROOT / "final_outputs" / "combined_roc_test.png"

# increase font sizes for better readability in reports
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})


def start_plot(figsize=(7, 5)) -> None:
    """Create a matplotlib figure with consistent dimensions."""
    plt.figure(figsize=figsize)


def finish_plot(path: Path) -> None:
    """Apply final layout settings, save the figure, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_roc(path: Path, label: str):
    """Load one ROC csv and add it to the current plot."""
    df = pd.read_csv(path)

    xcol = "fake_rate" if "fake_rate" in df.columns else "fpr"
    ycol = "acceptance" if "acceptance" in df.columns else "tpr"

    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    plt.plot(x, y, linewidth=2.0, label=label)


def main():
    """Plot the combined ROC comparison for all main classifiers."""
    roc_files = {
        "Baseline": PROJECT_ROOT / "final_outputs" / "baseline_classifier" / "baseline_classifier_roc.csv",
        "Isolation cut": PROJECT_ROOT / "final_outputs" / "isolation_cut_classifier" / "roc.csv",
        "Neural network": PROJECT_ROOT / "final_outputs" / "nn_classifier" / "single_run" / "roc_test.csv",
        "HistGradientBoosting": PROJECT_ROOT / "final_outputs" / "treehgb_classifier" / "single_run" / "roc_test.csv",
        "XGBoost": PROJECT_ROOT / "final_outputs" / "xgboost_classifier" / "single_run" / "roc_test.csv",
    }

    out_path = DEFAULT_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_plot()

    for label, path in roc_files.items():
        if not path.exists():
            print(f"missing file: {path}")
            continue
        load_roc(path, label)

    plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Random classifier")
    plt.xlabel("Jet fake rate")
    plt.ylabel("Photon acceptance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)

    finish_plot(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()