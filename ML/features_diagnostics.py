#!/usr/bin/env python3
# imports
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_feature_names():
    return [
        "obj_pt",
        "obj_eta",
        "obj_phi",
        "obj_e",
        "sum_pt",
        "n_tracks",
        "pt1_over_ptobj",
        "pt2_over_ptobj",
        "dr1",
        "dr2",
        "ntrk_r0_0p05",
        "sumpt_r0_0p05",
        "ntrk_r0p05_0p10",
        "sumpt_r0p05_0p10",
        "ntrk_r0p10_0p20",
        "sumpt_r0p10_0p20",
        "ntrk_r0p20_iso",
        "sumpt_r0p20_iso",
        "max_pt",
        "mean_dr",
        "ptw_mean_dr",
        "top2_sumpt_frac",
        "n_tracks_core",
        "sum_pt_core",
    ]


def plot_correlation_matrix(df, feature_names, out_dir):
    corr = df[feature_names].corr()

    plt.figure(figsize=(14, 12))
    im = plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_correlation_matrix.png"), dpi=200)
    plt.close()


def plot_signal_background_features(df, feature_names, out_dir):
    signal = df[df["label"] == 1]
    background = df[df["label"] == 0]

    n_features = len(feature_names)
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        s = signal[feature].dropna().values
        b = background[feature].dropna().values

        ax.hist(
            b,
            bins=50,
            density=True,
            histtype="step",
            label="background",
        )
        ax.hist(
            s,
            bins=50,
            density=True,
            histtype="step",
            label="signal",
        )

        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.legend()

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Signal vs Background Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "signal_background_features.png"), dpi=200)
    plt.close()


def plot_roc_curves(results_dir, out_dir):
    roc_test_path = os.path.join(results_dir, "roc_test.csv")
    roc_val_path = os.path.join(results_dir, "roc_val.csv")

    plt.figure()

    if os.path.exists(roc_val_path):
        roc_val = pd.read_csv(roc_val_path)
        plt.plot(roc_val["fpr"], roc_val["tpr"], label="validation ROC")

    if os.path.exists(roc_test_path):
        roc_test = pd.read_csv(roc_test_path)
        plt.plot(roc_test["fpr"], roc_test["tpr"], label="test ROC")

    plt.xlabel("Fake rate")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Fake Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "efficiency_vs_fake_rate.png"), dpi=200)
    plt.close()

def plot_classifier_score_distributions(results_dir, out_dir):
    val_path = os.path.join(results_dir, "scores_val.csv")
    test_path = os.path.join(results_dir, "scores_test.csv")

    if not (os.path.exists(val_path) and os.path.exists(test_path)):
        raise RuntimeError("scores_val.csv and/or scores_test.csv are missing.")

    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    val_sig = df_val[df_val["label"] == 1]["score"].to_numpy()
    val_bkg = df_val[df_val["label"] == 0]["score"].to_numpy()
    test_sig = df_test[df_test["label"] == 1]["score"].to_numpy()
    test_bkg = df_test[df_test["label"] == 0]["score"].to_numpy()

    plt.figure(figsize=(8, 6))

    plt.hist(
        val_bkg,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="step",
        label="validation background",
    )
    plt.hist(
        val_sig,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="step",
        label="validation signal",
    )
    plt.hist(
        test_bkg,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="step",
        linestyle="--",
        label="test background",
    )
    plt.hist(
        test_sig,
        bins=50,
        range=(0, 1),
        density=True,
        histtype="step",
        linestyle="--",
        label="test signal",
    )

    plt.xlabel("NN photon score")
    plt.ylabel("Density")
    plt.title("Classifier Score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "classifier_score_distributions.png"), dpi=200)
    plt.close()

def main():
    results_dir = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\ML\results"
    out_dir = os.path.join(results_dir, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = os.path.join(results_dir, "engineered_dataset.csv")
    df = pd.read_csv(dataset_path)

    feature_names = get_feature_names()

    if "label" not in df.columns:
        raise RuntimeError("engineered_dataset.csv does not contain a 'label' column.")

    plot_correlation_matrix(df, feature_names, out_dir)
    plot_signal_background_features(df, feature_names, out_dir)
    plot_roc_curves(results_dir, out_dir)
    plot_classifier_score_distributions(results_dir, out_dir)

    print("saved diagnostic plots to:", out_dir)


if __name__ == "__main__":
    main()