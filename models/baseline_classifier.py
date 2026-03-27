#!/usr/bin/env python3
"""
Simple n-tracks baseline classifier for photon-versus-jet discrimination using
per-event csv files.

The script:
- reads headerless photon, jet, and track files
- excludes converted photons by default
- counts tracks within a fixed ΔR cone around photons and jets
- evaluates the baseline rule accept if n_tracks <= threshold
- reports performance at fixed photon-efficiency working points
- saves ROC curves, count distributions, summary tables, and timing outputs

The intended use is for the Senior Honours Project study of fast, track-based
photon identification for the HL-LHC trigger.
"""

import os
import argparse
from pathlib import Path

# force single-thread execution for fair timing comparison across models
# this avoids parallel CPU execution artificially reducing per-object runtime
# for the trigger-feasibility study

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "initial_data" / "10k_ev"
DEFAULT_OUT_DIR = PROJECT_ROOT / "final_outputs" / "baseline_classifier"

N_EVENTS = 10000
DR = 0.20
TRK_PT_MIN = 0.75
TARGET_TPRS = [0.80, 0.90, 0.95]

# utilities

PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the n-tracks baseline photon-versus-jet classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing photons_<id>.csv, jets_<id>.csv, and tracks_<id>.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory to save plots, tables, and timing outputs",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=N_EVENTS,
        help="maximum number of events to process",
    )
    return parser.parse_args()


def start_plot() -> None:
    """Create a report-sized matplotlib figure with consistent dimensions."""
    plt.figure(figsize=(7, 5))


def finish_plot(path: Path) -> None:
    """Apply final layout settings, save the figure, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def find_event_ids(data_dir: Path):
    """Return sorted event IDs discovered from photons_<id>.csv in data_dir."""
    ids = []
    for path in data_dir.iterdir():
        m = PHOTON_RE.match(path.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def count_tracks_for_objects(obj_eta, obj_phi, trk_eta, trk_phi, dr):
    """Count tracks within a cone for many objects in one event.

    Parameters
    ----------
    obj_eta, obj_phi : array-like
        Object coordinates with shape (n_obj,).
    trk_eta, trk_phi : array-like
        Track coordinates with shape (n_trk,).
    dr : float
        Cone radius.

    Returns
    -------
    np.ndarray
        Integer track counts for each object.
    """
    obj_eta = np.asarray(obj_eta, dtype=float)
    obj_phi = np.asarray(obj_phi, dtype=float)
    trk_eta = np.asarray(trk_eta, dtype=float)
    trk_phi = np.asarray(trk_phi, dtype=float)

    if obj_eta.size == 0:
        return np.zeros(0, dtype=int)

    if trk_eta.size == 0:
        return np.zeros(obj_eta.size, dtype=int)

    deta = trk_eta[None, :] - obj_eta[:, None]
    dphi = (trk_phi[None, :] - obj_phi[:, None] + np.pi) % (2 * np.pi) - np.pi
    dr2 = deta * deta + dphi * dphi

    return np.count_nonzero(dr2 < dr * dr, axis=1)


def roc_from_counts(ph, jt):
    """Build the ROC curve for the baseline cut n_tracks <= threshold."""
    ph = np.asarray(ph, dtype=int)
    jt = np.asarray(jt, dtype=int)

    max_c = int(max(ph.max(), jt.max()))
    thresholds = np.arange(-1, max_c + 1)

    tpr = np.array([np.mean(ph <= t) for t in thresholds], dtype=float)
    fpr = np.array([np.mean(jt <= t) for t in thresholds], dtype=float)

    return thresholds, fpr, tpr


def auc(fpr, tpr):
    """Compute the AUC from ROC points."""
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def best_threshold_balanced(ph, jt):
    """Choose the threshold maximising balanced accuracy.

    The acceptance rule is n_tracks <= threshold.

    Returns
    -------
    tuple
        (best_threshold, photon_efficiency, jet_fake_rate, balanced_accuracy)
    """
    ph = np.asarray(ph, dtype=int)
    jt = np.asarray(jt, dtype=int)

    max_c = int(max(ph.max(), jt.max()))

    best_score = -1.0
    best_t = 0

    for t in range(-1, max_c + 1):
        photon_eff = np.mean(ph <= t)
        jet_fake = np.mean(jt <= t)
        jet_rej = 1.0 - jet_fake
        bal_acc = 0.5 * (photon_eff + jet_rej)

        if bal_acc > best_score:
            best_score = bal_acc
            best_t = t

    photon_eff = np.mean(ph <= best_t)
    jet_fake = np.mean(jt <= best_t)

    return best_t, photon_eff, jet_fake, best_score


def fake_at_target_tpr(thresholds, fpr, tpr, target_tpr=0.95):
    """Return the best threshold and fake rate achieving at least target_tpr."""
    mask = tpr >= target_tpr

    if not np.any(mask):
        return np.nan, np.nan, np.nan

    valid_thresholds = thresholds[mask]
    valid_fpr = fpr[mask]
    valid_tpr = tpr[mask]

    best_idx = np.argmin(valid_fpr)

    return (
        float(valid_thresholds[best_idx]),
        float(valid_fpr[best_idx]),
        float(valid_tpr[best_idx]),
    )


def main():
    """Run the n-tracks baseline analysis."""
    args = parse_args()
    total_start = time.perf_counter()

    data_dir = args.data_dir
    out_dir = args.out_dir
    n_events = args.n_events

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = out_dir / "baseline_classifier_summary.csv"
    roc_csv_path = out_dir / "baseline_classifier_roc.csv"
    roc_png_path = out_dir / "baseline_classifier_roc.png"
    ntracks_png_path = out_dir / "baseline_classifier_ntracks.png"
    timing_csv_path = out_dir / "baseline_classifier_timing.csv"

    print("running n_tracks baseline...")
    print(f"data directory: {data_dir}")
    print(f"output directory: {out_dir}")

    event_ids = find_event_ids(data_dir)[:n_events]

    photon_counts = []
    jet_counts = []

    # timing accumulators
    io_time_total = 0.0
    counting_time_total = 0.0
    postproc_time_total = 0.0

    n_events_attempted = 0
    n_events_used = 0
    n_photons_scored = 0
    n_jets_scored = 0

    for ev in event_ids:
        n_events_attempted += 1

        # file loading and event setup
        t0_io = time.perf_counter()

        try:
            df_ph = pd.read_csv(
                data_dir / f"photons_{ev}.csv",
                header=None,
                names=["pT", "eta", "phi", "e", "conversionType"],
            )
        except Exception:
            io_time_total += time.perf_counter() - t0_io
            continue

        try:
            df_j = pd.read_csv(
                data_dir / f"jets_{ev}.csv",
                header=None,
                names=["pT", "eta", "phi", "e"],
            )
        except Exception:
            df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])

        try:
            df_trk = pd.read_csv(
                data_dir / f"tracks_{ev}.csv",
                header=None,
                names=["pT", "eta", "phi", "eTot", "z0", "d0"],
            )
        except Exception:
            io_time_total += time.perf_counter() - t0_io
            continue

        # exclude converted photons so the baseline is consistent with the other classifiers
        conv_int = pd.to_numeric(df_ph["conversionType"], errors="coerce").fillna(-999).astype(int)
        df_ph = df_ph[conv_int == 0].reset_index(drop=True)

        # apply the global track pT threshold before counting
        df_trk = df_trk[df_trk["pT"] > TRK_PT_MIN]

        trk_eta = df_trk["eta"].to_numpy(dtype=float)
        trk_phi = df_trk["phi"].to_numpy(dtype=float)

        ph_eta = df_ph["eta"].to_numpy(dtype=float) if len(df_ph) else np.zeros(0, dtype=float)
        ph_phi = df_ph["phi"].to_numpy(dtype=float) if len(df_ph) else np.zeros(0, dtype=float)

        j_eta = df_j["eta"].to_numpy(dtype=float) if len(df_j) else np.zeros(0, dtype=float)
        j_phi = df_j["phi"].to_numpy(dtype=float) if len(df_j) else np.zeros(0, dtype=float)

        io_time_total += time.perf_counter() - t0_io

        # actual per-object counting step
        t0_count = time.perf_counter()

        ph_counts_evt = count_tracks_for_objects(ph_eta, ph_phi, trk_eta, trk_phi, DR)
        j_counts_evt = count_tracks_for_objects(j_eta, j_phi, trk_eta, trk_phi, DR)

        if ph_counts_evt.size > 0:
            photon_counts.extend(ph_counts_evt.tolist())
        if j_counts_evt.size > 0:
            jet_counts.extend(j_counts_evt.tolist())

        counting_time_total += time.perf_counter() - t0_count

        n_photons_scored += int(ph_counts_evt.size)
        n_jets_scored += int(j_counts_evt.size)

        # count the event as used if at least one object was scored
        if ph_counts_evt.size > 0 or j_counts_evt.size > 0:
            n_events_used += 1

    photon_counts = np.asarray(photon_counts, dtype=int)
    jet_counts = np.asarray(jet_counts, dtype=int)

    if len(photon_counts) == 0 or len(jet_counts) == 0:
        print("no valid photon or jet counts found")
        return

    # ROC, summaries, and output writing
    t0_post = time.perf_counter()

    thresholds, fpr, tpr = roc_from_counts(photon_counts, jet_counts)
    auc_val = auc(fpr, tpr)

    best_t, eff, fake, bal_acc = best_threshold_balanced(photon_counts, jet_counts)

    fixed_wp_rows = []
    for target in TARGET_TPRS:
        t_cut, fake_rate, achieved_tpr = fake_at_target_tpr(
            thresholds, fpr, tpr, target_tpr=target
        )
        fixed_wp_rows.append(
            {
                "target_tpr": float(target),
                "cut": float(t_cut),
                "achieved_tpr": float(achieved_tpr),
                "fake_rate": float(fake_rate),
            }
        )

    print(f"AUC: {auc_val:.4f}")
    print(f"best cut: n_tracks <= {best_t}")
    print(f"photon efficiency: {eff:.3f}")
    print(f"jet fake rate: {fake:.3f}")
    print(f"balanced accuracy: {bal_acc:.3f}")

    for row in fixed_wp_rows:
        wp = int(round(100.0 * row["target_tpr"]))
        if np.isfinite(row["fake_rate"]):
            print(
                f"fake@{wp}: {row['fake_rate']:.6f}  "
                f"(cut: n_tracks <= {int(row['cut'])}, achieved_tpr: {row['achieved_tpr']:.6f})"
            )
        else:
            print(f"fake@{wp}: not reachable")

    summary_df = pd.DataFrame([{
        "auc": auc_val,
        "best_cut": best_t,
        "best_photon_eff": eff,
        "best_jet_fake": fake,
        "best_balanced_accuracy": bal_acc,
        "cut_at_80_tpr": fixed_wp_rows[0]["cut"],
        "fake_at_80_tpr": fixed_wp_rows[0]["fake_rate"],
        "achieved_tpr_at_80": fixed_wp_rows[0]["achieved_tpr"],
        "cut_at_90_tpr": fixed_wp_rows[1]["cut"],
        "fake_at_90_tpr": fixed_wp_rows[1]["fake_rate"],
        "achieved_tpr_at_90": fixed_wp_rows[1]["achieved_tpr"],
        "cut_at_95_tpr": fixed_wp_rows[2]["cut"],
        "fake_at_95_tpr": fixed_wp_rows[2]["fake_rate"],
        "achieved_tpr_at_95": fixed_wp_rows[2]["achieved_tpr"],
        "dr": DR,
        "trk_pt_min": TRK_PT_MIN,
        "converted_photons": "excluded",
    }])
    summary_df.to_csv(summary_csv_path, index=False)

    roc_df = pd.DataFrame({
        "threshold": thresholds,
        "fake_rate": fpr,
        "acceptance": tpr,
    })
    roc_df.to_csv(roc_csv_path, index=False)

    start_plot()
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=rf"Baseline classifier (AUC = {auc_val:.3f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        linewidth=1.5,
        label="Random classifier",
    )
    plt.xlabel("Jet fake rate")
    plt.ylabel("Photon acceptance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    finish_plot(roc_png_path)

    start_plot()
    bins = np.arange(0, max(photon_counts.max(), jet_counts.max()) + 2) - 0.5

    plt.hist(photon_counts, bins=bins, alpha=0.6, density=True, label="Photons")
    plt.hist(jet_counts, bins=bins, alpha=0.6, density=True, label="Jets")
    plt.axvline(best_t + 0.5, linestyle="--", linewidth=1.5, label=f"Best cut = {best_t}")

    for row, linestyle in zip(fixed_wp_rows, [":", "-.", (0, (3, 1, 1, 1))]):
        if np.isfinite(row["cut"]):
            wp = int(round(100.0 * row["target_tpr"]))
            plt.axvline(
                row["cut"] + 0.5,
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{wp}% TPR cut = {int(row['cut'])}",
            )

    plt.xlabel(rf"Number of tracks within $\Delta R < {DR:.2f}$")
    plt.ylabel("Density")
    plt.legend(frameon=True, framealpha=0.9)
    finish_plot(ntracks_png_path)

    postproc_time_total += time.perf_counter() - t0_post
    total_wall_time = time.perf_counter() - total_start

    # derived timing quantities
    n_objects_scored = n_photons_scored + n_jets_scored

    io_time_per_event = io_time_total / n_events_used if n_events_used > 0 else np.nan
    counting_time_per_event = counting_time_total / n_events_used if n_events_used > 0 else np.nan
    counting_time_per_object = counting_time_total / n_objects_scored if n_objects_scored > 0 else np.nan
    total_wall_time_per_event = total_wall_time / n_events_used if n_events_used > 0 else np.nan

    print("\ntiming")
    print(f"events attempted: {n_events_attempted}")
    print(f"events used: {n_events_used}")
    print(f"photons scored: {n_photons_scored}")
    print(f"jets scored: {n_jets_scored}")
    print(f"total objects scored: {n_objects_scored}")
    print(f"io/loading time total: {io_time_total:.6f} s")
    print(f"track-counting time total: {counting_time_total:.6f} s")
    print(f"post-processing/output time total: {postproc_time_total:.6f} s")
    print(f"total wall time: {total_wall_time:.6f} s")
    print(f"io/loading time per used event: {io_time_per_event:.6e} s")
    print(f"track-counting time per used event: {counting_time_per_event:.6e} s")
    print(f"track-counting time per object: {counting_time_per_object:.6e} s")
    print(f"total wall time per used event: {total_wall_time_per_event:.6e} s")

    timing_df = pd.DataFrame([{
        "events_attempted": n_events_attempted,
        "events_used": n_events_used,
        "photons_scored": n_photons_scored,
        "jets_scored": n_jets_scored,
        "objects_scored": n_objects_scored,
        "io_time_total_s": io_time_total,
        "counting_time_total_s": counting_time_total,
        "postproc_time_total_s": postproc_time_total,
        "total_wall_time_s": total_wall_time,
        "io_time_per_used_event_s": io_time_per_event,
        "counting_time_per_used_event_s": counting_time_per_event,
        "counting_time_per_object_s": counting_time_per_object,
        "total_wall_time_per_used_event_s": total_wall_time_per_event,
        "dr": DR,
        "trk_pt_min": TRK_PT_MIN,
        "n_events_requested": n_events,
        "converted_photons": "excluded",
    }])
    timing_df.to_csv(timing_csv_path, index=False)


if __name__ == "__main__":
    main()