#!/usr/bin/env python3
"""
XGBoost classifier for photon-versus-jet discrimination using track-based
features engineered from per-event csv files.

The script:
- reads headerless photon, jet, and track files
- builds a per-object feature matrix using nearby-track information
- performs group-aware train/validation/test splits by event
- trains an XGBClassifier
- evaluates the classifier at fixed photon-efficiency working points
- saves scores, ROC curves, feature-importance outputs, plots, and timing summaries

The intended use is for the Senior Honours Project study of fast, track-based
photon identification for the HL-LHC trigger.
"""

from __future__ import annotations

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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "initial_data" / "10k_ev"
DEFAULT_OUT_DIR = PROJECT_ROOT / "final_outputs" / "xgboost_classifier"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"

PHOTON_RE = re.compile(r"^photons_(\d+)\.(csv|csv\.gz)$")


# file reading utilities for headerless csv files

def _is_gzip_file(path: Path) -> bool:
    """Return True if the file begins with the gzip magic bytes."""
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _is_tar_file(path: Path) -> bool:
    """Return True if the file looks like a tar archive."""
    try:
        with open(path, "rb") as f:
            f.seek(257)
            return f.read(5) == b"ustar"
    except OSError:
        return False


def read_noheader(path: Path, kind: str) -> pd.DataFrame:
    """Read a headerless photon, jet, or track csv file."""
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()

    if _is_tar_file(path):
        return pd.DataFrame()

    compression = "gzip" if _is_gzip_file(path) else None

    try:
        df = pd.read_csv(
            path,
            header=None,
            compression=compression,
            engine="python",
            on_bad_lines="skip",
        )
    except (pd.errors.EmptyDataError, EOFError, OSError, ValueError):
        print(f"[warn] skipping unreadable file: {path}")
        return pd.DataFrame()
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                path,
                header=None,
                compression=compression,
                engine="python",
                on_bad_lines="skip",
                encoding="latin-1",
            )
        except (pd.errors.EmptyDataError, EOFError, OSError, ValueError):
            print(f"[warn] skipping unreadable file: {path}")
            return pd.DataFrame()

    if df.empty:
        return df

    if kind == "photons":
        cols = ["pT", "eta", "phi", "e", "conversionType"]
        df.columns = cols[: len(df.columns)]
        keep = ["pT", "eta", "phi", "e"][: len(df.columns)]

    elif kind == "jets":
        cols = ["pT", "eta", "phi", "e", "conversionType"]
        df.columns = cols[: len(df.columns)]
        keep = ["pT", "eta", "phi", "e"][: len(df.columns)]

    elif kind == "tracks":
        cols = ["pT", "eta", "phi", "eTot", "z0", "d0"]
        df.columns = cols[: len(df.columns)]
        keep = ["pT", "eta", "phi", "z0", "d0"][: len(df.columns)]

    else:
        raise ValueError(f"unknown kind: {kind}")

    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if kind == "photons" and "conversionType" in df.columns:
        df["conversionType"] = pd.to_numeric(df["conversionType"], errors="coerce")

    df = df.dropna(subset=keep).reset_index(drop=True)
    return df


# plotting utilities

def start_plot() -> None:
    """Create a report-sized matplotlib figure with consistent dimensions."""
    plt.figure(figsize=(7, 5))


def finish_plot(path: Path) -> None:
    """Apply final layout settings, save the figure, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# angular utilities

def delta_phi(phi1, phi2):
    """Return the wrapped azimuthal angle difference in the range [-pi, pi)."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


# expected track-parameter resolution as a function of |eta|

ABS_ETA_EDGES = np.arange(0.0, 4.0 + 0.08, 0.08, dtype=float)

D0_RES = np.array([
    28.5165, 28.8721, 29.5522, 29.6418, 30.6022, 31.5378, 31.2663, 31.9463,
    32.6932, 33.8432, 33.8930, 37.1508, 36.2868, 38.9589, 40.0304, 42.9669,
    44.0429, 47.8792, 49.6747, 53.6438, 55.2563, 60.7846, 59.1389, 62.0128,
    65.3026, 73.5658, 75.2517, 80.0978, 80.0009, 87.2754, 88.1689, 87.0272,
    88.7612, 89.3318, 92.3440, 91.3461, 94.7994, 93.5043, 94.9451, 103.2830,
    104.1450, 110.4640, 120.2100, 123.5740, 130.5890, 133.5190, 147.4030, 143.2300,
    157.1470, 170.2160
], dtype=float) / 1000.0

Z0_RES = np.array([
    46.3440, 43.8577, 41.9337, 40.2396, 39.6328, 39.0040, 40.8912, 43.6346,
    44.6413, 47.9738, 54.4781, 55.0214, 61.1268, 68.1030, 75.0719, 84.3303,
    93.7729, 103.7640, 112.7640, 128.0570, 144.4800, 164.5350, 178.4040, 203.1490,
    231.8490, 268.7760, 283.5240, 332.1920, 371.2580, 425.7180, 462.1130, 525.7340,
    575.9410, 591.8680, 633.5750, 669.6620, 728.8720, 817.2550, 894.1260, 1003.2800,
    1196.3600, 1360.9600, 1516.3500, 1653.7000, 1889.0800, 2072.2600, 2481.8700, 2510.0300,
    2983.7900, 3219.5700
], dtype=float) / 1000.0


def get_expected_resolution(eta):
    """Look up the expected d0 and z0 resolutions for a given eta value."""
    aeta = abs(float(eta))
    idx = np.searchsorted(ABS_ETA_EDGES, aeta, side="right") - 1
    idx = int(np.clip(idx, 0, len(D0_RES) - 1))
    return float(D0_RES[idx]), float(Z0_RES[idx])


# feature definitions

def get_feature_names():
    """Return the ordered list of engineered feature names."""
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
        "iso_ratio",
        "core_iso_ratio",
        "maxpt_over_objpt",
        "core_frac",
        "mean_abs_d0",
        "max_abs_d0",
        "ptw_mean_abs_d0",
        "mean_abs_z0",
        "max_abs_z0",
        "ptw_mean_abs_z0",
        "mean_d0_sig",
        "max_d0_sig",
        "mean_z0_sig",
        "max_z0_sig",
    ]


# feature engineering for a single photon or jet candidate

def engineer_features(obj, tracks, iso_dr=0.30, trk_pt_min=1.0):
    """Build the full feature vector for one photon or jet candidate."""
    try:
        pt_obj = float(obj["pT"])
        eta_obj = float(obj["eta"])
        phi_obj = float(obj["phi"])
        e_obj = float(obj["e"]) if "e" in obj else 0.0
    except Exception:
        return None

    if not np.isfinite(pt_obj) or pt_obj <= 0:
        return None

    ring_edges = [
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, iso_dr),
    ]

    def padded_zero_features():
        """Return a zero-padded feature vector for objects with no associated tracks."""
        return [
            pt_obj,
            eta_obj,
            phi_obj,
            e_obj,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]

    if tracks.empty:
        return padded_zero_features()

    trk_pt = tracks["pT"].to_numpy(dtype=float)
    trk_eta = tracks["eta"].to_numpy(dtype=float)
    trk_phi = tracks["phi"].to_numpy(dtype=float)
    trk_d0 = tracks["d0"].to_numpy(dtype=float)
    trk_z0 = tracks["z0"].to_numpy(dtype=float)

    pt_mask = trk_pt > trk_pt_min
    if not np.any(pt_mask):
        return padded_zero_features()

    trk_pt = trk_pt[pt_mask]
    trk_eta = trk_eta[pt_mask]
    trk_phi = trk_phi[pt_mask]
    trk_d0 = trk_d0[pt_mask]
    trk_z0 = trk_z0[pt_mask]

    deta = trk_eta - eta_obj
    dphi = (trk_phi - phi_obj + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt(deta * deta + dphi * dphi)

    assoc_mask = dr < iso_dr
    if not np.any(assoc_mask):
        return padded_zero_features()

    pt_arr = trk_pt[assoc_mask]
    eta_arr = trk_eta[assoc_mask]
    dr_arr = dr[assoc_mask]
    d0_arr = trk_d0[assoc_mask]
    z0_arr = trk_z0[assoc_mask]

    order = np.argsort(-pt_arr)
    pt_arr = pt_arr[order]
    eta_arr = eta_arr[order]
    dr_arr = dr_arr[order]
    d0_arr = d0_arr[order]
    z0_arr = z0_arr[order]

    sum_pt = float(np.sum(pt_arr))
    n_tracks = float(len(pt_arr))

    pt1 = float(pt_arr[0]) if len(pt_arr) >= 1 else 0.0
    pt2 = float(pt_arr[1]) if len(pt_arr) >= 2 else 0.0
    dr1 = float(dr_arr[0]) if len(pt_arr) >= 1 else 0.0
    dr2 = float(dr_arr[1]) if len(pt_arr) >= 2 else 0.0

    pt1_over_ptobj = pt1 / pt_obj
    pt2_over_ptobj = pt2 / pt_obj

    ring_features = []
    for lo, hi in ring_edges:
        m = (dr_arr >= lo) & (dr_arr < hi)
        ring_features.append(float(np.count_nonzero(m)))
        ring_features.append(float(np.sum(pt_arr[m])) if np.any(m) else 0.0)

    max_pt = pt1
    mean_dr = float(np.mean(dr_arr)) if len(dr_arr) else 0.0
    ptw_mean_dr = float(np.sum(pt_arr * dr_arr) / sum_pt) if sum_pt > 0 else 0.0
    top2_sumpt_frac = float((pt1 + pt2) / sum_pt) if sum_pt > 0 else 0.0

    core_mask = dr_arr < 0.05
    n_tracks_core = float(np.count_nonzero(core_mask))
    sum_pt_core = float(np.sum(pt_arr[core_mask])) if np.any(core_mask) else 0.0

    iso_ratio = float(sum_pt / pt_obj) if pt_obj > 0 else 0.0
    core_iso_ratio = float(sum_pt_core / pt_obj) if pt_obj > 0 else 0.0
    maxpt_over_objpt = float(max_pt / pt_obj) if pt_obj > 0 else 0.0
    core_frac = float(sum_pt_core / sum_pt) if sum_pt > 0 else 0.0

    abs_d0 = np.abs(d0_arr)
    abs_z0 = np.abs(z0_arr)

    d0_sig = np.zeros_like(abs_d0)
    z0_sig = np.zeros_like(abs_z0)

    for i in range(len(eta_arr)):
        d0_res, z0_res = get_expected_resolution(eta_arr[i])
        d0_sig[i] = abs_d0[i] / d0_res if d0_res > 0 else 0.0
        z0_sig[i] = abs_z0[i] / z0_res if z0_res > 0 else 0.0

    mean_abs_d0 = float(np.mean(abs_d0)) if abs_d0.size > 0 else 0.0
    max_abs_d0 = float(np.max(abs_d0)) if abs_d0.size > 0 else 0.0
    ptw_mean_abs_d0 = float(np.sum(pt_arr * abs_d0) / np.sum(pt_arr)) if np.sum(pt_arr) > 0 else 0.0

    mean_abs_z0 = float(np.mean(abs_z0)) if abs_z0.size > 0 else 0.0
    max_abs_z0 = float(np.max(abs_z0)) if abs_z0.size > 0 else 0.0
    ptw_mean_abs_z0 = float(np.sum(pt_arr * abs_z0) / np.sum(pt_arr)) if np.sum(pt_arr) > 0 else 0.0

    mean_d0_sig = float(np.mean(d0_sig)) if d0_sig.size > 0 else 0.0
    max_d0_sig = float(np.max(d0_sig)) if d0_sig.size > 0 else 0.0
    mean_z0_sig = float(np.mean(z0_sig)) if z0_sig.size > 0 else 0.0
    max_z0_sig = float(np.max(z0_sig)) if z0_sig.size > 0 else 0.0

    return [
        pt_obj,
        eta_obj,
        phi_obj,
        e_obj,
        sum_pt,
        n_tracks,
        pt1_over_ptobj,
        pt2_over_ptobj,
        dr1,
        dr2,
        *ring_features,
        max_pt,
        mean_dr,
        ptw_mean_dr,
        top2_sumpt_frac,
        n_tracks_core,
        sum_pt_core,
        iso_ratio,
        core_iso_ratio,
        maxpt_over_objpt,
        core_frac,
        mean_abs_d0,
        max_abs_d0,
        ptw_mean_abs_d0,
        mean_abs_z0,
        max_abs_z0,
        ptw_mean_abs_z0,
        mean_d0_sig,
        max_d0_sig,
        mean_z0_sig,
        max_z0_sig,
    ]


# dataset construction

def find_event_ids(data_dir: Path) -> list[int]:
    """Return sorted event ids discovered from photons_<id>.csv or .csv.gz."""
    ids = []
    for path in data_dir.iterdir():
        m = PHOTON_RE.match(path.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def resolve_event_file(data_dir: Path, stem: str, event_id: int) -> Path | None:
    """Return the existing csv or csv.gz path for one event file."""
    csv_path = data_dir / f"{stem}_{event_id}.csv"
    gz_path = data_dir / f"{stem}_{event_id}.csv.gz"

    if csv_path.exists():
        return csv_path
    if gz_path.exists():
        return gz_path
    return None


def build_dataset(data_dir: Path, iso_dr, trk_pt_min, include_converted=False, n_events=None):
    """Build the full feature matrix, label array, and event-group array."""
    event_ids = find_event_ids(data_dir)
    if n_events is not None:
        event_ids = event_ids[:n_events]

    X = []
    y = []
    groups = []

    n_events_seen = 0
    n_events_used = 0
    n_photons_used = 0
    n_jets_used = 0

    io_time_total = 0.0
    feature_time_total = 0.0
    expected_n_features = len(get_feature_names())

    for event_id in event_ids:
        n_events_seen += 1

        ph_file = resolve_event_file(data_dir, "photons", event_id)
        jet_file = resolve_event_file(data_dir, "jets", event_id)
        trk_file = resolve_event_file(data_dir, "tracks", event_id)

        if ph_file is None or jet_file is None or trk_file is None:
            continue

        t0_io = time.perf_counter()

        photons = read_noheader(ph_file, "photons")
        jets = read_noheader(jet_file, "jets")
        tracks = read_noheader(trk_file, "tracks")

        if not include_converted and not photons.empty and "conversionType" in photons.columns:
            photons = photons[photons["conversionType"] == 0].reset_index(drop=True)

        if photons.empty and jets.empty:
            io_time_total += time.perf_counter() - t0_io
            continue

        if not photons.empty:
            for c in ("pT", "eta", "phi", "e"):
                if c in photons.columns:
                    photons[c] = pd.to_numeric(photons[c], errors="coerce")
            photons = photons.dropna(subset=[c for c in ("pT", "eta", "phi", "e") if c in photons.columns])

        if not jets.empty:
            for c in ("pT", "eta", "phi", "e"):
                if c in jets.columns:
                    jets[c] = pd.to_numeric(jets[c], errors="coerce")
            jets = jets.dropna(subset=[c for c in ("pT", "eta", "phi", "e") if c in jets.columns])

        if not tracks.empty:
            for c in ("pT", "eta", "phi", "z0", "d0"):
                if c in tracks.columns:
                    tracks[c] = pd.to_numeric(tracks[c], errors="coerce")
            tracks = tracks.dropna(subset=[c for c in ("pT", "eta", "phi", "z0", "d0") if c in tracks.columns])

        io_time_total += time.perf_counter() - t0_io

        local_objects = 0

        for _, ph in photons.iterrows():
            t0_feat = time.perf_counter()
            feats = engineer_features(ph, tracks, iso_dr=iso_dr, trk_pt_min=trk_pt_min)
            feature_time_total += time.perf_counter() - t0_feat

            if feats is not None and len(feats) != expected_n_features:
                raise RuntimeError(
                    f"feature length mismatch: got {len(feats)}, expected {expected_n_features}"
                )
            if feats is None:
                continue

            X.append(feats)
            y.append(1)
            groups.append(event_id)
            n_photons_used += 1
            local_objects += 1

        for _, jet in jets.iterrows():
            t0_feat = time.perf_counter()
            feats = engineer_features(jet, tracks, iso_dr=iso_dr, trk_pt_min=trk_pt_min)
            feature_time_total += time.perf_counter() - t0_feat

            if feats is not None and len(feats) != expected_n_features:
                raise RuntimeError(
                    f"feature length mismatch: got {len(feats)}, expected {expected_n_features}"
                )
            if feats is None:
                continue

            X.append(feats)
            y.append(0)
            groups.append(event_id)
            n_jets_used += 1
            local_objects += 1

        if local_objects > 0:
            n_events_used += 1

    metadata = {
        "n_events_seen": n_events_seen,
        "n_events_used": n_events_used,
        "n_photons_used": n_photons_used,
        "n_jets_used": n_jets_used,
        "n_objects_used": n_photons_used + n_jets_used,
        "io_time_total_s": io_time_total,
        "feature_time_total_s": feature_time_total,
    }

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups),
        metadata,
    )


# evaluation utilities

def fake_rate_at_target_tpr(y_true, y_score, target_tpr=0.95):
    """Interpolate the fake rate at a requested true-positive rate."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.size == 0:
        return np.nan, np.nan, np.nan

    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return np.nan, np.nan, np.nan

    fpr, tpr, thr = roc_curve(y_true, y_score)

    order = np.argsort(tpr)
    tpr_s = tpr[order]
    fpr_s = fpr[order]
    thr_s = thr[order]

    if target_tpr <= tpr_s.min():
        idx = int(np.argmin(tpr_s))
        return float(fpr_s[idx]), float(thr_s[idx]), float(tpr_s[idx])

    if target_tpr >= tpr_s.max():
        idx = int(np.argmax(tpr_s))
        return float(fpr_s[idx]), float(thr_s[idx]), float(tpr_s[idx])

    fpr_at = float(np.interp(target_tpr, tpr_s, fpr_s))
    thr_at = float(np.interp(target_tpr, tpr_s, thr_s))
    return fpr_at, thr_at, float(target_tpr)


def roc_points_df(y_true, y_score):
    """Return ROC points in dataframe form for saving and plotting."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return pd.DataFrame(
        {
            "threshold": thr.astype(float),
            "acceptance": tpr.astype(float),
            "fake_rate": fpr.astype(float),
        }
    )


def plot_fake_vs_acceptance(roc_df, out_path: Path, auc_val, model_label="XGBoost"):
    """Plot jet fake rate against photon acceptance and save the figure."""
    start_plot()
    plt.plot(
        roc_df["fake_rate"],
        roc_df["acceptance"],
        label=f"{model_label} (AUC = {auc_val:.3f})",
    )
    plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Random classifier")
    plt.xlabel("Jet fake rate")
    plt.ylabel("Photon acceptance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    finish_plot(out_path)


def permutation_importance_auc(model, X_val, y_val, feature_names, rng_seed=42):
    """Estimate permutation importance using the validation AUC drop."""
    rng = np.random.default_rng(rng_seed)

    base_pred = model.predict_proba(X_val)[:, 1]
    base_auc = float(roc_auc_score(y_val, base_pred))

    rows = []
    X_work = X_val.copy()

    for j, name in enumerate(feature_names):
        original = X_work[:, j].copy()
        shuffled = original.copy()
        rng.shuffle(shuffled)
        X_work[:, j] = shuffled

        pred = model.predict_proba(X_work)[:, 1]
        auc = float(roc_auc_score(y_val, pred))
        rows.append({"feature": name, "importance_auc_drop": base_auc - auc})

        X_work[:, j] = original

    return pd.DataFrame(rows).sort_values("importance_auc_drop", ascending=False), base_auc


# optional feature dropping without rebuilding the dataset

def select_feature_subset(X, feature_names, drop_features):
    """Remove named features from an existing feature matrix."""
    drop_features = set(drop_features)
    keep_idx = [i for i, name in enumerate(feature_names) if name not in drop_features]
    kept_names = [feature_names[i] for i in keep_idx]
    X_sel = X[:, keep_idx]
    return X_sel, kept_names


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the XGBoost photon-versus-jet classifier."
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
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="directory used to store cached engineered datasets",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=10000,
        help="maximum number of events to use when building the dataset",
    )
    parser.add_argument(
        "--include-converted",
        action="store_true",
        default=False,
        help="include converted photons (default: exclude them)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="single_run",
        help="subfolder name for this run",
    )
    return parser.parse_args()


def main():
    """Run dataset construction, training, evaluation, and output saving."""
    total_start = time.perf_counter()
    args = parse_args()

    np.random.seed(args.seed)

    # feature-building and evaluation settings
    ISO_DR = 0.20
    TRK_PT_MIN = 0.75
    TARGET_TPRS = [0.80, 0.90, 0.95]

    EXPERIMENT = "xgboost_classifier"
    CACHE_TAG = "xgboost"

    # keep available in case feature ablations are needed later
    DROP_FEATURES = []

    data_dir = args.data_dir
    base_out_dir = args.out_dir
    out_dir = base_out_dir / args.run_tag
    cache_dir = args.cache_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    conv_tag = "inclconv" if args.include_converted else "exclconv"
    cache_name = (
        f"dataset_{CACHE_TAG}_"
        f"iso{str(ISO_DR).replace('.', 'p')}_"
        f"pt{str(TRK_PT_MIN).replace('.', 'p')}_"
        f"{conv_tag}.npz"
    )
    cache_path = cache_dir / cache_name

    feature_names = get_feature_names()

    build_time_total = 0.0
    build_io_time_total = 0.0
    build_feature_time_total = 0.0
    dataset_meta = None
    loaded_from_cache = False

    t0_dataset = time.perf_counter()

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        groups = data["groups"]
        cached_feature_names = list(data["feature_names"])
        print(f"loaded cached dataset from {cache_path}")
        loaded_from_cache = True

        if cached_feature_names != feature_names:
            raise RuntimeError(
                "cached feature names do not match current get_feature_names(). "
                "delete the cache file and rebuild."
            )
    else:
        X, y, groups, dataset_meta = build_dataset(
            data_dir,
            iso_dr=ISO_DR,
            trk_pt_min=TRK_PT_MIN,
            include_converted=args.include_converted,
            n_events=args.n_events,
        )
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            groups=groups,
            feature_names=np.array(feature_names, dtype=object),
        )
        print(f"saved cached dataset to {cache_path}")

        build_time_total = time.perf_counter() - t0_dataset
        build_io_time_total = dataset_meta["io_time_total_s"]
        build_feature_time_total = dataset_meta["feature_time_total_s"]

    if loaded_from_cache:
        dataset_load_time_total = time.perf_counter() - t0_dataset
    else:
        dataset_load_time_total = 0.0

    t0_drop = time.perf_counter()
    if DROP_FEATURES:
        X, feature_names = select_feature_subset(X, feature_names, DROP_FEATURES)
        print("dropped features:", DROP_FEATURES)
    drop_feature_time_total = time.perf_counter() - t0_drop

    print("data directory:", data_dir)
    print("output directory:", out_dir)
    print("cache directory:", cache_dir)
    print("dataset shape:", X.shape)
    print("positives (photons):", int(y.sum()), "negatives (jets):", int((1 - y).sum()))
    print("feature names:", feature_names)
    print("converted photons:", "included" if args.include_converted else "excluded")
    print("seed:", args.seed)
    print("run tag:", args.run_tag)
    print("n_events requested:", args.n_events)
    print("working points:", [int(round(100 * t)) for t in TARGET_TPRS])

    if len(y) == 0:
        raise RuntimeError("no samples built. check file naming and data_dir")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx]

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=123)
    tr_idx, va_idx = next(splitter2.split(X_train_full, y_train_full, groups=groups_train_full))

    X_train, X_val = X_train_full[tr_idx], X_train_full[va_idx]
    y_train, y_val = y_train_full[tr_idx], y_train_full[va_idx]

    t0_model = time.perf_counter()
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=1,
    )
    model_build_time_total = time.perf_counter() - t0_model

    t0_train = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    training_time_total = time.perf_counter() - t0_train

    t0_post = time.perf_counter()

    t0_pred_val = time.perf_counter()
    y_val_pred = model.predict_proba(X_val)[:, 1]
    inference_time_val_total = time.perf_counter() - t0_pred_val

    t0_pred_test = time.perf_counter()
    y_test_pred = model.predict_proba(X_test)[:, 1]
    inference_time_test_total = time.perf_counter() - t0_pred_test

    pd.DataFrame({
        "label": y_val,
        "score": y_val_pred,
    }).to_csv(out_dir / "scores_val.csv", index=False)

    pd.DataFrame({
        "label": y_test,
        "score": y_test_pred,
    }).to_csv(out_dir / "scores_test.csv", index=False)

    val_auc = float(roc_auc_score(y_val, y_val_pred))
    test_auc = float(roc_auc_score(y_test, y_test_pred))

    roc_val = roc_points_df(y_val, y_val_pred)
    roc_test = roc_points_df(y_test, y_test_pred)

    roc_val.to_csv(out_dir / "roc_val.csv", index=False)
    roc_test.to_csv(out_dir / "roc_test.csv", index=False)

    plot_fake_vs_acceptance(
        roc_val,
        out_dir / "fake_vs_acceptance_val.png",
        val_auc,
    )

    plot_fake_vs_acceptance(
        roc_test,
        out_dir / "fake_vs_acceptance_test.png",
        test_auc,
    )

    dataset_df = pd.DataFrame(X, columns=feature_names)
    dataset_df["label"] = y
    dataset_df.to_csv(out_dir / "engineered_dataset.csv", index=False)

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "feature_importances_gain.csv", index=False)

    start_plot()
    top_imp = importances.head(15).iloc[::-1]
    plt.barh(top_imp["feature"], top_imp["importance"])
    plt.xlabel("XGBoost feature importance")
    plt.ylabel("Feature")
    finish_plot(out_dir / "feature_importances_top15.png")

    t0_perm = time.perf_counter()
    perm_df, base_auc_perm = permutation_importance_auc(model, X_val, y_val, feature_names)
    permutation_time_total = time.perf_counter() - t0_perm

    perm_df.to_csv(out_dir / "feature_importances_permutation_auc.csv", index=False)

    start_plot()
    top_perm = perm_df.head(15).iloc[::-1]
    plt.barh(top_perm["feature"], top_perm["importance_auc_drop"])
    plt.xlabel("Validation AUC drop after shuffling")
    plt.ylabel("Feature")
    finish_plot(out_dir / "feature_importances_permutation_top15.png")

    fixed_wp_rows = []
    for tpr in TARGET_TPRS:
        val_fpr, val_thr, val_achieved = fake_rate_at_target_tpr(y_val, y_val_pred, target_tpr=tpr)
        test_fpr, test_thr, test_achieved = fake_rate_at_target_tpr(y_test, y_test_pred, target_tpr=tpr)
        fixed_wp_rows.append(
            {
                "target_tpr": float(tpr),
                "val_fake_rate": float(val_fpr),
                "val_threshold": float(val_thr),
                "val_achieved_tpr": float(val_achieved),
                "test_fake_rate": float(test_fpr),
                "test_threshold": float(test_thr),
                "test_achieved_tpr": float(test_achieved),
            }
        )
    pd.DataFrame(fixed_wp_rows).to_csv(out_dir / "fixed_tpr_metrics.csv", index=False)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("This run uses an XGBoost model.\n")
        f.write(f"Experiment: {EXPERIMENT}\n")
        f.write("Model: XGBClassifier\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Run tag: {args.run_tag}\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Cache file: {cache_path}\n")
        f.write(f"n_events requested: {args.n_events}\n")
        f.write(f"Converted photons: {'included' if args.include_converted else 'excluded'}\n")

        if DROP_FEATURES:
            f.write("Dropped features:\n")
            for name in DROP_FEATURES:
                f.write(f"{name}\n")

        f.write("Feature set:\n")
        for name in feature_names:
            f.write(f"{name}\n")

        f.write("\nModel hyperparameters:\n")
        f.write("n_estimators: 500\n")
        f.write("learning_rate: 0.05\n")
        f.write("max_depth: 6\n")
        f.write("min_child_weight: 5\n")
        f.write("subsample: 0.8\n")
        f.write("colsample_bytree: 0.8\n")
        f.write("tree_method: hist\n")

        f.write(f"\nVal AUC: {val_auc}\n")
        f.write(f"Test AUC: {test_auc}\n")
        f.write(f"Permutation-importance base validation AUC: {base_auc_perm}\n")

        f.write("\nSaved outputs:\n")
        f.write("scores_val.csv\n")
        f.write("scores_test.csv\n")
        f.write("roc_val.csv\n")
        f.write("roc_test.csv\n")
        f.write("fake_vs_acceptance_val.png\n")
        f.write("fake_vs_acceptance_test.png\n")
        f.write("engineered_dataset.csv\n")
        f.write("feature_importances_gain.csv\n")
        f.write("feature_importances_top15.png\n")
        f.write("feature_importances_permutation_auc.csv\n")
        f.write("feature_importances_permutation_top15.png\n")
        f.write("fixed_tpr_metrics.csv\n")

        for row in fixed_wp_rows:
            wp = int(round(row["target_tpr"] * 100))
            f.write(f"\nVal fake@{wp}: {row['val_fake_rate']}\n")
            f.write(f"Test fake@{wp}: {row['test_fake_rate']}\n")
            f.write(f"Val threshold at {wp}% TPR: {row['val_threshold']}\n")
            f.write(f"Test threshold at {wp}% TPR: {row['test_threshold']}\n")

        f.write("\nTiming summary:\n")
        f.write(f"Loaded from cache: {loaded_from_cache}\n")
        f.write(f"Dataset load/build total [s]: {dataset_load_time_total if loaded_from_cache else build_time_total}\n")
        f.write(f"Dataset build io total [s]: {build_io_time_total}\n")
        f.write(f"Dataset build feature total [s]: {build_feature_time_total}\n")
        f.write(f"Drop-feature time [s]: {drop_feature_time_total}\n")
        f.write(f"Model build time [s]: {model_build_time_total}\n")
        f.write(f"Training time [s]: {training_time_total}\n")
        f.write(f"Validation inference time [s]: {inference_time_val_total}\n")
        f.write(f"Test inference time [s]: {inference_time_test_total}\n")
        f.write(f"Permutation importance time [s]: {permutation_time_total}\n")

    postproc_time_total = time.perf_counter() - t0_post
    total_wall_time = time.perf_counter() - total_start

    n_val = len(X_val)
    n_test = len(X_test)

    feature_build_time_per_object = (
        build_feature_time_total / dataset_meta["n_objects_used"]
        if (dataset_meta is not None and dataset_meta["n_objects_used"] > 0)
        else np.nan
    )
    inference_time_val_per_object = inference_time_val_total / n_val if n_val > 0 else np.nan
    inference_time_test_per_object = inference_time_test_total / n_test if n_test > 0 else np.nan
    feature_plus_test_inference_per_object = (
        feature_build_time_per_object + inference_time_test_per_object
        if np.isfinite(feature_build_time_per_object) and np.isfinite(inference_time_test_per_object)
        else np.nan
    )

    timing_df = pd.DataFrame([{
        "experiment": EXPERIMENT,
        "loaded_from_cache": bool(loaded_from_cache),
        "n_total_objects": int(len(X)),
        "n_train_objects": int(len(X_train)),
        "n_val_objects": int(n_val),
        "n_test_objects": int(n_test),
        "n_features_after_drop": int(len(feature_names)),
        "n_events_requested": int(args.n_events),
        "dataset_load_time_total_s": float(dataset_load_time_total),
        "dataset_build_time_total_s": float(build_time_total),
        "dataset_build_io_time_total_s": float(build_io_time_total),
        "dataset_build_feature_time_total_s": float(build_feature_time_total),
        "drop_feature_time_total_s": float(drop_feature_time_total),
        "model_build_time_total_s": float(model_build_time_total),
        "training_time_total_s": float(training_time_total),
        "inference_time_val_total_s": float(inference_time_val_total),
        "inference_time_test_total_s": float(inference_time_test_total),
        "permutation_time_total_s": float(permutation_time_total),
        "postproc_time_total_s": float(postproc_time_total),
        "total_wall_time_s": float(total_wall_time),
        "feature_build_time_per_object_s": float(feature_build_time_per_object),
        "inference_time_val_per_object_s": float(inference_time_val_per_object),
        "inference_time_test_per_object_s": float(inference_time_test_per_object),
        "feature_plus_test_inference_per_object_s": float(feature_plus_test_inference_per_object),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "iso_dr": float(ISO_DR),
        "trk_pt_min": float(TRK_PT_MIN),
        "converted_photons": "included" if args.include_converted else "excluded",
    }])
    timing_df.to_csv(out_dir / "timing_summary.csv", index=False)

    print("\nresults")
    print("This is an XGBoost model.")
    print("Features used:")
    for name in feature_names:
        print(" ", name)

    print(f"\nVal AUC: {val_auc:.6f}")
    print(f"Test AUC: {test_auc:.6f}")

    for row in fixed_wp_rows:
        wp = int(round(row["target_tpr"] * 100))
        print(f"\nWorking point = {wp}% photon efficiency")
        print(f"Val fake rate: {row['val_fake_rate']:.6f}")
        print(f"Test fake rate: {row['test_fake_rate']:.6f}")
        print(f"Val threshold: {row['val_threshold']:.6f}")
        print(f"Test threshold: {row['test_threshold']:.6f}")

    print("\ntiming")
    print(f"loaded_from_cache: {loaded_from_cache}")
    if dataset_meta is not None:
        print(f"dataset_build_events_seen: {dataset_meta['n_events_seen']}")
        print(f"dataset_build_events_used: {dataset_meta['n_events_used']}")
        print(f"dataset_build_photons_used: {dataset_meta['n_photons_used']}")
        print(f"dataset_build_jets_used: {dataset_meta['n_jets_used']}")
        print(f"dataset_build_objects_used: {dataset_meta['n_objects_used']}")
        print(f"dataset_build_io_time_total: {build_io_time_total:.6f} s")
        print(f"dataset_build_feature_time_total: {build_feature_time_total:.6f} s")
        print(f"dataset_build_time_total: {build_time_total:.6f} s")
        print(f"feature_build_time_per_object: {feature_build_time_per_object:.6e} s")
    else:
        print(f"dataset_load_time_total: {dataset_load_time_total:.6f} s")

    print(f"drop_feature_time_total: {drop_feature_time_total:.6f} s")
    print(f"model_build_time_total: {model_build_time_total:.6f} s")
    print(f"training_time_total: {training_time_total:.6f} s")
    print(f"inference_time_val_total: {inference_time_val_total:.6f} s")
    print(f"inference_time_test_total: {inference_time_test_total:.6f} s")
    print(f"inference_time_val_per_object: {inference_time_val_per_object:.6e} s")
    print(f"inference_time_test_per_object: {inference_time_test_per_object:.6e} s")
    print(f"feature_plus_test_inference_per_object: {feature_plus_test_inference_per_object:.6e} s")
    print(f"permutation_time_total: {permutation_time_total:.6f} s")
    print(f"postproc_time_total: {postproc_time_total:.6f} s")
    print(f"total_wall_time: {total_wall_time:.6f} s")


if __name__ == "__main__":
    main()