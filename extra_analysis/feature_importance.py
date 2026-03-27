#!/usr/bin/env python3
"""
Permutation-based feature-ranking study for the three main classifiers used in
the project: XGBoost, histogram-based gradient boosting, and a feed-forward
neural network.

The script:
- reads headerless photon, jet, and track files
- builds a per-object feature matrix using the same engineered features as the
  classifier scripts
- performs group-aware train/validation splitting by event
- trains each model using settings matched to the main classifier scripts
- evaluates permutation importance on the validation sample using the drop in
  validation AUC
- saves per-model rankings, plots, and combined summary tables

The intended use is to support the discussion of which features drive the
classifier performance in the Senior Honours Project study of fast, track-based
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
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# disable oneDNN threading optimisations for more consistent CPU behaviour
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
DEFAULT_OUT_DIR = PROJECT_ROOT / "final_outputs" / "feature_ranking"
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


def delta_r(eta1, phi1, eta2, phi2):
    """Return the angular separation delta-R between two eta-phi coordinates."""
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


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


def _ring_stats(assoc: pd.DataFrame, ring_edges):
    """Return track multiplicities and scalar pT sums in radial rings."""
    out = []
    for lo, hi in ring_edges:
        sub = assoc[(assoc["dR"] >= lo) & (assoc["dR"] < hi)]
        out.append(float(len(sub)))
        out.append(float(sub["pT"].sum()) if len(sub) else 0.0)
    return out


# feature engineering

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
        """Return a zero-padded vector if no associated tracks are available."""
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

    tracks = tracks[tracks["pT"] > trk_pt_min].copy()
    if tracks.empty:
        return padded_zero_features()

    tracks["dR"] = tracks.apply(
        lambda row: delta_r(eta_obj, phi_obj, float(row["eta"]), float(row["phi"])),
        axis=1,
    )

    assoc = tracks[tracks["dR"] < iso_dr].copy()
    if assoc.empty:
        return padded_zero_features()

    assoc = assoc.sort_values("pT", ascending=False).reset_index(drop=True)

    sum_pt = float(assoc["pT"].sum())
    n_tracks = float(len(assoc))

    pt1 = float(assoc.iloc[0]["pT"]) if len(assoc) >= 1 else 0.0
    pt2 = float(assoc.iloc[1]["pT"]) if len(assoc) >= 2 else 0.0

    dr1 = float(assoc.iloc[0]["dR"]) if len(assoc) >= 1 else 0.0
    dr2 = float(assoc.iloc[1]["dR"]) if len(assoc) >= 2 else 0.0

    pt1_over_ptobj = pt1 / pt_obj
    pt2_over_ptobj = pt2 / pt_obj

    ring_features = _ring_stats(assoc, ring_edges)

    max_pt = pt1
    mean_dr = float(assoc["dR"].mean()) if len(assoc) else 0.0
    ptw_mean_dr = float((assoc["pT"] * assoc["dR"]).sum() / sum_pt) if sum_pt > 0 else 0.0
    top2_sumpt_frac = float((pt1 + pt2) / sum_pt) if sum_pt > 0 else 0.0

    core = assoc[assoc["dR"] < 0.05]
    n_tracks_core = float(len(core))
    sum_pt_core = float(core["pT"].sum()) if len(core) else 0.0

    iso_ratio = float(sum_pt / pt_obj) if pt_obj > 0 else 0.0
    core_iso_ratio = float(sum_pt_core / pt_obj) if pt_obj > 0 else 0.0
    maxpt_over_objpt = float(max_pt / pt_obj) if pt_obj > 0 else 0.0
    core_frac = float(sum_pt_core / sum_pt) if sum_pt > 0 else 0.0

    pt_arr = assoc["pT"].to_numpy(dtype=float)
    abs_d0 = np.abs(assoc["d0"].to_numpy(dtype=float))
    abs_z0 = np.abs(assoc["z0"].to_numpy(dtype=float))
    etas = assoc["eta"].to_numpy(dtype=float)

    d0_sig = []
    z0_sig = []

    for i in range(len(etas)):
        d0_res, z0_res = get_expected_resolution(etas[i])
        d0_sig.append(abs_d0[i] / d0_res if d0_res > 0 else 0.0)
        z0_sig.append(abs_z0[i] / z0_res if z0_res > 0 else 0.0)

    d0_sig = np.array(d0_sig, dtype=float)
    z0_sig = np.array(z0_sig, dtype=float)

    mean_abs_d0 = float(abs_d0.mean()) if abs_d0.size > 0 else 0.0
    max_abs_d0 = float(abs_d0.max()) if abs_d0.size > 0 else 0.0
    ptw_mean_abs_d0 = float(np.sum(pt_arr * abs_d0) / np.sum(pt_arr)) if np.sum(pt_arr) > 0 else 0.0

    mean_abs_z0 = float(abs_z0.mean()) if abs_z0.size > 0 else 0.0
    max_abs_z0 = float(abs_z0.max()) if abs_z0.size > 0 else 0.0
    ptw_mean_abs_z0 = float(np.sum(pt_arr * abs_z0) / np.sum(pt_arr)) if np.sum(pt_arr) > 0 else 0.0

    mean_d0_sig = float(d0_sig.mean()) if d0_sig.size > 0 else 0.0
    max_d0_sig = float(d0_sig.max()) if d0_sig.size > 0 else 0.0
    mean_z0_sig = float(z0_sig.mean()) if z0_sig.size > 0 else 0.0
    max_z0_sig = float(z0_sig.max()) if z0_sig.size > 0 else 0.0

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
    """Build the full feature matrix, labels, and event-group array."""
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


# neural-network validation callback

class ValMetricsCallback(keras.callbacks.Callback):
    """Compute validation AUC and fake rate at a chosen working point each epoch."""

    def __init__(self, x_val, y_val, target_tpr=0.95):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.target_tpr = float(target_tpr)
        self.metric_name = f"val_fake{int(round(100 * self.target_tpr))}"

    def on_epoch_end(self, epoch, logs=None):
        """Evaluate the current network on the validation set after each epoch."""
        logs = logs or {}
        y_hat = self.model.predict(self.x_val, verbose=0).ravel()
        val_auc = float(roc_auc_score(self.y_val, y_hat))
        val_fpr, _, _ = fake_rate_at_target_tpr(self.y_val, y_hat, target_tpr=self.target_tpr)
        logs["val_auc_manual"] = val_auc
        logs[self.metric_name] = float(val_fpr)
        print(
            f" - val_auc_manual: {val_auc:.6f} - {self.metric_name}: {val_fpr:.6f}",
            end="",
        )


# neural network

def build_model(n_features, learning_rate=1e-3, l2=1e-4, dropout=0.20):
    """Construct and compile the feed-forward neural-network classifier."""
    reg = keras.regularizers.l2(l2)

    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(128, kernel_regularizer=reg, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout),

            layers.Dense(64, kernel_regularizer=reg, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout),

            layers.Dense(32, kernel_regularizer=reg, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout),

            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


class NNWrapper:
    """Minimal wrapper so the NN can be used with the shared permutation code."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        """Return two-column probabilities to match sklearn classifier output."""
        p = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1.0 - p, p])


# optional feature dropping

def select_feature_subset(X, feature_names, drop_features):
    """Remove named features from an existing feature matrix."""
    drop_features = set(drop_features)
    keep_idx = [i for i, name in enumerate(feature_names) if name not in drop_features]
    kept_names = [feature_names[i] for i in keep_idx]
    X_sel = X[:, keep_idx]
    return X_sel, kept_names


# permutation importance

def permutation_importance_auc(model, X_val, y_val, feature_names, rng_seed=42):
    """Estimate permutation importance using the drop in validation AUC."""
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
        rows.append({
            "feature": name,
            "importance_auc_drop": base_auc - auc,
        })

        X_work[:, j] = original

    df = pd.DataFrame(rows).sort_values("importance_auc_drop", ascending=False).reset_index(drop=True)
    return df, base_auc


def save_importance_plot(df, out_path: Path, top_n=15):
    """Save a horizontal bar chart of permutation importances."""
    plot_df = df.head(top_n).iloc[::-1]

    n_rows = len(plot_df)
    fig_height = max(7, 0.32 * n_rows)

    plt.figure(figsize=(10, fig_height))
    plt.barh(plot_df["feature"], plot_df["importance_auc_drop"])
    plt.xlabel("Validation AUC drop after shuffling")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_combined_top_features_plot(combined_df, out_path: Path, top_n=10):
    """Save a grouped horizontal comparison plot for the top features per model."""
    top_features = (
        combined_df.groupby("feature")["importance_auc_drop"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    plot_df = combined_df[combined_df["feature"].isin(top_features)].copy()
    table = plot_df.pivot(index="feature", columns="model", values="importance_auc_drop").fillna(0.0)
    table = table.loc[table.max(axis=1).sort_values().index]

    y = np.arange(len(table.index))
    width = 0.25
    cols = list(table.columns)

    start_plot()
    if len(cols) >= 1:
        plt.barh(y - width, table[cols[0]], height=width, label=cols[0])
    if len(cols) >= 2:
        plt.barh(y, table[cols[1]], height=width, label=cols[1])
    if len(cols) >= 3:
        plt.barh(y + width, table[cols[2]], height=width, label=cols[2])

    plt.yticks(y, table.index)
    plt.xlabel("Validation AUC drop after shuffling")
    plt.ylabel("Feature")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    finish_plot(out_path)


# output helpers

def save_model_importance_outputs(model_dir: Path, model_label, df, base_auc):
    """Save per-model ranking tables, plots, and a compact summary."""
    model_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        model_dir / "feature_importances_permutation_auc_all.csv",
        index=False,
    )

    save_importance_plot(
        df,
        out_path=model_dir / "feature_importances_permutation_all.png",
        top_n=len(df),
    )

    save_importance_plot(
        df,
        out_path=model_dir / "feature_importances_permutation_top15.png",
        top_n=15,
    )

    summary = pd.DataFrame([{
        "model": model_label,
        "base_val_auc_for_permutation": base_auc,
        "n_features": len(df),
    }])
    summary.to_csv(
        model_dir / "feature_importance_summary.csv",
        index=False,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the permutation-based feature-ranking study for all three classifiers."
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing photons_*, jets_*, and tracks_* files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory to save ranking outputs",
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
    return parser.parse_args()


def main():
    """Run the full feature-ranking study for all three classifiers."""
    total_start = time.perf_counter()
    args = parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    ISO_DR = 0.20
    TRK_PT_MIN = 0.75
    TARGET_TPRS = [0.80, 0.90, 0.95]
    PRIMARY_TPR = 0.95

    LR = 1e-3
    L2 = 1e-4
    DROPOUT = 0.20
    EPOCHS = 35
    BATCH_SIZE = 512
    PATIENCE = 12

    DROP_FEATURES = []

    base_out_dir = args.out_dir
    out_dir = base_out_dir / args.run_tag
    cache_dir = args.cache_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    conv_tag = "inclconv" if args.include_converted else "exclconv"
    cache_name = (
        f"dataset_feature_ranking_"
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
            args.data_dir,
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

    if DROP_FEATURES:
        X, feature_names = select_feature_subset(X, feature_names, DROP_FEATURES)
        print("dropped features:", DROP_FEATURES)
    else:
        print("dropped features: []")

    print("data directory:", args.data_dir)
    print("output directory:", out_dir)
    print("cache directory:", cache_dir)
    print("Dataset shape:", X.shape)
    print("positives (photons):", int(y.sum()), "negatives (jets):", int((1 - y).sum()))
    print("feature names:", feature_names)
    print("converted photons:", "included" if args.include_converted else "excluded")
    print("seed:", args.seed)
    print("run tag:", args.run_tag)
    print("n_events requested:", args.n_events)
    print("working points:", [int(round(100 * t)) for t in TARGET_TPRS])
    print("primary working point:", int(round(100 * PRIMARY_TPR)))

    if len(y) == 0:
        raise RuntimeError("no samples built. check file naming and data_dir")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train_full, _ = X[train_idx], X[test_idx]
    y_train_full, _ = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx]

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed + 1)
    tr_idx, va_idx = next(splitter2.split(X_train_full, y_train_full, groups=groups_train_full))

    X_train, X_val = X_train_full[tr_idx], X_train_full[va_idx]
    y_train, y_val = y_train_full[tr_idx], y_train_full[va_idx]

    combined_rows = []
    summary_rows = []

    # xgboost
    xgb_dir = out_dir / "xgboost_classifier"
    xgb_dir.mkdir(parents=True, exist_ok=True)

    xgb_model = XGBClassifier(
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

    print("\ntraining xgboost_classifier ...")
    xgb_t0 = time.perf_counter()
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_train_time = time.perf_counter() - xgb_t0

    print("running permutation importance for xgboost_classifier ...")
    xgb_perm_t0 = time.perf_counter()
    xgb_perm_df, xgb_base_auc = permutation_importance_auc(
        xgb_model,
        X_val,
        y_val,
        feature_names,
        rng_seed=args.seed,
    )
    xgb_perm_time = time.perf_counter() - xgb_perm_t0

    save_model_importance_outputs(
        xgb_dir,
        "XGBoost",
        xgb_perm_df,
        xgb_base_auc,
    )

    for _, row in xgb_perm_df.iterrows():
        combined_rows.append({
            "model": "XGBoost",
            "feature": row["feature"],
            "importance_auc_drop": row["importance_auc_drop"],
        })

    summary_rows.append({
        "model": "XGBoost",
        "base_val_auc_for_permutation": xgb_base_auc,
        "n_features": len(feature_names),
        "training_time_total_s": xgb_train_time,
        "permutation_time_total_s": xgb_perm_time,
    })

    # treeHGB
    hgb_dir = out_dir / "treeHGB_classifier"
    hgb_dir.mkdir(parents=True, exist_ok=True)

    hgb_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1e-3,
        early_stopping=False,
        random_state=args.seed,
    )

    print("\ntraining treeHGB_classifier ...")
    hgb_t0 = time.perf_counter()
    hgb_model.fit(X_train, y_train)
    hgb_train_time = time.perf_counter() - hgb_t0

    print("running permutation importance for treeHGB_classifier ...")
    hgb_perm_t0 = time.perf_counter()
    hgb_perm_df, hgb_base_auc = permutation_importance_auc(
        hgb_model,
        X_val,
        y_val,
        feature_names,
        rng_seed=args.seed,
    )
    hgb_perm_time = time.perf_counter() - hgb_perm_t0

    save_model_importance_outputs(
        hgb_dir,
        "HistGradientBoosting",
        hgb_perm_df,
        hgb_base_auc,
    )

    for _, row in hgb_perm_df.iterrows():
        combined_rows.append({
            "model": "HistGradientBoosting",
            "feature": row["feature"],
            "importance_auc_drop": row["importance_auc_drop"],
        })

    summary_rows.append({
        "model": "HistGradientBoosting",
        "base_val_auc_for_permutation": hgb_base_auc,
        "n_features": len(feature_names),
        "training_time_total_s": hgb_train_time,
        "permutation_time_total_s": hgb_perm_time,
    })

    # neural network
    nn_dir = out_dir / "nn_classifier"
    nn_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_val_nn = scaler.transform(X_val)

    nn_model = build_model(
        n_features=X_train_nn.shape[1],
        learning_rate=LR,
        l2=L2,
        dropout=DROPOUT,
    )

    val_cb = ValMetricsCallback(
        x_val=X_val_nn,
        y_val=y_val,
        target_tpr=PRIMARY_TPR,
    )

    es = keras.callbacks.EarlyStopping(
        monitor=val_cb.metric_name,
        mode="min",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\ntraining nn_classifier ...")
    nn_t0 = time.perf_counter()
    nn_model.fit(
        X_train_nn,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_nn, y_val),
        verbose=0,
        callbacks=[val_cb, es],
        class_weight=None,
    )
    nn_train_time = time.perf_counter() - nn_t0

    nn_wrapper = NNWrapper(nn_model)

    print("\nrunning permutation importance for nn_classifier ...")
    nn_perm_t0 = time.perf_counter()
    nn_perm_df, nn_base_auc = permutation_importance_auc(
        nn_wrapper,
        X_val_nn,
        y_val,
        feature_names,
        rng_seed=args.seed,
    )
    nn_perm_time = time.perf_counter() - nn_perm_t0

    save_model_importance_outputs(
        nn_dir,
        "Neural network",
        nn_perm_df,
        nn_base_auc,
    )

    for _, row in nn_perm_df.iterrows():
        combined_rows.append({
            "model": "Neural network",
            "feature": row["feature"],
            "importance_auc_drop": row["importance_auc_drop"],
        })

    summary_rows.append({
        "model": "Neural network",
        "base_val_auc_for_permutation": nn_base_auc,
        "n_features": len(feature_names),
        "training_time_total_s": nn_train_time,
        "permutation_time_total_s": nn_perm_time,
    })

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(
        out_dir / "combined_feature_importances_permutation_auc_all_models.csv",
        index=False,
    )

    save_combined_top_features_plot(
        combined_df,
        out_dir / "combined_feature_importances_top10.png",
        top_n=10,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        out_dir / "feature_importance_summary.csv",
        index=False,
    )

    total_wall_time = time.perf_counter() - total_start

    timing_df = pd.DataFrame([{
        "loaded_from_cache": bool(loaded_from_cache),
        "n_events_requested": int(args.n_events),
        "dataset_load_time_total_s": float(dataset_load_time_total),
        "dataset_build_time_total_s": float(build_time_total),
        "dataset_build_io_time_total_s": float(build_io_time_total),
        "dataset_build_feature_time_total_s": float(build_feature_time_total),
        "xgboost_training_time_total_s": float(xgb_train_time),
        "xgboost_permutation_time_total_s": float(xgb_perm_time),
        "treehgb_training_time_total_s": float(hgb_train_time),
        "treehgb_permutation_time_total_s": float(hgb_perm_time),
        "nn_training_time_total_s": float(nn_train_time),
        "nn_permutation_time_total_s": float(nn_perm_time),
        "total_wall_time_s": float(total_wall_time),
        "iso_dr": float(ISO_DR),
        "trk_pt_min": float(TRK_PT_MIN),
        "converted_photons": "included" if args.include_converted else "excluded",
    }])
    timing_df.to_csv(
        out_dir / "timing_summary.csv",
        index=False,
    )

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("Permutation-based feature-ranking study for the three main classifiers.\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Run tag: {args.run_tag}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Cache file: {cache_path}\n")
        f.write(f"n_events requested: {args.n_events}\n")
        f.write(f"Converted photons: {'included' if args.include_converted else 'excluded'}\n")
        f.write(f"Working points: {', '.join(str(int(round(100 * t))) for t in TARGET_TPRS)}\n")
        f.write(f"Primary NN working point: {int(round(100 * PRIMARY_TPR))}\n")
        f.write(f"Number of features after drops: {len(feature_names)}\n")

        f.write("\nDropped features:\n")
        if DROP_FEATURES:
            for name in DROP_FEATURES:
                f.write(f"{name}\n")
        else:
            f.write("none\n")

        f.write("\nBase validation AUC used for permutation:\n")
        f.write(f"XGBoost: {xgb_base_auc}\n")
        f.write(f"HistGradientBoosting: {hgb_base_auc}\n")
        f.write(f"Neural network: {nn_base_auc}\n")

        f.write("\nTiming summary [s]:\n")
        f.write(f"dataset load total: {dataset_load_time_total:.3f}\n")
        f.write(f"dataset build total: {build_time_total:.3f}\n")
        f.write(f"dataset build io total: {build_io_time_total:.3f}\n")
        f.write(f"dataset build feature total: {build_feature_time_total:.3f}\n")
        f.write(f"XGBoost training: {xgb_train_time:.3f}\n")
        f.write(f"XGBoost permutation: {xgb_perm_time:.3f}\n")
        f.write(f"HistGradientBoosting training: {hgb_train_time:.3f}\n")
        f.write(f"HistGradientBoosting permutation: {hgb_perm_time:.3f}\n")
        f.write(f"Neural network training: {nn_train_time:.3f}\n")
        f.write(f"Neural network permutation: {nn_perm_time:.3f}\n")
        f.write(f"total wall time: {total_wall_time:.3f}\n")

        f.write("\nSaved outputs:\n")
        f.write("combined_feature_importances_permutation_auc_all_models.csv\n")
        f.write("combined_feature_importances_top10.png\n")
        f.write("feature_importance_summary.csv\n")
        f.write("timing_summary.csv\n")
        f.write("xgboost_classifier/feature_importances_permutation_auc_all.csv\n")
        f.write("xgboost_classifier/feature_importances_permutation_top15.png\n")
        f.write("treeHGB_classifier/feature_importances_permutation_auc_all.csv\n")
        f.write("treeHGB_classifier/feature_importances_permutation_top15.png\n")
        f.write("nn_classifier/feature_importances_permutation_auc_all.csv\n")
        f.write("nn_classifier/feature_importances_permutation_top15.png\n")

    print("\nfinished.")
    print("saved outputs to:")
    print(out_dir)


if __name__ == "__main__":
    main()