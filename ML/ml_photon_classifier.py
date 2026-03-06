#!/usr/bin/env python3
# imports
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# io utilities (headerless CSVs)
# -----------------------------

def _is_gzip_file(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def _is_tar_file(path: str) -> bool:
    # tar "ustar" signature typically appears at byte offset 257
    try:
        with open(path, "rb") as f:
            f.seek(257)
            return f.read(5) == b"ustar"
    except OSError:
        return False


def read_noheader(path: str, kind: str) -> pd.DataFrame:
    try:
        if os.path.getsize(path) == 0:
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
    except pd.errors.EmptyDataError:
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
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    if df.empty:
        return df

    if kind in ("photons", "jets"):
        cols = ["pT", "eta", "phi", "e", "conversionType"]
        df.columns = cols[: len(df.columns)]
        keep = ["pT", "eta", "phi", "e"][: len(df.columns)]
    elif kind == "tracks":
        cols = ["pT", "eta", "phi", "eTot", "z0", "d0"]
        df.columns = cols[: len(df.columns)]
        keep = ["pT", "eta", "phi"][: len(df.columns)]
    else:
        raise ValueError(f"unknown kind: {kind}")

    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=keep)

    return df


# -----------------------------
# geometry utilities
# -----------------------------

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


# -----------------------------
# feature names
# -----------------------------

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
        "dr1_over_ptobj",
        "dr2_over_ptobj",
    ]


# -----------------------------
# feature engineering
# -----------------------------

def engineer_features(obj, tracks, iso_dr=0.30, trk_pt_min=1.0):
    """
    Features aligned to weekly task:
      - object 4-vector: pT, eta, phi, e
      - sum_pt
      - N_tracks
      - pt_highest_track / pt_object
      - pt_2_highest_track / pt_object
      - dR_highest_track / pt_object
      - dR_2_highest_track / pt_object

    Notes:
      - highest / 2-highest are by track pT
      - if fewer than 2 associated tracks exist, missing values are padded with 0
    """
    try:
        pt_obj = float(obj["pT"])
        eta_obj = float(obj["eta"])
        phi_obj = float(obj["phi"])
        e_obj = float(obj["e"]) if "e" in obj else 0.0
    except Exception:
        return None

    if not np.isfinite(pt_obj) or pt_obj <= 0:
        return None

    if tracks.empty:
        return [
            pt_obj,
            eta_obj,
            phi_obj,
            e_obj,
            0.0,  # sum_pt
            0.0,  # n_tracks
            0.0,  # pt1/ptobj
            0.0,  # pt2/ptobj
            0.0,  # dr1/ptobj
            0.0,  # dr2/ptobj
        ]

    tracks = tracks[tracks["pT"] > trk_pt_min].copy()
    if tracks.empty:
        return [
            pt_obj,
            eta_obj,
            phi_obj,
            e_obj,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    tracks["dR"] = tracks.apply(
        lambda row: delta_r(eta_obj, phi_obj, float(row["eta"]), float(row["phi"])),
        axis=1,
    )

    assoc = tracks[tracks["dR"] < iso_dr].copy()
    if assoc.empty:
        return [
            pt_obj,
            eta_obj,
            phi_obj,
            e_obj,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    assoc = assoc.sort_values("pT", ascending=False).reset_index(drop=True)

    sum_pt = float(assoc["pT"].sum())
    n_tracks = float(len(assoc))

    pt1 = float(assoc.iloc[0]["pT"]) if len(assoc) >= 1 else 0.0
    pt2 = float(assoc.iloc[1]["pT"]) if len(assoc) >= 2 else 0.0

    dr1 = float(assoc.iloc[0]["dR"]) if len(assoc) >= 1 else 0.0
    dr2 = float(assoc.iloc[1]["dR"]) if len(assoc) >= 2 else 0.0

    pt1_over_ptobj = pt1 / pt_obj
    pt2_over_ptobj = pt2 / pt_obj
    dr1_over_ptobj = dr1 / pt_obj
    dr2_over_ptobj = dr2 / pt_obj

    return [
        pt_obj,
        eta_obj,
        phi_obj,
        e_obj,
        sum_pt,
        n_tracks,
        pt1_over_ptobj,
        pt2_over_ptobj,
        dr1_over_ptobj,
        dr2_over_ptobj,
    ]


# -----------------------------
# load dataset (returns groups by event_id)
# -----------------------------

def build_dataset(data_dir, iso_dr, trk_pt_min):
    photon_files = sorted(glob.glob(os.path.join(data_dir, "photons_*.csv")))

    X = []
    y = []
    groups = []

    for ph_file in photon_files:
        event_id = os.path.basename(ph_file).split("_")[-1].replace(".csv", "")

        jet_file = os.path.join(data_dir, f"jets_{event_id}.csv")
        trk_file = os.path.join(data_dir, f"tracks_{event_id}.csv")

        if not (os.path.exists(jet_file) and os.path.exists(trk_file)):
            continue

        photons = read_noheader(ph_file, "photons")
        jets = read_noheader(jet_file, "jets")
        tracks = read_noheader(trk_file, "tracks")

        if photons.empty and jets.empty:
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
            for c in ("pT", "eta", "phi"):
                if c in tracks.columns:
                    tracks[c] = pd.to_numeric(tracks[c], errors="coerce")
            tracks = tracks.dropna(subset=[c for c in ("pT", "eta", "phi") if c in tracks.columns])

        # photons = 1
        for _, ph in photons.iterrows():
            feats = engineer_features(ph, tracks, iso_dr=iso_dr, trk_pt_min=trk_pt_min)
            if feats is None:
                continue
            X.append(feats)
            y.append(1)
            groups.append(event_id)

        # jets = 0
        for _, jet in jets.iterrows():
            feats = engineer_features(jet, tracks, iso_dr=iso_dr, trk_pt_min=trk_pt_min)
            if feats is None:
                continue
            X.append(feats)
            y.append(0)
            groups.append(event_id)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups),
    )


# -----------------------------
# metrics
# -----------------------------

def fake_rate_at_target_tpr(y_true, y_score, target_tpr=0.99):
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


def partial_auc_maxfpr(y_true, y_score, max_fpr=0.05):
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def roc_points_df(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return pd.DataFrame(
        {
            "threshold": thr.astype(float),
            "tpr": tpr.astype(float),
            "fpr": fpr.astype(float),
        }
    )


# -----------------------------
# callbacks
# -----------------------------

class ValMetricsCallback(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, max_fpr=0.05, target_tpr=0.99):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.max_fpr = float(max_fpr)
        self.target_tpr = float(target_tpr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_hat = self.model.predict(self.x_val, verbose=0).ravel()

        val_pauc = partial_auc_maxfpr(self.y_val, y_hat, max_fpr=self.max_fpr)
        val_fpr99, _, _ = fake_rate_at_target_tpr(self.y_val, y_hat, target_tpr=self.target_tpr)

        logs["val_pauc05"] = val_pauc
        logs["val_fake99"] = float(val_fpr99)

        print(f" - val_pauc05: {val_pauc:.6f} - val_fake99: {val_fpr99:.6f}", end="")


# -----------------------------
# neural network
# -----------------------------

def build_model(n_features, learning_rate=1e-3, l2=1e-4, dropout=0.20):
    reg = keras.regularizers.l2(l2)

    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
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
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


# -----------------------------
# main
# -----------------------------

def main():
    ISO_DR = 0.20
    TRK_PT_MIN = 0.75
    TARGET_TPR = 0.99
    MAX_FPR_PAUC = 0.05

    LR = 1e-3
    L2 = 1e-4
    DROPOUT = 0.20
    EPOCHS = 100
    BATCH_SIZE = 256
    PATIENCE = 12

    data_dir = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    X, y, groups = build_dataset(data_dir, iso_dr=ISO_DR, trk_pt_min=TRK_PT_MIN)

    print("Dataset shape:", X.shape)
    print("positives (photons):", int(y.sum()), "negatives (jets):", int((1 - y).sum()))
    print("feature names:", get_feature_names())

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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = build_model(
        n_features=X_train.shape[1],
        learning_rate=LR,
        l2=L2,
        dropout=DROPOUT,
    )

    val_cb = ValMetricsCallback(
        x_val=X_val,
        y_val=y_val,
        max_fpr=MAX_FPR_PAUC,
        target_tpr=TARGET_TPR,
    )

    es = keras.callbacks.EarlyStopping(
        monitor="val_fake99",
        mode="min",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[val_cb, es],
    )

    y_val_pred = model.predict(X_val, verbose=0).ravel()
    y_test_pred = model.predict(X_test, verbose=0).ravel()

    val_auc = float(roc_auc_score(y_val, y_val_pred))
    test_auc = float(roc_auc_score(y_test, y_test_pred))

    val_fpr99, val_thr99, _ = fake_rate_at_target_tpr(y_val, y_val_pred, target_tpr=TARGET_TPR)
    test_fpr99, _, _ = fake_rate_at_target_tpr(y_test, y_test_pred, target_tpr=TARGET_TPR)

    roc_val = roc_points_df(y_val, y_val_pred)
    roc_test = roc_points_df(y_test, y_test_pred)

    roc_val.to_csv(os.path.join(out_dir, "roc_val.csv"), index=False)
    roc_test.to_csv(os.path.join(out_dir, "roc_test.csv"), index=False)

    pd.DataFrame(X, columns=get_feature_names()).to_csv(
        os.path.join(out_dir, "engineered_dataset.csv"),
        index=False,
    )

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("This run uses a neural network.\n")
        f.write("Feature set aligned to weekly task:\n")
        for name in get_feature_names():
            f.write(f"{name}\n")
        f.write(f"\nVal AUC: {val_auc}\n")
        f.write(f"Test AUC: {test_auc}\n")
        f.write(f"Val fake@99: {val_fpr99}\n")
        f.write(f"Test fake@99: {test_fpr99}\n")
        f.write(f"Val threshold at 99% TPR: {val_thr99}\n")

    print("\n--- results ---")
    print("This is a neural network model.")
    print("Features used:")
    for name in get_feature_names():
        print(" ", name)

    print(f"\nVal AUC: {val_auc:.6f}")
    print(f"Test AUC: {test_auc:.6f}")
    print(f"Val fake@99: {val_fpr99:.6f}")
    print(f"Test fake@99: {test_fpr99:.6f}")
    print(f"Val threshold at 99% TPR: {val_thr99:.6f}")


if __name__ == "__main__":
    main()