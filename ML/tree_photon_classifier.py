#!/usr/bin/env python3
# imports
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier


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
        keep = ["pT", "eta", "phi"][: len(df.columns)]
    else:
        raise ValueError(f"unknown kind: {kind}")

    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if kind == "photons" and "conversionType" in df.columns:
        df["conversionType"] = pd.to_numeric(df["conversionType"], errors="coerce")

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
    ]


# -----------------------------
# ring features
# -----------------------------

def _ring_stats(assoc: pd.DataFrame, ring_edges):
    out = []
    for lo, hi in ring_edges:
        sub = assoc[(assoc["dR"] >= lo) & (assoc["dR"] < hi)]
        out.append(float(len(sub)))
        out.append(float(sub["pT"].sum()) if len(sub) else 0.0)
    return out


# -----------------------------
# feature engineering
# -----------------------------

def engineer_features(obj, tracks, iso_dr=0.30, trk_pt_min=1.0):
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
    ]


# -----------------------------
# load dataset
# -----------------------------

def build_dataset(data_dir, iso_dr, trk_pt_min, include_converted=False):
    photon_files = sorted(
        glob.glob(os.path.join(data_dir, "photons_*.csv")) +
        glob.glob(os.path.join(data_dir, "photons_*.csv.gz"))
    )

    X = []
    y = []
    groups = []

    for ph_file in photon_files:
        base = os.path.basename(ph_file)
        event_id = base.split("_")[-1].replace(".csv.gz", "").replace(".csv", "")

        if ph_file.endswith(".csv.gz"):
            jet_file = os.path.join(data_dir, f"jets_{event_id}.csv.gz")
            trk_file = os.path.join(data_dir, f"tracks_{event_id}.csv.gz")
        else:
            jet_file = os.path.join(data_dir, f"jets_{event_id}.csv")
            trk_file = os.path.join(data_dir, f"tracks_{event_id}.csv")

        if not (os.path.exists(jet_file) and os.path.exists(trk_file)):
            continue

        photons = read_noheader(ph_file, "photons")
        jets = read_noheader(jet_file, "jets")
        tracks = read_noheader(trk_file, "tracks")

        if not include_converted and not photons.empty and "conversionType" in photons.columns:
            photons = photons[photons["conversionType"] == 0].reset_index(drop=True)

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

        for _, ph in photons.iterrows():
            feats = engineer_features(ph, tracks, iso_dr=iso_dr, trk_pt_min=trk_pt_min)
            if feats is None:
                continue
            X.append(feats)
            y.append(1)
            groups.append(event_id)

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
# feature subset utility
# -----------------------------

def select_feature_subset(X, feature_names, drop_features):
    drop_features = set(drop_features)
    keep_idx = [i for i, name in enumerate(feature_names) if name not in drop_features]
    kept_names = [feature_names[i] for i in keep_idx]
    X_sel = X[:, keep_idx]
    return X_sel, kept_names


# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-converted",
        action="store_true",
        default=False,
        help="include converted photons (default: exclude them)",
    )
    args = parser.parse_args()

    ISO_DR = 0.20
    TRK_PT_MIN = 0.75
    TARGET_TPRS = [0.95, 0.97, 0.99]

    data_dir = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\10k_ev"

    EXPERIMENT = "tree_hgb_baseline"

    # best fair comparison to your best NN:
    # drop the 4 ratio features that hurt the NN
    DROP_FEATURES = [
        "iso_ratio",
        "core_iso_ratio",
        "maxpt_over_objpt",
        "core_frac",
    ]

    out_dir = os.path.join(os.path.dirname(__file__), f"results_{EXPERIMENT}")
    os.makedirs(out_dir, exist_ok=True)

    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    conv_tag = "inclconv" if args.include_converted else "exclconv"
    cache_name = (
        f"dataset_iso{str(ISO_DR).replace('.', 'p')}_"
        f"pt{str(TRK_PT_MIN).replace('.', 'p')}_"
        f"{conv_tag}.npz"
    )
    cache_path = os.path.join(cache_dir, cache_name)

    feature_names = get_feature_names()

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        groups = data["groups"]
        cached_feature_names = list(data["feature_names"])
        print(f"loaded cached dataset from {cache_path}")

        if cached_feature_names != feature_names:
            raise RuntimeError(
                "cached feature names do not match current get_feature_names(). "
                "delete the cache file and rebuild."
            )
    else:
        X, y, groups = build_dataset(
            data_dir,
            iso_dr=ISO_DR,
            trk_pt_min=TRK_PT_MIN,
            include_converted=args.include_converted,
        )
        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            groups=groups,
            feature_names=np.array(feature_names, dtype=object),
        )
        print(f"saved cached dataset to {cache_path}")

    if DROP_FEATURES:
        X, feature_names = select_feature_subset(X, feature_names, DROP_FEATURES)
        print("dropped features:", DROP_FEATURES)

    print("Dataset shape:", X.shape)
    print("positives (photons):", int(y.sum()), "negatives (jets):", int((1 - y).sum()))
    print("feature names:", feature_names)
    print("converted photons:", "included" if args.include_converted else "excluded")

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

    # tree model
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=None,
        n_iter_no_change=20,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    pd.DataFrame({
        "label": y_val,
        "score": y_val_pred,
    }).to_csv(os.path.join(out_dir, "scores_val.csv"), index=False)

    pd.DataFrame({
        "label": y_test,
        "score": y_test_pred,
    }).to_csv(os.path.join(out_dir, "scores_test.csv"), index=False)

    val_auc = float(roc_auc_score(y_val, y_val_pred))
    test_auc = float(roc_auc_score(y_test, y_test_pred))

    roc_val = roc_points_df(y_val, y_val_pred)
    roc_test = roc_points_df(y_test, y_test_pred)

    roc_val.to_csv(os.path.join(out_dir, "roc_val.csv"), index=False)
    roc_test.to_csv(os.path.join(out_dir, "roc_test.csv"), index=False)

    dataset_df = pd.DataFrame(X, columns=feature_names)
    dataset_df["label"] = y
    dataset_df.to_csv(os.path.join(out_dir, "engineered_dataset.csv"), index=False)

    summary_row = pd.DataFrame([{
        "experiment": EXPERIMENT,
        "model": "HistGradientBoostingClassifier",
        "n_features": len(feature_names),
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_fake95": fake_rate_at_target_tpr(y_val, y_val_pred, 0.95)[0],
        "test_fake95": fake_rate_at_target_tpr(y_test, y_test_pred, 0.95)[0],
        "val_fake97": fake_rate_at_target_tpr(y_val, y_val_pred, 0.97)[0],
        "test_fake97": fake_rate_at_target_tpr(y_test, y_test_pred, 0.97)[0],
        "val_fake99": fake_rate_at_target_tpr(y_val, y_val_pred, 0.99)[0],
        "test_fake99": fake_rate_at_target_tpr(y_test, y_test_pred, 0.99)[0],
        "dropped_features": ",".join(DROP_FEATURES) if DROP_FEATURES else "",
    }])

    summary_csv = os.path.join(os.path.dirname(__file__), "ablation_summary.csv")

    if os.path.exists(summary_csv):
        old = pd.read_csv(summary_csv)
        old = old[old["experiment"] != EXPERIMENT]
        summary_row = pd.concat([old, summary_row], ignore_index=True)

    summary_row.to_csv(summary_csv, index=False)

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("This run uses a tree model.\n")
        f.write(f"Experiment: {EXPERIMENT}\n")
        f.write("Model: HistGradientBoostingClassifier\n")
        f.write(f"Cache file: {cache_path}\n")
        f.write(f"Converted photons: {'included' if args.include_converted else 'excluded'}\n")
        if DROP_FEATURES:
            f.write("Dropped features:\n")
            for name in DROP_FEATURES:
                f.write(f"{name}\n")

        f.write("Feature set:\n")
        for name in feature_names:
            f.write(f"{name}\n")

        f.write(f"\nVal AUC: {val_auc}\n")
        f.write(f"Test AUC: {test_auc}\n")

        for tpr in TARGET_TPRS:
            val_fpr, val_thr, _ = fake_rate_at_target_tpr(y_val, y_val_pred, target_tpr=tpr)
            test_fpr, _, _ = fake_rate_at_target_tpr(y_test, y_test_pred, target_tpr=tpr)
            f.write(f"\nVal fake@{int(tpr * 100)}: {val_fpr}\n")
            f.write(f"Test fake@{int(tpr * 100)}: {test_fpr}\n")
            f.write(f"Val threshold at {int(tpr * 100)}% TPR: {val_thr}\n")

    print("\n--- results ---")
    print("This is a tree model.")
    print("Features used:")
    for name in feature_names:
        print(" ", name)

    print(f"\nVal AUC: {val_auc:.6f}")
    print(f"Test AUC: {test_auc:.6f}")

    for tpr in TARGET_TPRS:
        val_fpr, val_thr, _ = fake_rate_at_target_tpr(y_val, y_val_pred, target_tpr=tpr)
        test_fpr, _, _ = fake_rate_at_target_tpr(y_test, y_test_pred, target_tpr=tpr)

        print(f"\nTPR = {tpr:.2f}")
        print(f"Val fake rate: {val_fpr:.6f}")
        print(f"Test fake rate: {test_fpr:.6f}")
        print(f"Val threshold: {val_thr:.6f}")


if __name__ == "__main__":
    main()