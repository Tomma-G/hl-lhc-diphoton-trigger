#!/usr/bin/env python3
"""
HL-LHC diphoton toy analysis.

This script performs a simplified diphoton analysis using per-event CSV inputs
for photons, jets, and tracks. It implements track-based isolation, jet–photon
overlap removal, and basic performance metrics suitable for parameter scans.

Crucially (for realistic trigger/ML-style evaluation):
  - isolation and track-count features are computed using the full track collection
    in the event (no photon-conditioned track filtering).
  - any "tracks near photons" filtering is computed for diagnostics only and is
    never used to build features or performance curves.

Per-event inputs (no headers):
  photons_<id>.csv : pT, eta, phi, e, conversionType
  jets_<id>.csv    : pT, eta, phi, e
  tracks_<id>.csv  : pT, eta, phi, eTot, z0, d0

Main outputs:
  - m_gg invariant mass (CSV + histogram, 2-photon events)
  - jet–photon overlap removal diagnostics
  - per-object track counts
  - track isolation values
  - acceptance vs fake-rate ROC curve (PNG + points CSV)
  - AUC (full or partial, configurable)
  - track_reduction.csv (global track preselection diagnostic only)

Scan support:
  --fast-scan  : write only iso_values.csv, acceptance_vs_fake_rate_points.csv, and track_reduction.csv
  --quiet      : suppress informational output (for batch scans)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import hashlib
import os
import re
import sys
import tarfile
from typing import Dict, List, Tuple

# --- Windows console robustness: force UTF-8 output ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")

CFG: Dict[str, float] = {
    "dr_overlap_default": 0.20,
    "dr_track_default": 0.10,
    "iso_dr_default": 0.20,
    "dr_track_keep_default": 0.03,  # diagnostic only
}

QUIET = False


def _safe_print(prefix: str, msg: str, *, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    try:
        print(f"{prefix} {msg}", file=stream)
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(f"{prefix} {safe}", file=stream)

def write_roc_csv(out_dir, fpr, tpr, fname="roc.csv"):
    import os, csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr"])
        for x, y in zip(fpr, tpr):
            w.writerow([float(x), float(y)])
    return path


def write_ntracks_csv(out_dir, n_iso, n_total, fname="ntracks.csv"):
    import os, csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ntracks_iso", "ntracks_total", "frac_kept"])
        for a, b in zip(n_iso, n_total):
            a = int(a)
            b = int(b)
            frac = (a / b) if b > 0 else 0.0
            w.writerow([a, b, frac])
    return path

def info(msg: str) -> None:
    if QUIET:
        return
    _safe_print("[info]", msg)


def warn(msg: str) -> None:
    if QUIET:
        return
    _safe_print("[warn]", msg)


def err(msg: str) -> None:
    _safe_print("[error]", msg, stream=sys.stderr)


def find_event_ids(data_dir: str) -> List[int]:
    ids: List[int] = []
    for name in os.listdir(data_dir):
        m = PHOTON_RE.match(name)
        if m:
            ids.append(int(m.group(1)))
    ids.sort()
    return ids


def photons_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"photons_{event_id}.csv")


def jets_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"jets_{event_id}.csv")


def tracks_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"tracks_{event_id}.csv")


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    return float((dphi + np.pi) % (2.0 * np.pi) - np.pi)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return float(np.sqrt(deta * deta + dphi * dphi))


def pxyz_from_ptetaphi(pt: float, eta: float, phi: float) -> Tuple[float, float, float]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return float(px), float(py), float(pz)


def inv_mass_from_two_objects(obj1: Dict[str, float], obj2: Dict[str, float]) -> float:
    px1, py1, pz1 = pxyz_from_ptetaphi(obj1["pT"], obj1["eta"], obj1["phi"])
    px2, py2, pz2 = pxyz_from_ptetaphi(obj2["pT"], obj2["eta"], obj2["phi"])

    E = obj1["e"] + obj2["e"]
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = E * E - (px * px + py * py + pz * pz)
    return float(np.sqrt(max(m2, 0.0)))


def _dr_mask_to_object(
    trk_eta: np.ndarray,
    trk_phi: np.ndarray,
    obj_eta: float,
    obj_phi: float,
    dr_max: float,
) -> np.ndarray:
    if trk_eta.size == 0:
        return np.zeros(0, dtype=bool)
    deta = trk_eta - float(obj_eta)
    dphi = (trk_phi - float(obj_phi) + np.pi) % (2.0 * np.pi) - np.pi
    dr2 = deta * deta + dphi * dphi
    return dr2 < float(dr_max) * float(dr_max)


def count_tracks_near_object_fast(
    obj_eta: float,
    obj_phi: float,
    trk_eta: np.ndarray,
    trk_phi: np.ndarray,
    dr_max: float,
) -> int:
    return int(np.count_nonzero(_dr_mask_to_object(trk_eta, trk_phi, obj_eta, obj_phi, dr_max)))


def track_iso_scalar_sum_pt_fast(
    obj_eta: float,
    obj_phi: float,
    obj_pt: float,
    trk_eta: np.ndarray,
    trk_phi: np.ndarray,
    trk_pt: np.ndarray,
    dr_max: float,
) -> float:
    if obj_pt <= 0.0:
        return float("nan")
    if trk_eta.size == 0:
        return 0.0
    m = _dr_mask_to_object(trk_eta, trk_phi, obj_eta, obj_phi, dr_max)
    if m.size == 0:
        return 0.0
    return float(np.sum(trk_pt[m], dtype=float) / float(obj_pt))


def filter_tracks_close_to_photons(df_trk: pd.DataFrame, df_ph: pd.DataFrame, dr_keep: float) -> pd.DataFrame:
    """Return tracks within dr_keep of any photon (diagnostic only)."""
    if len(df_trk) == 0 or len(df_ph) == 0:
        return df_trk

    trk_eta = df_trk["eta"].to_numpy(dtype=float)
    trk_phi = df_trk["phi"].to_numpy(dtype=float)

    keep_mask = np.zeros(len(df_trk), dtype=bool)
    for _, pho in df_ph.iterrows():
        pho_eta = float(pho["eta"])
        pho_phi = float(pho["phi"])
        deta = trk_eta - pho_eta
        dphi = (trk_phi - pho_phi + np.pi) % (2.0 * np.pi) - np.pi
        dr2 = deta * deta + dphi * dphi
        keep_mask |= (dr2 < float(dr_keep) * float(dr_keep))

    return df_trk.loc[keep_mask].reset_index(drop=True)


def roc_from_iso(ph_iso: np.ndarray, jet_iso: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ROC curve for isolation cut I < c. Returns (fpr, tpr) arrays for plotting/integration."""
    ph = ph_iso[np.isfinite(ph_iso)].astype(float)
    jt = jet_iso[np.isfinite(jet_iso)].astype(float)
    if ph.size == 0 or jt.size == 0:
        return np.array([]), np.array([])

    vals = np.unique(np.concatenate([ph, jt]))
    vals.sort()
    if vals.size == 1:
        c = float(vals[0])
        tpr = float(np.mean(ph < c))
        fpr = float(np.mean(jt < c))
        return np.array([fpr], dtype=float), np.array([tpr], dtype=float)

    mids = (vals[:-1] + vals[1:]) * 0.5
    cuts_plot = np.concatenate(([vals[0] - 1.0], mids, [vals[-1] + 1.0]))

    tpr_plot = np.array([float(np.mean(ph < c)) for c in cuts_plot], dtype=float)
    fpr_plot = np.array([float(np.mean(jt < c)) for c in cuts_plot], dtype=float)

    # ensure monotonically increasing in fpr for stable integration/plotting
    order = np.argsort(fpr_plot)
    return fpr_plot[order], tpr_plot[order]


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray, *, max_fpr: float = 1.0) -> float:
    """
    Compute (partial) AUC = ∫ TPR d(FPR) over FPR in [0, max_fpr].
    Uses trapezoidal integration after clipping to the requested FPR range.
    """
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")

    f = np.asarray(fpr, dtype=float)
    t = np.asarray(tpr, dtype=float)

    # sort by FPR
    order = np.argsort(f)
    f = f[order]
    t = t[order]

    # drop any NaNs/infs (shouldn't be present but keep it robust)
    m = np.isfinite(f) & np.isfinite(t)
    f = f[m]
    t = t[m]
    if f.size == 0:
        return float("nan")

    max_fpr = float(max_fpr)
    if max_fpr <= 0.0:
        return 0.0

    # clip to [0, max_fpr]
    # ensure start at f=0
    if f[0] > 0.0:
        f = np.concatenate(([0.0], f))
        t = np.concatenate(([0.0], t))

    # truncate/clip at max_fpr, and add an interpolated endpoint if needed
    if f[-1] < max_fpr:
        # extend flat to max_fpr with last tpr
        f = np.concatenate((f, [max_fpr]))
        t = np.concatenate((t, [t[-1]]))
    else:
        # cut everything above max_fpr, then append interpolated point at max_fpr
        above = f > max_fpr
        if np.any(above):
            first_above = int(np.argmax(above))
            f0, t0 = f[first_above - 1], t[first_above - 1]
            f1, t1 = f[first_above], t[first_above]
            if f1 == f0:
                t_at = t0
            else:
                frac = (max_fpr - f0) / (f1 - f0)
                t_at = t0 + frac * (t1 - t0)
            f = np.concatenate((f[:first_above], [max_fpr]))
            t = np.concatenate((t[:first_above], [t_at]))

    return float(np.trapz(t, f))


def cut_at_fixed_tpr(ph_iso: np.ndarray, jt_iso: np.ndarray, target_tpr: float):
    ph = ph_iso[np.isfinite(ph_iso)]
    jt = jt_iso[np.isfinite(jt_iso)]
    if ph.size == 0 or jt.size == 0:
        return np.nan, np.nan, np.nan

    cuts = np.sort(ph) + 1e-12
    tpr = np.array([(ph < c).mean() for c in cuts])

    idx = np.searchsorted(tpr, target_tpr, side="left")
    idx = min(idx, len(cuts) - 1)

    cut = cuts[idx]
    return cut, tpr[idx], (jt < cut).mean()


def best_threshold_separation(photon_counts: List[int], jet_counts: List[int]) -> Tuple[int, float, float, float]:
    """Optimal track-count cut n_tracks ≤ t."""
    if len(photon_counts) == 0 or len(jet_counts) == 0:
        return 0, float("nan"), float("nan"), float("nan")

    p = np.array(photon_counts, dtype=int)
    j = np.array(jet_counts, dtype=int)
    max_c = int(max(p.max(), j.max()))

    best = (-1.0, 0, 0.0, 1.0)
    for t in range(0, max_c + 1):
        photon_eff = float(np.mean(p <= t))
        jet_misid = float(np.mean(j <= t))
        accuracy = float((np.sum(p <= t) + np.sum(j > t)) / (len(p) + len(j)))
        if accuracy > best[0]:
            best = (accuracy, t, photon_eff, jet_misid)

    accuracy, t, photon_eff, jet_misid = best
    return int(t), float(photon_eff), float(jet_misid), float(accuracy)


def read_csv_maybe_gzip(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV; auto-detect gzip via magic bytes."""
    with open(path, "rb") as f:
        head = f.read(2)
    is_gz = (len(head) == 2 and head[0] == 0x1F and head[1] == 0x8B)
    if is_gz:
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def looks_like_text_file(path: str, *, nbytes: int = 4096) -> bool:
    """
    Heuristic: return False if the file looks binary/corrupt (NUL bytes, tar header markers, etc).
    Prevents pandas producing garbage rows that later crash float() conversions.
    """
    try:
        with open(path, "rb") as f:
            buf = f.read(nbytes)
    except OSError:
        return False

    if not buf:
        return False
    if b"\x00" in buf:
        return False
    if b"ustar" in buf:
        return False
    return True


def coerce_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Force selected columns to numeric and drop rows that fail."""
    if df.empty:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=cols).reset_index(drop=True)
    return out


def prepare_data_dir(path: str) -> str:
    """
    If path is a directory: return it unchanged.
    If path ends with .tar.gz: extract once to a cached folder and return that folder.
    """
    abs_path = os.path.abspath(path)

    if os.path.isdir(abs_path):
        return abs_path

    if not abs_path.lower().endswith(".tar.gz"):
        return abs_path

    h = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:10]
    extract_dir = os.path.join(os.path.dirname(abs_path), f"_extracted_{h}")

    if os.path.isdir(extract_dir):
        info(f"using cached extracted dataset: {extract_dir}")
        return extract_dir

    info(f"extracting {abs_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(abs_path, "r:gz") as tar:
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)

    info("extraction complete")
    return extract_dir


def main() -> None:
    global QUIET

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/1k_ev",
    )
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--out-png", default="mgg_hist.png")
    parser.add_argument("--out-csv", default="mgg_values.csv")
    parser.add_argument("--out-dir", default=None)

    parser.add_argument(
        "--include-converted",
        action="store_true",
        default=False,
        help="include converted photons (default: exclude them)",
    )

    parser.add_argument("--dr-overlap", type=float, default=float(CFG["dr_overlap_default"]))
    parser.add_argument("--dr-track", type=float, default=float(CFG["dr_track_default"]))
    parser.add_argument("--dr-track-keep", type=float, default=float(CFG["dr_track_keep_default"]))

    parser.add_argument(
        "--trk-pt-min",
        type=float,
        default=0.0,
        help="only use tracks with pT > trk_pt_min for n_tracks + iso features",
    )


    parser.add_argument("--tracks-photons-png", default="tracks_near_photons.png")
    parser.add_argument("--tracks-jets-png", default="tracks_near_jets.png")
    parser.add_argument("--tracks-out-csv", default="track_counts.csv")

    parser.add_argument("--mindr-before-png", default="min_dr_jet_photon_before.png")
    parser.add_argument("--mindr-after-png", default="min_dr_jet_photon_after.png")
    parser.add_argument("--tracks-photons-split-png", default="tracks_near_photons_split_conversion.png")

    parser.add_argument("--iso-dr", type=float, default=float(CFG["iso_dr_default"]))
    parser.add_argument("--iso-out-csv", default="iso_values.csv")
    parser.add_argument("--iso-photons-png", default="iso_photons.png")
    parser.add_argument("--iso-jets-png", default="iso_jets.png")
    parser.add_argument("--roc-png", default="acceptance_vs_fake_rate.png")

    parser.add_argument("--roc-out-csv", default="acceptance_vs_fake_rate_points.csv")
    parser.add_argument(
        "--auc-max-fpr",
        type=float,
        default=1.0,
        help="compute (partial) AUC integrating fpr in [0, auc_max_fpr]. Use 1.0 for full AUC.",
    )

    parser.add_argument(
        "--fast-scan",
        action="store_true",
        default=False,
        help="scan mode: only writes iso_values.csv + acceptance_vs_fake_rate_points.csv + track_reduction.csv "
        "(skips plots and other outputs)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="suppress most stdout (useful when called repeatedly by scan script)",
    )

    args = parser.parse_args()
    QUIET = bool(args.quiet)

    exclude_converted = not args.include_converted
    args.data_dir = prepare_data_dir(args.data_dir)

    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.abspath(os.path.join(args.data_dir, "..", "analysis_outputs"))
    )
    os.makedirs(out_dir, exist_ok=True)
    info(f"writing outputs to: {out_dir}")

    out_csv_path = os.path.join(out_dir, args.out_csv)
    out_png_path = os.path.join(out_dir, args.out_png)

    tracks_out_csv_path = os.path.join(out_dir, args.tracks_out_csv)
    tracks_photons_png_path = os.path.join(out_dir, args.tracks_photons_png)
    tracks_jets_png_path = os.path.join(out_dir, args.tracks_jets_png)

    mindr_before_png_path = os.path.join(out_dir, args.mindr_before_png)
    mindr_after_png_path = os.path.join(out_dir, args.mindr_after_png)
    tracks_photons_split_png_path = os.path.join(out_dir, args.tracks_photons_split_png)

    iso_out_csv_path = os.path.join(out_dir, args.iso_out_csv)
    iso_photons_png_path = os.path.join(out_dir, args.iso_photons_png)
    iso_jets_png_path = os.path.join(out_dir, args.iso_jets_png)
    roc_png_path = os.path.join(out_dir, args.roc_png)

    roc_out_csv_path = os.path.join(out_dir, args.roc_out_csv)
    auc_max_fpr = float(args.auc_max_fpr)

    if not os.path.isdir(args.data_dir):
        err(f"data-dir does not exist: {args.data_dir}")
        sys.exit(1)

    event_ids = find_event_ids(args.data_dir)
    if not event_ids:
        err(f"no photons_<id>.csv found in {args.data_dir}")
        sys.exit(1)

    chosen = event_ids[: args.n_events]
    info(f"found {len(event_ids)} photon files; processing {len(chosen)} events")
    info(f"photon conversion handling: {'excluding' if exclude_converted else 'including'} converted photons")
    info(f"track preselection (diagnostic only): tracks within dR(track, any photon) < {args.dr_track_keep}")
    info("features (iso + track counts) use ALL tracks (no photon-conditioned track filtering)")
    info(f"track pT cut: pT > {args.trk_pt_min:g} GeV (applies to features + diagnostics)")
    
    if args.fast_scan:
        info("fast-scan: ON (only iso_values.csv + acceptance_vs_fake_rate_points.csv + track_reduction.csv will be written)")

    mgg_rows: List[Dict[str, float]] = []

    n_read = 0
    n_two_photon = 0
    n_bad_ph = 0

    n_jets_before = 0
    n_jets_after = 0
    n_bad_jets = 0

    min_dr_before_per_event: List[float] = []
    min_dr_after_per_event: List[float] = []

    photon_track_counts: List[int] = []
    jet_track_counts: List[int] = []
    track_rows: List[Dict[str, float]] = []

    iso_rows: List[Dict[str, float]] = []
    photon_iso_vals: List[float] = []
    jet_iso_vals: List[float] = []
    photon_iso_ntracks: List[int] = []
    jet_iso_ntracks: List[int] = []
    photon_total_ntracks: List[int] = []
    jet_total_ntracks: List[int] = []



    n_trk_all_total = 0
    n_trk_phclose_total = 0  # diagnostic only

    example_printed = False

    for ev in chosen:
        ph_path = photons_path(args.data_dir, ev)
        if not os.path.exists(ph_path) or os.path.getsize(ph_path) == 0:
            n_bad_ph += 1
            continue

        try:
            df_ph = pd.read_csv(
                ph_path,
                header=None,
                names=["pT", "eta", "phi", "e", "conversionType"],
            )[["pT", "eta", "phi", "e", "conversionType"]]
            df_ph = coerce_numeric_df(df_ph, ["pT", "eta", "phi", "e"])
            n_read += 1
        except Exception as e:
            warn(f"failed to read {ph_path}: {e}")
            n_bad_ph += 1
            continue

        if exclude_converted:
            conv_int = pd.to_numeric(df_ph["conversionType"], errors="coerce").fillna(-999).astype(int)
            df_ph = df_ph[conv_int == 0].reset_index(drop=True)

        # ---- jets (robust) ----
        jet_p = jets_path(args.data_dir, ev)
        if os.path.exists(jet_p) and os.path.getsize(jet_p) > 0:
            if not looks_like_text_file(jet_p):
                warn(f"jets file looks non-text/corrupt (skipping): {jet_p}")
                n_bad_jets += 1
                df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])
            else:
                try:
                    df_j = read_csv_maybe_gzip(
                        jet_p,
                        header=None,
                        names=["pT", "eta", "phi", "e"],
                        engine="python",
                        on_bad_lines="skip",
                    )
                    df_j = df_j[["pT", "eta", "phi", "e"]]
                    df_j = coerce_numeric_df(df_j, ["pT", "eta", "phi", "e"])
                except Exception as e:
                    warn(f"failed to read {jet_p}: {e}")
                    n_bad_jets += 1
                    df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])
        else:
            df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])

        # ---- tracks ----
        trk_p = tracks_path(args.data_dir, ev)
        if os.path.exists(trk_p) and os.path.getsize(trk_p) > 0:
            try:
                df_trk_all = pd.read_csv(
                    trk_p,
                    header=None,
                    names=["pT", "eta", "phi", "eTot", "z0", "d0"],
                )
                df_trk_all = coerce_numeric_df(df_trk_all, ["pT", "eta", "phi"])
            except Exception as e:
                warn(f"failed to read {trk_p}: {e}")
                df_trk_all = pd.DataFrame(columns=["pT", "eta", "phi", "eTot", "z0", "d0"])
        else:
            df_trk_all = pd.DataFrame(columns=["pT", "eta", "phi", "eTot", "z0", "d0"])

        # ---- optional track pT cut (applies to BOTH features and diagnostics) ----
        if args.trk_pt_min > 0.0 and len(df_trk_all) > 0:
            df_trk_all = df_trk_all[df_trk_all["pT"] > float(args.trk_pt_min)].reset_index(drop=True)

        # diagnostic: how many tracks are near photons (do NOT use downstream)
        n_trk_all_total += len(df_trk_all)
        df_trk_phclose = filter_tracks_close_to_photons(df_trk_all, df_ph, float(args.dr_track_keep))
        n_trk_phclose_total += len(df_trk_phclose)

        # feature inputs: full track collection
        trk_all_eta = df_trk_all["eta"].to_numpy(dtype=float) if len(df_trk_all) else np.zeros(0, dtype=float)
        trk_all_phi = df_trk_all["phi"].to_numpy(dtype=float) if len(df_trk_all) else np.zeros(0, dtype=float)
        trk_all_pt = df_trk_all["pT"].to_numpy(dtype=float) if len(df_trk_all) else np.zeros(0, dtype=float)

        n_jets_before += len(df_j)

        if (not example_printed) and (len(df_ph) > 0) and (len(df_j) > 0) and (not args.fast_scan):
            info(f"example event {ev}")
            info(f"photon row: {df_ph.iloc[0].to_dict()}")
            info(f"jet row:    {df_j.iloc[0].to_dict()}")
            example_printed = True

        # ---- jet–photon overlap removal ----
        kept: List[pd.Series] = []
        min_dr_before = np.inf

        for _, jet in df_j.iterrows():
            overlap = False
            for _, pho in df_ph.iterrows():
                dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                if not args.fast_scan:
                    min_dr_before = min(min_dr_before, dr)
                if dr < args.dr_overlap:
                    overlap = True
            if not overlap:
                kept.append(jet)

        if (not args.fast_scan) and np.isfinite(min_dr_before):
            min_dr_before_per_event.append(float(min_dr_before))

        if (not args.fast_scan) and len(kept) > 0 and len(df_ph) > 0:
            min_dr_after = np.inf
            for jet in kept:
                for _, pho in df_ph.iterrows():
                    dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                    min_dr_after = min(min_dr_after, dr)
            if np.isfinite(min_dr_after):
                min_dr_after_per_event.append(float(min_dr_after))

        n_jets_after += len(kept)

        # ---- photons: features use ALL tracks (realistic) ----
        for i, pho in df_ph.iterrows():
            pho_eta = float(pho["eta"])
            pho_phi = float(pho["phi"])
            pho_pt = float(pho["pT"])

            ntrk = count_tracks_near_object_fast(pho_eta, pho_phi, trk_all_eta, trk_all_phi, args.dr_track)
            iso = track_iso_scalar_sum_pt_fast(pho_eta, pho_phi, pho_pt, trk_all_eta, trk_all_phi, trk_all_pt, args.iso_dr)

            ntrk_iso = count_tracks_near_object_fast(pho_eta, pho_phi, trk_all_eta, trk_all_phi, args.iso_dr)
            photon_iso_ntracks.append(ntrk_iso)

            photon_total_ntracks.append(int(len(trk_all_pt)))

            conv_val = pd.to_numeric(pho["conversionType"], errors="coerce")
            conv_int = int(conv_val) if pd.notna(conv_val) else -999

            photon_iso_vals.append(iso)
            iso_rows.append(
                {
                    "event_id": ev,
                    "object": "photon",
                    "index": int(i),
                    "iso_dr": float(args.iso_dr),
                    "iso": float(iso),
                    "conversionType": conv_int,
                    "obj_pt": pho_pt,
                }
            )

            if not args.fast_scan:
                photon_track_counts.append(ntrk)
                track_rows.append(
                    {
                        "event_id": ev,
                        "object": "photon",
                        "index": int(i),
                        "n_tracks": int(ntrk),
                        "conversionType": conv_int,
                    }
                )

        # ---- jets: features use ALL tracks (realistic) ----
        for i, jet in enumerate(kept):
            jet_eta = float(jet["eta"])
            jet_phi = float(jet["phi"])
            jet_pt = float(jet["pT"])

            ntrk = count_tracks_near_object_fast(jet_eta, jet_phi, trk_all_eta, trk_all_phi, args.dr_track)
            iso = track_iso_scalar_sum_pt_fast(jet_eta, jet_phi, jet_pt, trk_all_eta, trk_all_phi, trk_all_pt, args.iso_dr)

            ntrk_iso = count_tracks_near_object_fast(jet_eta, jet_phi, trk_all_eta, trk_all_phi, args.iso_dr)
            jet_iso_ntracks.append(ntrk_iso)

            jet_total_ntracks.append(int(len(trk_all_pt)))

            jet_iso_vals.append(iso)
            iso_rows.append(
                {
                    "event_id": ev,
                    "object": "jet",
                    "index": int(i),
                    "iso_dr": float(args.iso_dr),
                    "iso": float(iso),
                    "conversionType": np.nan,
                    "obj_pt": jet_pt,
                }
            )

            if not args.fast_scan:
                jet_track_counts.append(ntrk)
                track_rows.append(
                    {
                        "event_id": ev,
                        "object": "jet",
                        "index": int(i),
                        "n_tracks": int(ntrk),
                        "conversionType": np.nan,
                    }
                )

        # mgg only if not fast-scan
        if args.fast_scan:
            continue

        if len(df_ph) != 2:
            continue

        n_two_photon += 1
        pho1 = {
            "pT": float(df_ph.iloc[0]["pT"]),
            "eta": float(df_ph.iloc[0]["eta"]),
            "phi": float(df_ph.iloc[0]["phi"]),
            "e": float(df_ph.iloc[0]["e"]),
        }
        pho2 = {
            "pT": float(df_ph.iloc[1]["pT"]),
            "eta": float(df_ph.iloc[1]["eta"]),
            "phi": float(df_ph.iloc[1]["phi"]),
            "e": float(df_ph.iloc[1]["e"]),
        }
        mgg_rows.append({"event_id": ev, "m_gg": inv_mass_from_two_objects(pho1, pho2)})

    # ---- track preselection diagnostic ----
    if n_trk_all_total > 0:
        frac = float(n_trk_phclose_total) / float(n_trk_all_total)
        info(f"track preselection kept {n_trk_phclose_total}/{n_trk_all_total} = {frac:.3f}")

        red_path = os.path.join(out_dir, "track_reduction.csv")
        pd.DataFrame(
            [
                {
                    "kept_tracks": int(n_trk_phclose_total),
                    "all_tracks": int(n_trk_all_total),
                    "kept_fraction": float(frac),
                    "dr_track_keep": float(args.dr_track_keep),
                }
            ]
        ).to_csv(red_path, index=False)
        info(f"wrote {red_path}")

    # Always write iso_values.csv
    pd.DataFrame(iso_rows).to_csv(iso_out_csv_path, index=False)
    info(f"wrote {iso_out_csv_path}")

    # Write Ntracks distributions used by iso cone (for justification plots)
    ph_nt_path = write_ntracks_csv(out_dir, photon_iso_ntracks, photon_total_ntracks, fname="ntracks_photons.csv")
    jt_nt_path = write_ntracks_csv(out_dir, jet_iso_ntracks, jet_total_ntracks, fname="ntracks_jets.csv")

    comb_iso = photon_iso_ntracks + jet_iso_ntracks
    comb_tot = photon_total_ntracks + jet_total_ntracks
    comb_path = write_ntracks_csv(out_dir, comb_iso, comb_tot, fname="ntracks.csv")
    info(f"wrote {comb_path}")

    # Always compute + write ROC points (needed for scans)
    ph_iso = np.array(photon_iso_vals, dtype=float)
    jt_iso = np.array(jet_iso_vals, dtype=float)
    fpr, tpr = roc_from_iso(ph_iso, jt_iso)
    auc = float("nan")
    if len(fpr) > 0:
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(roc_out_csv_path, index=False)
        info(f"wrote {roc_out_csv_path}")

        roc_std_path = write_roc_csv(out_dir, fpr, tpr, fname="roc.csv")
        info(f"wrote {roc_std_path}")


        auc = auc_from_roc(fpr, tpr, max_fpr=auc_max_fpr)
        if np.isfinite(auc):
            info(f"AUC (max_fpr={auc_max_fpr:g}) = {auc:.6f}")
        else:
            warn("AUC could not be computed (non-finite ROC arrays)")
    else:
        warn("could not compute iso ROC (need finite iso for both photons and jets)")

    TARGET_TPR = 0.99
    cut99, tpr99, fpr99 = cut_at_fixed_tpr(ph_iso, jt_iso, TARGET_TPR)
    if np.isfinite(cut99):
        info(
            f"FIXED TPR={TARGET_TPR:.2f}: cut={cut99:.6g}, "
            f"acc={tpr99:.3f}, fake={fpr99:.3f}, rej={1.0-fpr99:.3f}"
        )

    if args.fast_scan:
        return

    # ---- full-analysis mode ----
    if not mgg_rows:
        err("no valid 2-photon events found")
        sys.exit(1)

    out = pd.DataFrame(mgg_rows)
    info(f"read photon files: {n_read} (skipped bad/empty: {n_bad_ph})")
    info(f"2-photon events: {n_two_photon}")

    out.to_csv(out_csv_path, index=False)
    info(f"wrote {out_csv_path}")

    plt.figure()
    plt.hist(out["m_gg"].to_numpy(dtype=float), bins=args.bins)
    plt.xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    plt.ylabel("Entries")
    plt.title(r"Di-photon invariant mass (events with exactly 2 photons)")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    info(f"wrote {out_png_path}")

    if n_bad_jets:
        info(f"bad jets files: {n_bad_jets}")

    if len(min_dr_before_per_event) > 0:
        arr = np.array(min_dr_before_per_event, dtype=float)
        info(f"min dR(jet,photon) before: n={len(arr)} min/med/max={arr.min():.3f}/{np.median(arr):.3f}/{arr.max():.3f}")
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(r"min $\Delta R$(jet, photon) per event")
        plt.ylabel("Entries")
        plt.title(r"min $\Delta R$(jet, photon) before overlap removal")
        plt.tight_layout()
        plt.savefig(mindr_before_png_path, dpi=200)
        info(f"wrote {mindr_before_png_path}")

    if len(min_dr_after_per_event) > 0:
        arr = np.array(min_dr_after_per_event, dtype=float)
        info(f"min dR(jet,photon) after: n={len(arr)} min/med/max={arr.min():.3f}/{np.median(arr):.3f}/{arr.max():.3f}")
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(r"min $\Delta R$(jet, photon) per event")
        plt.ylabel("Entries")
        plt.title(r"min $\Delta R$(jet, photon) after overlap removal")
        plt.tight_layout()
        plt.savefig(mindr_after_png_path, dpi=200)
        info(f"wrote {mindr_after_png_path}")

    track_df = pd.DataFrame(track_rows)
    track_df.to_csv(tracks_out_csv_path, index=False)
    info(f"wrote {tracks_out_csv_path}")

    if len(photon_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(photon_track_counts, dtype=int), bins=range(0, max(photon_track_counts) + 2))
        plt.xlabel(f"# tracks within dR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("tracks near photons (using all tracks)")
        plt.tight_layout()
        plt.savefig(tracks_photons_png_path, dpi=200)
        info(f"wrote {tracks_photons_png_path}")

    if len(jet_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(jet_track_counts, dtype=int), bins=range(0, max(jet_track_counts) + 2))
        plt.xlabel(f"# tracks within dR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("tracks near jets (after overlap removal; using all tracks)")
        plt.tight_layout()
        plt.savefig(tracks_jets_png_path, dpi=200)
        info(f"wrote {tracks_jets_png_path}")

    t, ph_eff, jet_misid, acc = best_threshold_separation(photon_track_counts, jet_track_counts)
    if np.isfinite(acc):
        info(
            f"track-count cut (dR<{args.dr_track}): photon if n_tracks<=t, "
            f"t={t}, eff={ph_eff:.3f}, fake={jet_misid:.3f}, rej={1.0-jet_misid:.3f}"
        )

    if np.any(np.isfinite(ph_iso)):
        plt.figure()
        plt.hist(ph_iso[np.isfinite(ph_iso)], bins=60)
        plt.xlabel(rf"$I$ (track isolation), $\Delta R<{args.iso_dr}$")
        plt.ylabel("Entries")
        plt.title("track isolation for photons (using all tracks)")
        plt.tight_layout()
        plt.savefig(iso_photons_png_path, dpi=200)
        info(f"wrote {iso_photons_png_path}")

    if np.any(np.isfinite(jt_iso)):
        plt.figure()
        plt.hist(jt_iso[np.isfinite(jt_iso)], bins=60)
        plt.xlabel(rf"$I$ (track isolation), $\Delta R<{args.iso_dr}$")
        plt.ylabel("Entries")
        plt.title("track isolation for jets (using all tracks)")
        plt.tight_layout()
        plt.savefig(iso_jets_png_path, dpi=200)
        info(f"wrote {iso_jets_png_path}")

    ph_f = ph_iso[np.isfinite(ph_iso)]
    jt_f = jt_iso[np.isfinite(jt_iso)]
    info(f"iso stats: photons n={ph_f.size}, jets n={jt_f.size}")

    # plot ROC (points already computed/written above)
    if len(fpr) > 0:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("Fake rate (jets passing cut)")
        plt.ylabel("Acceptance (photons passing cut)")
        plt.title(rf"Acceptance vs Fake rate (iso), $\Delta R<{args.iso_dr}$")
        plt.tight_layout()
        plt.savefig(roc_png_path, dpi=200)
        info(f"wrote {roc_png_path}")

    TARGET_TPR = 0.99
    cut99, tpr99, fpr99 = cut_at_fixed_tpr(ph_iso, jt_iso, TARGET_TPR)
    if np.isfinite(cut99):
        info(
            f"FIXED TPR={TARGET_TPR:.2f}: cut={cut99:.6g}, "
            f"acc={tpr99:.3f}, fake={fpr99:.3f}, rej={1.0-fpr99:.3f}"
        )


if __name__ == "__main__":
    main()
