#!/usr/bin/env python3
"""
Simple isolation-cut classifier for photon-versus-jet discrimination using
track-based isolation from per-event csv files.

The script:
- reads headerless photon, jet, and track files
- applies jet-photon overlap removal
- computes track-count and isolation quantities around photons and jets
- evaluates a simple isolation-cut classifier at fixed photon-efficiency working points
- saves mass, isolation, ROC, diagnostic plots, and timing summaries

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

import hashlib
import re
import sys
import tarfile
import csv
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# windows console robustness: try to force UTF-8 output where supported
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "initial_data" / "10k_ev"
DEFAULT_OUT_DIR = PROJECT_ROOT / "final_outputs" / "isolation_cut_classifier"

# event discovery pattern: photons_<id>.csv defines the event list
PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")

# default configuration values, also exposed through the command line
CFG: Dict[str, float] = {
    "dr_overlap_default": 0.20,
    "dr_track_default": 0.10,
    "iso_dr_default": 0.20,
    "dr_track_keep_default": 0.05,
}


def _safe_print(prefix: str, msg: str, *, stream=None) -> None:
    """Print defensively to avoid unicode crashes on misconfigured terminals."""
    if stream is None:
        stream = sys.stdout
    try:
        print(f"{prefix} {msg}", file=stream)
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(f"{prefix} {safe}", file=stream)


def info(msg: str) -> None:
    """Write a compact informational message to stdout."""
    _safe_print("[info]", msg)


def warn(msg: str) -> None:
    """Write a warning message to stdout."""
    _safe_print("[warn]", msg)


def err(msg: str) -> None:
    """Write an error message to stderr."""
    _safe_print("[error]", msg, stream=sys.stderr)


def start_plot() -> None:
    """Create a report-sized matplotlib figure with consistent dimensions."""
    plt.figure(figsize=(7, 5))


def finish_plot(path: Path) -> None:
    """Apply final layout settings, save the figure, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_roc_csv(out_dir: Path, fpr, tpr, fname: str = "roc.csv") -> Path:
    """Write a simple two-column ROC CSV to out_dir/fname."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr"])
        for x, y in zip(fpr, tpr):
            w.writerow([float(x), float(y)])
    return path


def write_ntracks_csv(out_dir: Path, n_iso, n_total, fname: str = "ntracks.csv") -> Path:
    """Write track multiplicities inside the iso cone alongside total tracks.

    Columns
    -------
    ntracks_iso
        Tracks within ΔR < iso_dr.
    ntracks_total
        Total tracks in the event after the track pT cut.
    frac_kept
        ntracks_iso / ntracks_total.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ntracks_iso", "ntracks_total", "frac_kept"])
        for a, b in zip(n_iso, n_total):
            a = int(a)
            b = int(b)
            frac = (a / b) if b > 0 else 0.0
            w.writerow([a, b, frac])
    return path


def find_event_ids(data_dir: Path) -> List[int]:
    """Return sorted event IDs discovered from photons_<id>.csv in data_dir."""
    ids: List[int] = []
    for path in data_dir.iterdir():
        m = PHOTON_RE.match(path.name)
        if m:
            ids.append(int(m.group(1)))
    ids.sort()
    return ids


def photons_path(data_dir: Path, event_id: int) -> Path:
    """Return the photon csv path for one event."""
    return data_dir / f"photons_{event_id}.csv"


def jets_path(data_dir: Path, event_id: int) -> Path:
    """Return the jet csv path for one event."""
    return data_dir / f"jets_{event_id}.csv"


def tracks_path(data_dir: Path, event_id: int) -> Path:
    """Return the track csv path for one event."""
    return data_dir / f"tracks_{event_id}.csv"


def delta_phi(phi1: float, phi2: float) -> float:
    """Compute Δφ wrapped into (-π, π]."""
    dphi = phi1 - phi2
    return float((dphi + np.pi) % (2.0 * np.pi) - np.pi)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """Compute ΔR = sqrt((Δη)^2 + (Δφ)^2)."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return float(np.sqrt(deta * deta + dphi * dphi))


def pxyz_from_ptetaphi(pt: float, eta: float, phi: float) -> Tuple[float, float, float]:
    """Convert (pT, η, φ) to Cartesian momentum components."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return float(px), float(py), float(pz)


def inv_mass_from_two_objects(obj1: Dict[str, float], obj2: Dict[str, float]) -> float:
    """Compute the invariant mass of two objects with pT, eta, phi, and e."""
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
    """Return a vectorised ΔR mask selecting tracks within dr_max of one object."""
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
    """Count tracks within ΔR < dr_max of an object using vectorised masking."""
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
    """Compute the scalar track-isolation variable.

    The isolation is defined as

        I = (Σ pT of tracks within ΔR < dr_max) / pT(object)

    Returns NaN if obj_pt <= 0.
    """
    if obj_pt <= 0.0:
        return float("nan")
    if trk_eta.size == 0:
        return 0.0
    m = _dr_mask_to_object(trk_eta, trk_phi, obj_eta, obj_phi, dr_max)
    if m.size == 0:
        return 0.0
    return float(np.sum(trk_pt[m], dtype=float) / float(obj_pt))


def filter_tracks_close_to_photons(df_trk: pd.DataFrame, df_ph: pd.DataFrame, dr_keep: float) -> pd.DataFrame:
    """Return tracks within dr_keep of any photon.

    This filter is diagnostic only and is not used for the downstream classifier.
    """
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
    """Build the ROC curve for a simple isolation cut I < c.

    Photon acceptance
        TPR(c) = P_photon(I < c)

    Jet fake rate
        FPR(c) = P_jet(I < c)

    Returns
    -------
    tuple
        (fpr, tpr) arrays in non-decreasing FPR order.
    """
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

    order = np.argsort(fpr_plot)
    f = fpr_plot[order]
    t = tpr_plot[order]

    uf, first_idx = np.unique(f, return_index=True)
    t_best = np.empty_like(uf)
    for k in range(len(uf)):
        start = first_idx[k]
        end = first_idx[k + 1] if k + 1 < len(uf) else len(f)
        t_best[k] = np.max(t[start:end])

    t_best = np.maximum.accumulate(t_best)
    return uf, t_best


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray, *, max_fpr: float = 1.0) -> float:
    """Compute a full or partial AUC by integrating TPR over FPR."""
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")

    f = np.asarray(fpr, dtype=float)
    t = np.asarray(tpr, dtype=float)

    order = np.argsort(f)
    f = f[order]
    t = t[order]

    m = np.isfinite(f) & np.isfinite(t)
    f = f[m]
    t = t[m]
    if f.size == 0:
        return float("nan")

    max_fpr = float(max_fpr)
    if max_fpr <= 0.0:
        return 0.0

    if f[0] > 0.0:
        f = np.concatenate(([0.0], f))
        t = np.concatenate(([0.0], t))

    if f[-1] < max_fpr:
        f = np.concatenate((f, [max_fpr]))
        t = np.concatenate((t, [t[-1]]))
    else:
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


def cut_at_fixed_tpr(ph_iso: np.ndarray, jt_iso: np.ndarray, target_tpr: float) -> Tuple[float, float, float]:
    """Find the isolation cut giving approximately the requested photon TPR.

    Returns
    -------
    tuple
        (cut, achieved_tpr, achieved_fpr)
    """
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


def read_csv_maybe_gzip(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file and auto-detect gzip compression from the magic bytes."""
    with open(path, "rb") as f:
        head = f.read(2)
    is_gz = (len(head) == 2 and head[0] == 0x1F and head[1] == 0x8B)
    if is_gz:
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def looks_like_text_file(path: Path, *, nbytes: int = 4096) -> bool:
    """Heuristic guard for jets files that may be binary or corrupt.

    Returns False if the file contains NUL bytes or tar markers, or is empty
    or unreadable.
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
    """Force selected columns to numeric and drop rows that fail coercion."""
    if df.empty:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=cols).reset_index(drop=True)
    return out


def prepare_data_dir(path: Path) -> Path:
    """Normalise the dataset path.

    Behaviour
    ---------
    - if path is a directory: return it unchanged
    - if path ends with .tar.gz: extract once to a cached folder and return that folder
    - otherwise: return the absolute path unchanged
    """
    abs_path = path.resolve()

    if abs_path.is_dir():
        return abs_path

    if not str(abs_path).lower().endswith(".tar.gz"):
        return abs_path

    h = hashlib.md5(str(abs_path).encode("utf-8")).hexdigest()[:10]
    extract_dir = abs_path.parent / f"_extracted_{h}"

    if extract_dir.is_dir():
        info(f"using cached extracted dataset: {extract_dir}")
        return extract_dir

    info(f"extracting {abs_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(abs_path, "r:gz") as tar:
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)

    info("extraction complete")
    return extract_dir


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the isolation-cut photon-versus-jet classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing photons_<id>.csv, jets_<id>.csv, and tracks_<id>.csv",
    )
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--out-png", default="mgg_hist.png")
    parser.add_argument("--out-csv", default="mgg_values.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory to save plots, tables, and timing outputs",
    )

    parser.add_argument(
        "--include-converted",
        action="store_true",
        default=False,
        help="include converted photons (default: exclude them)",
    )

    # geometry and feature radii
    parser.add_argument("--dr-overlap", type=float, default=float(CFG["dr_overlap_default"]))
    parser.add_argument("--dr-track", type=float, default=float(CFG["dr_track_default"]))
    parser.add_argument("--dr-track-keep", type=float, default=float(CFG["dr_track_keep_default"]))

    # track pT cut applied before all track-based quantities are computed
    parser.add_argument(
        "--trk-pt-min",
        type=float,
        default=0.75,
        help="only use tracks with pT > trk_pt_min for n_tracks and isolation features",
    )

    # track-count output products
    parser.add_argument("--tracks-photons-png", default="tracks_near_photons.png")
    parser.add_argument("--tracks-jets-png", default="tracks_near_jets.png")
    parser.add_argument("--tracks-out-csv", default="track_counts.csv")

    # overlap-removal diagnostics
    parser.add_argument("--mindr-before-png", default="min_dr_jet_photon_before.png")
    parser.add_argument("--mindr-after-png", default="min_dr_jet_photon_after.png")

    # isolation and ROC products
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
        help="compute a full or partial AUC over FPR in [0, auc_max_fpr]",
    )
    parser.add_argument(
        "--target-tprs",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95],
        help="target photon acceptances at which to report fake rates",
    )
    parser.add_argument(
        "--fixed-tpr-out-csv",
        default="fixed_tpr_metrics.csv",
        help="output CSV listing cut, achieved_tpr, and fake_rate for each target TPR",
    )
    return parser.parse_args()


def main() -> None:
    """Run the isolation-cut analysis and write all outputs."""
    args = parse_args()
    total_start = time.perf_counter()

    exclude_converted = not args.include_converted
    data_dir = prepare_data_dir(args.data_dir)
    out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"data: {data_dir}")
    info(f"out:  {out_dir}")
    info(
        "cfg: "
        f"dr_overlap={args.dr_overlap:g}  dr_track={args.dr_track:g}  dr_track_keep={args.dr_track_keep:g}  "
        f"iso_dr={args.iso_dr:g}  trk_pt_min={args.trk_pt_min:g}  "
        f"converted={'incl' if not exclude_converted else 'excl'}"
    )

    out_csv_path = out_dir / args.out_csv
    out_png_path = out_dir / args.out_png

    tracks_out_csv_path = out_dir / args.tracks_out_csv
    tracks_photons_png_path = out_dir / args.tracks_photons_png
    tracks_jets_png_path = out_dir / args.tracks_jets_png

    mindr_before_png_path = out_dir / args.mindr_before_png
    mindr_after_png_path = out_dir / args.mindr_after_png

    iso_out_csv_path = out_dir / args.iso_out_csv
    iso_photons_png_path = out_dir / args.iso_photons_png
    iso_jets_png_path = out_dir / args.iso_jets_png
    roc_png_path = out_dir / args.roc_png

    roc_out_csv_path = out_dir / args.roc_out_csv
    fixed_tpr_out_csv_path = out_dir / args.fixed_tpr_out_csv
    timing_out_csv_path = out_dir / "timing_summary.csv"

    auc_max_fpr = float(args.auc_max_fpr)
    target_tprs = [float(x) for x in args.target_tprs]

    if not data_dir.is_dir():
        err(f"data-dir does not exist: {data_dir}")
        sys.exit(1)

    event_ids = find_event_ids(data_dir)
    if not event_ids:
        err(f"no photons_<id>.csv found in {data_dir}")
        sys.exit(1)

    chosen = event_ids[: args.n_events]

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
    n_trk_phclose_total = 0

    n_trk_before_pt_total = 0
    n_trk_after_pt_total = 0

    # timing accumulators
    io_time_total = 0.0
    overlap_time_total = 0.0
    feature_time_total = 0.0
    diagnostic_track_filter_time_total = 0.0
    mass_time_total = 0.0
    postproc_time_total = 0.0

    # counters for timing normalisation
    n_events_attempted = len(chosen)
    n_events_used = 0
    n_photons_scored = 0
    n_jets_scored = 0
    n_objects_scored = 0

    for ev in chosen:
        event_used = False

        # I/O and event preparation
        t0_io = time.perf_counter()

        ph_path = photons_path(data_dir, ev)
        if not ph_path.exists() or ph_path.stat().st_size == 0:
            n_bad_ph += 1
            io_time_total += time.perf_counter() - t0_io
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
            io_time_total += time.perf_counter() - t0_io
            continue

        # by default only unconverted photons are used
        if exclude_converted:
            conv_int = pd.to_numeric(df_ph["conversionType"], errors="coerce").fillna(-999).astype(int)
            df_ph = df_ph[conv_int == 0].reset_index(drop=True)

        jet_p = jets_path(data_dir, ev)
        if jet_p.exists() and jet_p.stat().st_size > 0:
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

        trk_p = tracks_path(data_dir, ev)
        if trk_p.exists() and trk_p.stat().st_size > 0:
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

        # apply the global track pT threshold before any downstream use
        n_trk_before_pt = int(len(df_trk_all))
        if args.trk_pt_min > 0.0 and n_trk_before_pt > 0:
            df_trk_all = df_trk_all[df_trk_all["pT"] > float(args.trk_pt_min)].reset_index(drop=True)
        n_trk_after_pt = int(len(df_trk_all))

        n_trk_before_pt_total += n_trk_before_pt
        n_trk_after_pt_total += n_trk_after_pt
        n_trk_all_total += len(df_trk_all)

        trk_all_eta = df_trk_all["eta"].to_numpy(dtype=float) if len(df_trk_all) else np.zeros(0, dtype=float)
        trk_all_phi = df_trk_all["phi"].to_numpy(dtype=float) if len(df_trk_all) else np.zeros(0, dtype=float)
        trk_all_pt = df_trk_all["pT"].to_numpy(dtype=float) if len(df_trk_all) else np.zeros(0, dtype=float)

        io_time_total += time.perf_counter() - t0_io

        # diagnostic-only photon-close track filter
        t0_diag = time.perf_counter()
        df_trk_phclose = filter_tracks_close_to_photons(df_trk_all, df_ph, float(args.dr_track_keep))
        n_trk_phclose_total += len(df_trk_phclose)
        diagnostic_track_filter_time_total += time.perf_counter() - t0_diag

        # overlap removal
        t0_overlap = time.perf_counter()

        n_jets_before += len(df_j)

        kept: List[pd.Series] = []
        min_dr_before = np.inf

        for _, jet in df_j.iterrows():
            overlap = False
            for _, pho in df_ph.iterrows():
                dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                min_dr_before = min(min_dr_before, dr)
                if dr < args.dr_overlap:
                    overlap = True
            if not overlap:
                kept.append(jet)

        if np.isfinite(min_dr_before):
            min_dr_before_per_event.append(float(min_dr_before))

        if len(kept) > 0 and len(df_ph) > 0:
            min_dr_after = np.inf
            for jet in kept:
                for _, pho in df_ph.iterrows():
                    dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                    min_dr_after = min(min_dr_after, dr)
            if np.isfinite(min_dr_after):
                min_dr_after_per_event.append(float(min_dr_after))

        n_jets_after += len(kept)
        overlap_time_total += time.perf_counter() - t0_overlap

        # track-based feature construction for the isolation-cut classifier
        t0_feat = time.perf_counter()

        local_photons = 0
        local_jets = 0

        for i, pho in df_ph.iterrows():
            pho_eta = float(pho["eta"])
            pho_phi = float(pho["phi"])
            pho_pt = float(pho["pT"])

            ntrk = count_tracks_near_object_fast(pho_eta, pho_phi, trk_all_eta, trk_all_phi, args.dr_track)
            iso = track_iso_scalar_sum_pt_fast(
                pho_eta, pho_phi, pho_pt, trk_all_eta, trk_all_phi, trk_all_pt, args.iso_dr
            )

            ntrk_iso = count_tracks_near_object_fast(pho_eta, pho_phi, trk_all_eta, trk_all_phi, args.iso_dr)
            photon_total_ntracks.append(int(len(trk_all_pt)))
            photon_iso_ntracks.append(ntrk_iso)

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

            local_photons += 1

        for i, jet in enumerate(kept):
            jet_eta = float(jet["eta"])
            jet_phi = float(jet["phi"])
            jet_pt = float(jet["pT"])

            ntrk = count_tracks_near_object_fast(jet_eta, jet_phi, trk_all_eta, trk_all_phi, args.dr_track)
            iso = track_iso_scalar_sum_pt_fast(
                jet_eta, jet_phi, jet_pt, trk_all_eta, trk_all_phi, trk_all_pt, args.iso_dr
            )

            ntrk_iso = count_tracks_near_object_fast(jet_eta, jet_phi, trk_all_eta, trk_all_phi, args.iso_dr)
            jet_total_ntracks.append(int(len(trk_all_pt)))
            jet_iso_ntracks.append(ntrk_iso)

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

            local_jets += 1

        feature_time_total += time.perf_counter() - t0_feat

        n_photons_scored += local_photons
        n_jets_scored += local_jets
        n_objects_scored += local_photons + local_jets
        event_used = True

        # diphoton mass building for events with exactly two photons
        if len(df_ph) != 2:
            if event_used:
                n_events_used += 1
            continue

        t0_mass = time.perf_counter()

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

        mass_time_total += time.perf_counter() - t0_mass

        if event_used:
            n_events_used += 1

    # post-processing, CSVs, and plots
    t0_post = time.perf_counter()

    if n_trk_all_total > 0:
        frac = float(n_trk_phclose_total) / float(n_trk_all_total)
        pd.DataFrame(
            [
                {
                    "kept_tracks": int(n_trk_phclose_total),
                    "all_tracks": int(n_trk_all_total),
                    "kept_fraction": float(frac),
                    "dr_track_keep": float(args.dr_track_keep),
                }
            ]
        ).to_csv(out_dir / "track_reduction.csv", index=False)

    pt_frac = (
        float(n_trk_after_pt_total) / float(n_trk_before_pt_total) if n_trk_before_pt_total > 0 else float("nan")
    )
    pd.DataFrame(
        [
            {
                "trk_pt_min": float(args.trk_pt_min),
                "tracks_before_pt": int(n_trk_before_pt_total),
                "tracks_after_pt": int(n_trk_after_pt_total),
                "kept_fraction_pt": float(pt_frac),
            }
        ]
    ).to_csv(out_dir / "track_ptmin_reduction.csv", index=False)

    pd.DataFrame(iso_rows).to_csv(iso_out_csv_path, index=False)

    write_ntracks_csv(out_dir, photon_iso_ntracks, photon_total_ntracks, fname="ntracks_photons.csv")
    write_ntracks_csv(out_dir, jet_iso_ntracks, jet_total_ntracks, fname="ntracks_jets.csv")
    write_ntracks_csv(
        out_dir, photon_iso_ntracks + jet_iso_ntracks, photon_total_ntracks + jet_total_ntracks, fname="ntracks.csv"
    )

    ph_iso = np.array(photon_iso_vals, dtype=float)
    jt_iso = np.array(jet_iso_vals, dtype=float)

    fpr, tpr = roc_from_iso(ph_iso, jt_iso)

    auc = float("nan")
    fixed_tpr_rows: List[Dict[str, float]] = []

    if len(fpr) > 0:
        pd.DataFrame({"fake_rate": fpr, "acceptance": tpr}).to_csv(roc_out_csv_path, index=False)
        write_roc_csv(out_dir, fpr, tpr, fname="roc.csv")

        auc = auc_from_roc(fpr, tpr, max_fpr=auc_max_fpr)

        for target in target_tprs:
            cut, achieved_tpr, achieved_fpr = cut_at_fixed_tpr(ph_iso, jt_iso, target)
            fixed_tpr_rows.append(
                {
                    "target_tpr": float(target),
                    "cut": float(cut),
                    "achieved_tpr": float(achieved_tpr),
                    "fake_rate": float(achieved_fpr),
                }
            )

        pd.DataFrame(fixed_tpr_rows).to_csv(fixed_tpr_out_csv_path, index=False)

    if not mgg_rows:
        err("no valid 2-photon events found")
        sys.exit(1)

    out = pd.DataFrame(mgg_rows)
    out.to_csv(out_csv_path, index=False)

    start_plot()
    plt.hist(out["m_gg"].to_numpy(dtype=float), bins=args.bins)
    plt.xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    plt.ylabel("Entries")
    finish_plot(out_png_path)

    if len(min_dr_before_per_event) > 0:
        arr = np.array(min_dr_before_per_event, dtype=float)
        start_plot()
        plt.hist(arr, bins=60, range=(0.0, 1.0))
        plt.xlabel(r"Minimum $\Delta R(\mathrm{jet}, \gamma)$ per event")
        plt.ylabel("Entries")
        finish_plot(mindr_before_png_path)

    if len(min_dr_after_per_event) > 0:
        arr = np.array(min_dr_after_per_event, dtype=float)
        start_plot()
        plt.hist(arr, bins=60, range=(0.0, 1.0))
        plt.xlabel(r"Minimum $\Delta R(\mathrm{jet}, \gamma)$ per event")
        plt.ylabel("Entries")
        finish_plot(mindr_after_png_path)

    pd.DataFrame(track_rows).to_csv(tracks_out_csv_path, index=False)

    if len(photon_track_counts) > 0:
        start_plot()
        plt.hist(
            np.array(photon_track_counts, dtype=int),
            bins=range(0, max(photon_track_counts) + 2),
            density=True,
        )
        plt.xlabel(rf"Number of tracks within $\Delta R < {args.dr_track:.2f}$")
        plt.ylabel("Density")
        finish_plot(tracks_photons_png_path)

    if len(jet_track_counts) > 0:
        start_plot()
        plt.hist(
            np.array(jet_track_counts, dtype=int),
            bins=range(0, max(jet_track_counts) + 2),
            density=True,
        )
        plt.xlabel(rf"Number of tracks within $\Delta R < {args.dr_track:.2f}$")
        plt.ylabel("Density")
        finish_plot(tracks_jets_png_path)

    # main isolation comparison plot for the report
    mask_ph = (ph_iso > 0) & np.isfinite(ph_iso)
    mask_jt = (jt_iso > 0) & np.isfinite(jt_iso)

    if np.any(mask_ph) and np.any(mask_jt):
        start_plot()
        plt.hist(
            ph_iso[mask_ph],
            bins=60,
            range=(0.0, 0.08),
            histtype="step",
            linewidth=2,
            density=True,
            label="Photons",
        )
        plt.hist(
            jt_iso[mask_jt],
            bins=60,
            range=(0.0, 0.08),
            histtype="step",
            linewidth=2,
            density=True,
            label="Jets",
        )
        plt.yscale("log")
        plt.xlabel(rf"Track isolation $I$ within $\Delta R < {args.iso_dr:.2f}$")
        plt.ylabel("Normalised density")
        plt.xlim(0.0, 0.08)
        plt.xticks([0.00, 0.02, 0.04, 0.06, 0.08])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        plt.legend(loc="lower right", frameon=True)
        finish_plot(out_dir / "iso_zoomed.png")

    if len(fpr) > 0:
        start_plot()
        plt.plot(fpr, tpr, label=rf"Isolation classifier (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="Random classifier")
        plt.xlabel("Jet fake rate")
        plt.ylabel("Photon acceptance")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.legend(loc="lower right", frameon=True)
        finish_plot(roc_png_path)

    postproc_time_total += time.perf_counter() - t0_post
    total_wall_time = time.perf_counter() - total_start

    kept_frac_diag = (float(n_trk_phclose_total) / float(n_trk_all_total)) if n_trk_all_total > 0 else float("nan")

    fixed_tpr_summary = ""
    if fixed_tpr_rows:
        parts = []
        for row in fixed_tpr_rows:
            pct = int(round(100.0 * row["target_tpr"]))
            parts.append(f"fake@{pct}={row['fake_rate']:.3f}")
        fixed_tpr_summary = "  " + "  ".join(parts)

    # derived timing metrics
    io_time_per_event = io_time_total / n_events_used if n_events_used > 0 else np.nan
    overlap_time_per_event = overlap_time_total / n_events_used if n_events_used > 0 else np.nan
    feature_time_per_event = feature_time_total / n_events_used if n_events_used > 0 else np.nan
    diagnostic_track_filter_time_per_event = (
        diagnostic_track_filter_time_total / n_events_used if n_events_used > 0 else np.nan
    )
    mass_time_per_two_photon_event = mass_time_total / n_two_photon if n_two_photon > 0 else np.nan
    total_wall_time_per_event = total_wall_time / n_events_used if n_events_used > 0 else np.nan

    feature_time_per_object = feature_time_total / n_objects_scored if n_objects_scored > 0 else np.nan
    overlap_plus_feature_time_total = overlap_time_total + feature_time_total
    overlap_plus_feature_time_per_event = (
        overlap_plus_feature_time_total / n_events_used if n_events_used > 0 else np.nan
    )

    timing_df = pd.DataFrame(
        [
            {
                "events_attempted": int(n_events_attempted),
                "events_used": int(n_events_used),
                "events_read": int(n_read),
                "two_photon_events": int(n_two_photon),
                "photons_scored": int(n_photons_scored),
                "jets_scored": int(n_jets_scored),
                "objects_scored": int(n_objects_scored),
                "io_time_total_s": float(io_time_total),
                "diagnostic_track_filter_time_total_s": float(diagnostic_track_filter_time_total),
                "overlap_time_total_s": float(overlap_time_total),
                "feature_time_total_s": float(feature_time_total),
                "mass_time_total_s": float(mass_time_total),
                "postproc_time_total_s": float(postproc_time_total),
                "overlap_plus_feature_time_total_s": float(overlap_plus_feature_time_total),
                "total_wall_time_s": float(total_wall_time),
                "io_time_per_used_event_s": float(io_time_per_event),
                "diagnostic_track_filter_time_per_used_event_s": float(diagnostic_track_filter_time_per_event),
                "overlap_time_per_used_event_s": float(overlap_time_per_event),
                "feature_time_per_used_event_s": float(feature_time_per_event),
                "feature_time_per_object_s": float(feature_time_per_object),
                "overlap_plus_feature_time_per_used_event_s": float(overlap_plus_feature_time_per_event),
                "mass_time_per_two_photon_event_s": float(mass_time_per_two_photon_event),
                "total_wall_time_per_used_event_s": float(total_wall_time_per_event),
                "dr_overlap": float(args.dr_overlap),
                "dr_track": float(args.dr_track),
                "iso_dr": float(args.iso_dr),
                "trk_pt_min": float(args.trk_pt_min),
                "auc_max_fpr": float(auc_max_fpr),
                "auc": float(auc),
                "n_events_requested": int(args.n_events),
                "converted_photons": "excluded" if exclude_converted else "included",
            }
        ]
    )
    timing_df.to_csv(timing_out_csv_path, index=False)

    info(
        "summary: "
        f"events={len(chosen)} read={n_read} bad_ph={n_bad_ph} "
        f"two_photon={n_two_photon} bad_jets={n_bad_jets} "
        f"trk_keep_frac_diag={kept_frac_diag:.3f} trk_keep_frac_pt={pt_frac:.3f} "
        f"auc(max_fpr={auc_max_fpr:g})={auc:.6f}"
        f"{fixed_tpr_summary}"
    )

    info(
        "timing: "
        f"io_total={io_time_total:.3f}s  "
        f"overlap_total={overlap_time_total:.3f}s  "
        f"feature_total={feature_time_total:.3f}s  "
        f"diag_filter_total={diagnostic_track_filter_time_total:.3f}s  "
        f"postproc_total={postproc_time_total:.3f}s  "
        f"wall_total={total_wall_time:.3f}s"
    )
    info(
        "timing_per_event: "
        f"io={io_time_per_event:.6e}s  "
        f"overlap={overlap_time_per_event:.6e}s  "
        f"feature={feature_time_per_event:.6e}s  "
        f"overlap+feature={overlap_plus_feature_time_per_event:.6e}s  "
        f"wall={total_wall_time_per_event:.6e}s"
    )
    info(f"timing_per_object: feature={feature_time_per_object:.6e}s")


if __name__ == "__main__":
    main()