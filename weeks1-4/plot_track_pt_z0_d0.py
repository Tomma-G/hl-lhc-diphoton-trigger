#!/usr/bin/env python3
"""
Track feature diagnostics for the HL-LHC diphoton toy dataset.

Reads the same per-event CSVs as your main analysis script (no headers):
  photons_<id>.csv : pT, eta, phi, e, conversionType
  tracks_<id>.csv  : pT, eta, phi, eTot, z0, d0

Outputs (into --out-dir):
  - 1D hists for pT/z0/d0 (entries + density)
  - zoom variants (percentile-clipped x-range)
  - optional y-axis capped variants for linear-y plots (prevents one huge bin crushing others)
  - optional log-x binning for full-range pT (helps long tails)
  - 2D: z0_vs_d0, pt_vs_d0, pt_vs_z0 (+ zoom variants), with LogNorm + vmax clipping options
  - trk_feature_summary_*.csv with fractions + percentiles
  - trk_pt_outliers_*.csv with event_id + rows for very large pT

Optional:
  --dr-track <val> keeps only tracks with min ΔR(track, any photon) < val
  (inspection only; does not change your main analysis features)

Extras:
  --zoom / --no-zoom : percentile-clipped "zoom" plots to avoid single outliers ruining axes

Binning:
  For fine binning, use explicit bin widths via:
    --binw-pt, --binw-z0, --binw-d0
  These create explicit bin edges (robust against outliers).
  --bins is kept as a fallback if edges cannot be constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tarfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --- Windows console robustness: force UTF-8 output ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")

QUIET = False


def _safe_print(prefix: str, msg: str, *, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    try:
        print(f"{prefix} {msg}", file=stream)
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(f"{prefix} {safe}", file=stream)


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


def tracks_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"tracks_{event_id}.csv")


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


def coerce_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Force selected columns to numeric and drop rows that fail."""
    if df.empty:
        return df
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=cols).reset_index(drop=True)
    return out


def delta_phi(phi1: np.ndarray, phi2: float) -> np.ndarray:
    dphi = phi1 - float(phi2)
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def min_dr_track_to_photons(
    trk_eta: np.ndarray,
    trk_phi: np.ndarray,
    pho_eta: np.ndarray,
    pho_phi: np.ndarray,
) -> np.ndarray:
    """For each track, compute min ΔR(track, photon) over photons in the event."""
    if trk_eta.size == 0:
        return np.zeros(0, dtype=float)
    if pho_eta.size == 0:
        return np.full(trk_eta.shape[0], np.inf, dtype=float)

    dr2_min = np.full(trk_eta.shape[0], np.inf, dtype=float)
    for j in range(pho_eta.size):
        deta = trk_eta - float(pho_eta[j])
        dphi = delta_phi(trk_phi, float(pho_phi[j]))
        dr2 = deta * deta + dphi * dphi
        dr2_min = np.minimum(dr2_min, dr2)
    return np.sqrt(dr2_min)


def savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def safe_percentile(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def make_edges(lo: float, hi: float, binw: float) -> np.ndarray:
    """
    Construct explicit histogram bin edges from [lo, hi] and a desired bin width.
    Returns empty array if inputs are invalid.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or not np.isfinite(binw) or binw <= 0:
        return np.array([])
    if hi <= lo:
        return np.array([])

    start = binw * np.floor(lo / binw)
    stop = binw * np.ceil(hi / binw)
    edges = np.arange(start, stop + binw, binw, dtype=float)
    if edges.size < 2:
        return np.array([])
    return edges


def make_log_edges(lo: float, hi: float, n_edges: int) -> np.ndarray:
    """
    Construct log-spaced bin edges from [lo, hi]. Requires lo>0.
    n_edges is number of edges (so bins = n_edges-1).
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= lo or n_edges < 2:
        return np.array([])
    return np.logspace(np.log10(lo), np.log10(hi), n_edges, dtype=float)


def finite_minmax(arr: np.ndarray) -> tuple[float, float]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.min(a)), float(np.max(a)))


def hist_ylim_cap(x: np.ndarray, bins, q: float, *, density: bool) -> Optional[float]:
    """
    Compute a y-axis cap using the percentile of *actual histogram heights*,
    matching whether we're plotting density or counts.

    This fixes the "blank density plot" bug (cap was previously computed on counts
    even when density=True).
    """
    if q is None or (not np.isfinite(q)) or q <= 0:
        return None

    x = x[np.isfinite(x)]
    if x.size == 0:
        return None

    try:
        heights, _ = np.histogram(x, bins=bins, density=density)
    except Exception:
        return None

    nz = heights[heights > 0]
    if nz.size == 0:
        return None

    return float(np.percentile(nz, q))


def hist_1d(
    x: np.ndarray,
    bins,  # int OR np.ndarray of edges
    title: str,
    xlabel: str,
    outpath: str,
    *,
    logy: bool = False,
    logx: bool = False,
    density: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    x = x[np.isfinite(x)]
    if x.size == 0:
        warn(f"no finite entries for: {title}")
        return

    plt.figure()
    plt.hist(x, bins=bins, density=density)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density" if density else "entries")
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    savefig(outpath)


def hist_2d(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: str,
    *,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    lognorm: bool = True,
    vmax_q: float = 99.5,
) -> None:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        warn(f"no finite entries for: {title}")
        return

    plt.figure()

    # compute a stable norm (LogNorm + vmax clipping), using histogram counts
    norm = None
    if lognorm:
        try:
            h, _, _ = np.histogram2d(x, y, bins=bins)
            nz = h[h > 0]
            if nz.size:
                vmax = float(np.percentile(nz, vmax_q)) if np.isfinite(vmax_q) and vmax_q > 0 else float(np.max(nz))
                vmax = max(vmax, 1.0)
                norm = mcolors.LogNorm(vmin=1.0, vmax=vmax)
        except Exception:
            norm = None

    plt.hist2d(x, y, bins=bins, norm=norm)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="entries")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    savefig(outpath)


def summarise_noise(pt: np.ndarray, z0: np.ndarray, d0: np.ndarray) -> Dict[str, float]:
    pt = pt[np.isfinite(pt)]
    z0 = z0[np.isfinite(z0)]
    d0 = d0[np.isfinite(d0)]

    out: Dict[str, float] = {"n_tracks": float(pt.size)}
    if pt.size == 0:
        return out

    out["frac_pt_lt_0p5"] = float(np.mean(pt < 0.5))
    out["frac_pt_lt_1p0"] = float(np.mean(pt < 1.0))
    out["frac_abs_d0_gt_2"] = float(np.mean(np.abs(d0) > 2.0))
    out["frac_abs_d0_gt_5"] = float(np.mean(np.abs(d0) > 5.0))
    out["frac_abs_z0_gt_50"] = float(np.mean(np.abs(z0) > 50.0))
    out["frac_abs_z0_gt_200"] = float(np.mean(np.abs(z0) > 200.0))

    for name, arr in [("pt", pt), ("d0", d0), ("z0", z0)]:
        qs = np.percentile(arr, [1, 5, 50, 95, 99])
        out[f"{name}_p01"] = float(qs[0])
        out[f"{name}_p05"] = float(qs[1])
        out[f"{name}_p50"] = float(qs[2])
        out[f"{name}_p95"] = float(qs[3])
        out[f"{name}_p99"] = float(qs[4])

    return out


def main() -> None:
    global QUIET

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/1k_ev",
    )
    ap.add_argument(
        "--out-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/track_feature_plots",
    )
    ap.add_argument("--n-events", type=int, default=1000)

    ap.add_argument("--bins", type=int, default=80, help="fallback 1D bin count (used only if explicit edges fail)")
    ap.add_argument("--bins2d", type=int, default=80)

    ap.add_argument("--binw-pt", type=float, default=0.2, help="pT bin width for 1D hists (default 0.2)")
    ap.add_argument("--binw-z0", type=float, default=1.0, help="z0 bin width for 1D hists (default 1.0)")
    ap.add_argument("--binw-d0", type=float, default=0.05, help="d0 bin width for 1D hists (default 0.05)")

    ap.add_argument(
        "--ycap-q",
        type=float,
        default=99.5,
        help="for linear-y hists: cap y-axis at this percentile of nonzero bin heights (default 99.5); set <=0 to disable",
    )

    ap.add_argument("--pt-logbins", dest="pt_logbins", action="store_true", default=True,
                    help="use log-spaced bins + log-x axis for full-range pT plots (default on)")
    ap.add_argument("--no-pt-logbins", dest="pt_logbins", action="store_false",
                    help="disable log-spaced bins for full-range pT plots")

    ap.add_argument(
        "--hist2d-lognorm",
        dest="hist2d_lognorm",
        action="store_true",
        default=True,
        help="use LogNorm scaling for 2D hists (default on)",
    )
    ap.add_argument(
        "--no-hist2d-lognorm",
        dest="hist2d_lognorm",
        action="store_false",
        help="disable LogNorm scaling for 2D hists",
    )
    ap.add_argument(
        "--hist2d-vmax-q",
        type=float,
        default=99.5,
        help="2D colour vmax percentile over nonzero bins (default 99.5); <=0 disables clipping",
    )

    ap.add_argument(
        "--dr-track",
        type=float,
        default=None,
        help="if set: keep only tracks with min ΔR(track, any photon) < dr-track",
    )
    ap.add_argument(
        "--include-converted",
        action="store_true",
        default=False,
        help="if false: exclude converted photons when computing min ΔR selection",
    )
    ap.add_argument("--quiet", action="store_true", default=False)

    ap.add_argument("--zoom", dest="zoom", action="store_true", default=True, help="also write percentile-zoom plots (default on)")
    ap.add_argument("--no-zoom", dest="zoom", action="store_false", help="disable zoom plots")
    ap.add_argument("--zoom-pt-q", type=float, default=99.9, help="upper percentile for pT zoom (default 99.9)")
    ap.add_argument("--zoom-qlo", type=float, default=0.1, help="lower percentile for z0/d0 zoom (default 0.1)")
    ap.add_argument("--zoom-qhi", type=float, default=99.9, help="upper percentile for z0/d0 zoom (default 99.9)")

    ap.add_argument("--pt-outlier-threshold", type=float, default=2000.0, help="write tracks with pT above this to CSV (default 2000)")
    ap.add_argument("--max-outliers", type=int, default=2000, help="cap rows stored for outlier CSV (default 2000)")

    args = ap.parse_args()
    QUIET = bool(args.quiet)

    data_dir = prepare_data_dir(args.data_dir)
    if not os.path.isdir(data_dir):
        err(f"data-dir does not exist: {data_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    event_ids = find_event_ids(data_dir)
    if not event_ids:
        err(f"no photons_<id>.csv found in {data_dir}")
        sys.exit(1)

    chosen = event_ids[: args.n_events]
    info(f"found {len(event_ids)} photon files; processing {len(chosen)} events")
    info(f"track selection: {'min ΔR < ' + str(args.dr_track) if args.dr_track is not None else 'none (using all tracks)'}")
    info(f"zoom plots: {'ON' if args.zoom else 'OFF'}")
    info(f"pt outlier threshold: {args.pt_outlier_threshold:g}")
    info(f"1D bin widths: pt={args.binw_pt:g}, z0={args.binw_z0:g}, d0={args.binw_d0:g} (fallback bins={args.bins})")
    info(f"linear-y y-cap percentile: {args.ycap_q:g} (<=0 disables)")
    info(f"pT full-range log bins + log-x axis: {'ON' if args.pt_logbins else 'OFF'}")
    info(f"2D LogNorm: {'ON' if args.hist2d_lognorm else 'OFF'} (vmax_q={args.hist2d_vmax_q:g})")

    all_pt: List[float] = []
    all_z0: List[float] = []
    all_d0: List[float] = []

    outlier_rows: List[dict] = []

    n_used = 0
    n_bad = 0
    n_no_tracks = 0
    n_no_photons = 0

    for ev in chosen:
        ph_path = photons_path(data_dir, ev)
        tr_path = tracks_path(data_dir, ev)

        if (not os.path.exists(ph_path)) or os.path.getsize(ph_path) == 0:
            n_bad += 1
            continue

        if (not os.path.exists(tr_path)) or os.path.getsize(tr_path) == 0:
            n_no_tracks += 1
            continue

        try:
            df_ph = pd.read_csv(ph_path, header=None, names=["pT", "eta", "phi", "e", "conversionType"])[
                ["pT", "eta", "phi", "e", "conversionType"]
            ]
            df_ph = coerce_numeric_df(df_ph, ["pT", "eta", "phi", "e"])
        except Exception as e:
            warn(f"failed to read {ph_path}: {e}")
            n_bad += 1
            continue

        if not args.include_converted:
            conv_int = pd.to_numeric(df_ph["conversionType"], errors="coerce").fillna(-999).astype(int)
            df_ph = df_ph[conv_int == 0].reset_index(drop=True)

        try:
            df_trk = pd.read_csv(tr_path, header=None, names=["pT", "eta", "phi", "eTot", "z0", "d0"])
            df_trk = coerce_numeric_df(df_trk, ["pT", "eta", "phi", "z0", "d0"])
        except Exception as e:
            warn(f"failed to read {tr_path}: {e}")
            n_bad += 1
            continue

        if len(df_trk) == 0:
            n_no_tracks += 1
            continue

        if len(df_ph) == 0:
            n_no_photons += 1

        if args.dr_track is not None:
            trk_eta = df_trk["eta"].to_numpy(dtype=float)
            trk_phi = df_trk["phi"].to_numpy(dtype=float)
            pho_eta = df_ph["eta"].to_numpy(dtype=float) if len(df_ph) else np.zeros(0, dtype=float)
            pho_phi = df_ph["phi"].to_numpy(dtype=float) if len(df_ph) else np.zeros(0, dtype=float)

            drmin = min_dr_track_to_photons(trk_eta, trk_phi, pho_eta, pho_phi)
            df_trk = df_trk.loc[drmin < float(args.dr_track)].reset_index(drop=True)

        if len(df_trk) == 0:
            continue

        if args.pt_outlier_threshold is not None and len(outlier_rows) < int(args.max_outliers):
            m = df_trk["pT"].to_numpy(dtype=float) > float(args.pt_outlier_threshold)
            if np.any(m):
                take = df_trk.loc[m, ["pT", "eta", "phi", "z0", "d0", "eTot"]].copy()
                take.insert(0, "event_id", int(ev))
                outlier_rows.extend(take.to_dict(orient="records"))
                if len(outlier_rows) > int(args.max_outliers):
                    outlier_rows = outlier_rows[: int(args.max_outliers)]

        all_pt.extend(df_trk["pT"].to_numpy(dtype=float).tolist())
        all_z0.extend(df_trk["z0"].to_numpy(dtype=float).tolist())
        all_d0.extend(df_trk["d0"].to_numpy(dtype=float).tolist())
        n_used += 1

    pt = np.array(all_pt, dtype=float)
    z0 = np.array(all_z0, dtype=float)
    d0 = np.array(all_d0, dtype=float)

    suffix = "all_tracks" if args.dr_track is None else f"drtrk_lt_{args.dr_track:g}"
    info(f"events used: {n_used}")
    info(f"events skipped (bad photons/tracks read): {n_bad}")
    info(f"events with no tracks file/empty: {n_no_tracks}")
    info(f"events with 0 photons after conv filter: {n_no_photons}")
    info(f"total tracks collected: {pt.size} ({suffix})")

    # --------------------------
    # bin edges
    # --------------------------
    pt_lo_full, pt_hi_full = finite_minmax(pt)
    z0_lo_full, z0_hi_full = finite_minmax(z0)
    d0_lo_full, d0_hi_full = finite_minmax(d0)

    if np.isfinite(pt_lo_full) and np.isfinite(pt_hi_full):
        pt_lo_full = max(0.0, pt_lo_full)

    pt_edges_full = np.array([])
    pt_logx_full = False
    if args.pt_logbins:
        pt_pos = pt[np.isfinite(pt) & (pt > 0)]
        if pt_pos.size:
            pt_edges_full = make_log_edges(float(np.min(pt_pos)), float(np.max(pt_pos)), max(60, int(args.bins) + 1))
            if pt_edges_full.size:
                pt_logx_full = True
    if pt_edges_full.size == 0:
        pt_edges_full = make_edges(pt_lo_full, pt_hi_full, float(args.binw_pt))

    z0_edges_full = make_edges(z0_lo_full, z0_hi_full, float(args.binw_z0))
    d0_edges_full = make_edges(d0_lo_full, d0_hi_full, float(args.binw_d0))

    if args.zoom:
        pt_lo_zoom = 0.0
        pt_hi_zoom = safe_percentile(pt, float(args.zoom_pt_q))
        z0_lo_zoom = safe_percentile(z0, float(args.zoom_qlo))
        z0_hi_zoom = safe_percentile(z0, float(args.zoom_qhi))
        d0_lo_zoom = safe_percentile(d0, float(args.zoom_qlo))
        d0_hi_zoom = safe_percentile(d0, float(args.zoom_qhi))

        pt_edges_zoom = make_edges(pt_lo_zoom, pt_hi_zoom, float(args.binw_pt))
        z0_edges_zoom = make_edges(z0_lo_zoom, z0_hi_zoom, float(args.binw_z0))
        d0_edges_zoom = make_edges(d0_lo_zoom, d0_hi_zoom, float(args.binw_d0))
    else:
        pt_lo_zoom = pt_hi_zoom = float("nan")
        z0_lo_zoom = z0_hi_zoom = float("nan")
        d0_lo_zoom = d0_hi_zoom = float("nan")
        pt_edges_zoom = np.array([])
        z0_edges_zoom = np.array([])
        d0_edges_zoom = np.array([])

    # --------------------------
    # 1D: base plots
    # --------------------------
    bins_pt_full = pt_edges_full if pt_edges_full.size else args.bins
    bins_z0_full = z0_edges_full if z0_edges_full.size else args.bins
    bins_d0_full = d0_edges_full if d0_edges_full.size else args.bins

    # pT entries: linear (capped) + logy
    ycap = hist_ylim_cap(pt, bins_pt_full, float(args.ycap_q), density=False)
    hist_1d(
        pt,
        bins=bins_pt_full,
        title=f"track pT ({suffix})",
        xlabel="pT",
        outpath=os.path.join(args.out_dir, f"trk_pt_{suffix}_linear.png"),
        logy=False,
        logx=pt_logx_full,
        density=False,
        ylim=(0.0, ycap) if ycap is not None else None,
    )
    hist_1d(
        pt,
        bins=bins_pt_full,
        title=f"track pT ({suffix})",
        xlabel="pT",
        outpath=os.path.join(args.out_dir, f"trk_pt_{suffix}.png"),
        logy=True,
        logx=pt_logx_full,
        density=False,
    )

    # z0 entries: linear capped + uncapped
    ycap = hist_ylim_cap(z0, bins_z0_full, float(args.ycap_q), density=False)
    hist_1d(
        z0,
        bins=bins_z0_full,
        title=f"track z0 ({suffix})",
        xlabel="z0",
        outpath=os.path.join(args.out_dir, f"trk_z0_{suffix}_linear.png"),
        density=False,
        ylim=(0.0, ycap) if ycap is not None else None,
    )
    hist_1d(
        z0,
        bins=bins_z0_full,
        title=f"track z0 ({suffix})",
        xlabel="z0",
        outpath=os.path.join(args.out_dir, f"trk_z0_{suffix}.png"),
        density=False,
    )

    # d0 entries: linear capped + uncapped
    ycap = hist_ylim_cap(d0, bins_d0_full, float(args.ycap_q), density=False)
    hist_1d(
        d0,
        bins=bins_d0_full,
        title=f"track d0 ({suffix})",
        xlabel="d0",
        outpath=os.path.join(args.out_dir, f"trk_d0_{suffix}_linear.png"),
        density=False,
        ylim=(0.0, ycap) if ycap is not None else None,
    )
    hist_1d(
        d0,
        bins=bins_d0_full,
        title=f"track d0 ({suffix})",
        xlabel="d0",
        outpath=os.path.join(args.out_dir, f"trk_d0_{suffix}.png"),
        density=False,
    )

    # --------------------------
    # 1D: density plots (THIS is where your blank plots came from)
    # --------------------------
    ycap = hist_ylim_cap(pt, bins_pt_full, float(args.ycap_q), density=True)
    hist_1d(
        pt,
        bins=bins_pt_full,
        title=f"track pT density ({suffix})",
        xlabel="pT",
        outpath=os.path.join(args.out_dir, f"trk_pt_density_{suffix}_linear.png"),
        logx=pt_logx_full,
        density=True,
        ylim=(0.0, ycap) if ycap is not None else None,
    )
    hist_1d(
        pt,
        bins=bins_pt_full,
        title=f"track pT density ({suffix})",
        xlabel="pT",
        outpath=os.path.join(args.out_dir, f"trk_pt_density_{suffix}.png"),
        logy=True,
        logx=pt_logx_full,
        density=True,
    )

    ycap = hist_ylim_cap(z0, bins_z0_full, float(args.ycap_q), density=True)
    hist_1d(
        z0,
        bins=bins_z0_full,
        title=f"track z0 density ({suffix})",
        xlabel="z0",
        outpath=os.path.join(args.out_dir, f"trk_z0_density_{suffix}_linear.png"),
        density=True,
        ylim=(0.0, ycap) if ycap is not None else None,
    )
    hist_1d(
        z0,
        bins=bins_z0_full,
        title=f"track z0 density ({suffix})",
        xlabel="z0",
        outpath=os.path.join(args.out_dir, f"trk_z0_density_{suffix}.png"),
        density=True,
    )

    ycap = hist_ylim_cap(d0, bins_d0_full, float(args.ycap_q), density=True)
    hist_1d(
        d0,
        bins=bins_d0_full,
        title=f"track d0 density ({suffix})",
        xlabel="d0",
        outpath=os.path.join(args.out_dir, f"trk_d0_density_{suffix}_linear.png"),
        density=True,
        ylim=(0.0, ycap) if ycap is not None else None,
    )
    hist_1d(
        d0,
        bins=bins_d0_full,
        title=f"track d0 density ({suffix})",
        xlabel="d0",
        outpath=os.path.join(args.out_dir, f"trk_d0_density_{suffix}.png"),
        density=True,
    )

    # --------------------------
    # 1D: zoomed plots
    # --------------------------
    if args.zoom:
        bins_pt_zoom = pt_edges_zoom if pt_edges_zoom.size else args.bins
        bins_z0_zoom = z0_edges_zoom if z0_edges_zoom.size else args.bins
        bins_d0_zoom = d0_edges_zoom if d0_edges_zoom.size else args.bins

        if np.isfinite(pt_hi_zoom) and pt_hi_zoom > 0:
            ycap = hist_ylim_cap(pt, bins_pt_zoom, float(args.ycap_q), density=False)
            hist_1d(
                pt,
                bins=bins_pt_zoom,
                title=f"track pT zoom ({suffix})",
                xlabel="pT",
                outpath=os.path.join(args.out_dir, f"trk_pt_{suffix}_zoom_linear.png"),
                density=False,
                xlim=(pt_lo_zoom, pt_hi_zoom),
                ylim=(0.0, ycap) if ycap is not None else None,
            )
            hist_1d(
                pt,
                bins=bins_pt_zoom,
                title=f"track pT zoom ({suffix})",
                xlabel="pT",
                outpath=os.path.join(args.out_dir, f"trk_pt_{suffix}_zoom.png"),
                logy=True,
                density=False,
                xlim=(pt_lo_zoom, pt_hi_zoom),
            )

            ycap = hist_ylim_cap(pt, bins_pt_zoom, float(args.ycap_q), density=True)
            hist_1d(
                pt,
                bins=bins_pt_zoom,
                title=f"track pT density zoom ({suffix})",
                xlabel="pT",
                outpath=os.path.join(args.out_dir, f"trk_pt_density_{suffix}_zoom_linear.png"),
                density=True,
                xlim=(pt_lo_zoom, pt_hi_zoom),
                ylim=(0.0, ycap) if ycap is not None else None,
            )
            hist_1d(
                pt,
                bins=bins_pt_zoom,
                title=f"track pT density zoom ({suffix})",
                xlabel="pT",
                outpath=os.path.join(args.out_dir, f"trk_pt_density_{suffix}_zoom.png"),
                logy=True,
                density=True,
                xlim=(pt_lo_zoom, pt_hi_zoom),
            )

        if np.isfinite(z0_lo_zoom) and np.isfinite(z0_hi_zoom) and z0_lo_zoom < z0_hi_zoom:
            ycap = hist_ylim_cap(z0, bins_z0_zoom, float(args.ycap_q), density=False)
            hist_1d(
                z0,
                bins=bins_z0_zoom,
                title=f"track z0 zoom ({suffix})",
                xlabel="z0",
                outpath=os.path.join(args.out_dir, f"trk_z0_{suffix}_zoom_linear.png"),
                density=False,
                xlim=(z0_lo_zoom, z0_hi_zoom),
                ylim=(0.0, ycap) if ycap is not None else None,
            )
            hist_1d(
                z0,
                bins=bins_z0_zoom,
                title=f"track z0 zoom ({suffix})",
                xlabel="z0",
                outpath=os.path.join(args.out_dir, f"trk_z0_{suffix}_zoom.png"),
                density=False,
                xlim=(z0_lo_zoom, z0_hi_zoom),
            )

            ycap = hist_ylim_cap(z0, bins_z0_zoom, float(args.ycap_q), density=True)
            hist_1d(
                z0,
                bins=bins_z0_zoom,
                title=f"track z0 density zoom ({suffix})",
                xlabel="z0",
                outpath=os.path.join(args.out_dir, f"trk_z0_density_{suffix}_zoom_linear.png"),
                density=True,
                xlim=(z0_lo_zoom, z0_hi_zoom),
                ylim=(0.0, ycap) if ycap is not None else None,
            )
            hist_1d(
                z0,
                bins=bins_z0_zoom,
                title=f"track z0 density zoom ({suffix})",
                xlabel="z0",
                outpath=os.path.join(args.out_dir, f"trk_z0_density_{suffix}_zoom.png"),
                density=True,
                xlim=(z0_lo_zoom, z0_hi_zoom),
            )

        if np.isfinite(d0_lo_zoom) and np.isfinite(d0_hi_zoom) and d0_lo_zoom < d0_hi_zoom:
            ycap = hist_ylim_cap(d0, bins_d0_zoom, float(args.ycap_q), density=False)
            hist_1d(
                d0,
                bins=bins_d0_zoom,
                title=f"track d0 zoom ({suffix})",
                xlabel="d0",
                outpath=os.path.join(args.out_dir, f"trk_d0_{suffix}_zoom_linear.png"),
                density=False,
                xlim=(d0_lo_zoom, d0_hi_zoom),
                ylim=(0.0, ycap) if ycap is not None else None,
            )
            hist_1d(
                d0,
                bins=bins_d0_zoom,
                title=f"track d0 zoom ({suffix})",
                xlabel="d0",
                outpath=os.path.join(args.out_dir, f"trk_d0_{suffix}_zoom.png"),
                density=False,
                xlim=(d0_lo_zoom, d0_hi_zoom),
            )

            ycap = hist_ylim_cap(d0, bins_d0_zoom, float(args.ycap_q), density=True)
            hist_1d(
                d0,
                bins=bins_d0_zoom,
                title=f"track d0 density zoom ({suffix})",
                xlabel="d0",
                outpath=os.path.join(args.out_dir, f"trk_d0_density_{suffix}_zoom_linear.png"),
                density=True,
                xlim=(d0_lo_zoom, d0_hi_zoom),
                ylim=(0.0, ycap) if ycap is not None else None,
            )
            hist_1d(
                d0,
                bins=bins_d0_zoom,
                title=f"track d0 density zoom ({suffix})",
                xlabel="d0",
                outpath=os.path.join(args.out_dir, f"trk_d0_density_{suffix}_zoom.png"),
                density=True,
                xlim=(d0_lo_zoom, d0_hi_zoom),
            )

    # --------------------------
    # 2D diagnostics
    # --------------------------
    hist_2d(
        z0, d0,
        bins=args.bins2d,
        title=f"z0 vs d0 ({suffix})",
        xlabel="z0",
        ylabel="d0",
        outpath=os.path.join(args.out_dir, f"trk_z0_vs_d0_{suffix}.png"),
        lognorm=args.hist2d_lognorm,
        vmax_q=float(args.hist2d_vmax_q),
    )
    hist_2d(
        pt, d0,
        bins=args.bins2d,
        title=f"pT vs d0 ({suffix})",
        xlabel="pT",
        ylabel="d0",
        outpath=os.path.join(args.out_dir, f"trk_pt_vs_d0_{suffix}.png"),
        lognorm=args.hist2d_lognorm,
        vmax_q=float(args.hist2d_vmax_q),
    )
    hist_2d(
        pt, z0,
        bins=args.bins2d,
        title=f"pT vs z0 ({suffix})",
        xlabel="pT",
        ylabel="z0",
        outpath=os.path.join(args.out_dir, f"trk_pt_vs_z0_{suffix}.png"),
        lognorm=args.hist2d_lognorm,
        vmax_q=float(args.hist2d_vmax_q),
    )

    if args.zoom:
        if np.isfinite(pt_hi_zoom) and np.isfinite(d0_lo_zoom) and np.isfinite(d0_hi_zoom) and d0_lo_zoom < d0_hi_zoom and pt_hi_zoom > 0:
            hist_2d(
                pt, d0,
                bins=args.bins2d,
                title=f"pT vs d0 zoom ({suffix})",
                xlabel="pT",
                ylabel="d0",
                outpath=os.path.join(args.out_dir, f"trk_pt_vs_d0_{suffix}_zoom.png"),
                xlim=(0.0, pt_hi_zoom),
                ylim=(d0_lo_zoom, d0_hi_zoom),
                lognorm=args.hist2d_lognorm,
                vmax_q=float(args.hist2d_vmax_q),
            )
        if np.isfinite(pt_hi_zoom) and np.isfinite(z0_lo_zoom) and np.isfinite(z0_hi_zoom) and z0_lo_zoom < z0_hi_zoom and pt_hi_zoom > 0:
            hist_2d(
                pt, z0,
                bins=args.bins2d,
                title=f"pT vs z0 zoom ({suffix})",
                xlabel="pT",
                ylabel="z0",
                outpath=os.path.join(args.out_dir, f"trk_pt_vs_z0_{suffix}_zoom.png"),
                xlim=(0.0, pt_hi_zoom),
                ylim=(z0_lo_zoom, z0_hi_zoom),
                lognorm=args.hist2d_lognorm,
                vmax_q=float(args.hist2d_vmax_q),
            )
        if np.isfinite(z0_lo_zoom) and np.isfinite(z0_hi_zoom) and np.isfinite(d0_lo_zoom) and np.isfinite(d0_hi_zoom) and z0_lo_zoom < z0_hi_zoom and d0_lo_zoom < d0_hi_zoom:
            hist_2d(
                z0, d0,
                bins=args.bins2d,
                title=f"z0 vs d0 zoom ({suffix})",
                xlabel="z0",
                ylabel="d0",
                outpath=os.path.join(args.out_dir, f"trk_z0_vs_d0_{suffix}_zoom.png"),
                xlim=(z0_lo_zoom, z0_hi_zoom),
                ylim=(d0_lo_zoom, d0_hi_zoom),
                lognorm=args.hist2d_lognorm,
                vmax_q=float(args.hist2d_vmax_q),
            )

    summary = summarise_noise(pt, z0, d0)
    out_csv = os.path.join(args.out_dir, f"trk_feature_summary_{suffix}.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    info(f"wrote summary: {out_csv}")

    if outlier_rows:
        outlier_df = pd.DataFrame(outlier_rows)
        outlier_df = outlier_df.sort_values("pT", ascending=False).reset_index(drop=True)
        out_path = os.path.join(args.out_dir, f"trk_pt_outliers_{suffix}.csv")
        outlier_df.to_csv(out_path, index=False)
        info(f"wrote pT outliers: {out_path} (n={len(outlier_df)})")
    else:
        info("no pT outliers found above threshold")

    info(f"wrote plots into: {args.out_dir}")


if __name__ == "__main__":
    main()
