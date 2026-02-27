#!/usr/bin/env python3
"""
iso_dr scan with performance-vs-cost diagnostics.

This script runs the analysis once per iso_dr value and summarises:
  - performance from the ROC curve (AUC and fake rate at a fixed target TPR)
  - a simple compute/IO cost proxy from track-occupancy-in-cone summaries

For each iso_dr point it:
  1) runs the analysis script, writing outputs into a dedicated run folder
  2) makes a per-run ROC plot (acceptance vs fake rate)
  3) records performance metrics from roc.csv (and optionally a points file)
  4) reads ntracks_*.csv files to estimate "tracks-in-cone" occupancy statistics

Why the cost proxy is useful:
  - if downstream processing is limited to tracks inside the iso cone, then the number of
    tracks inside the cone is directly proportional to compute and IO.
  - even when iterating over all tracks, cone occupancy is still a meaningful measure
    of how much activity survives the isolation requirement.

Required analysis outputs per run (in --out-dir):
  - roc.csv
  - ntracks_photons.csv
  - ntracks_jets.csv
  - ntracks.csv
Optional (used to refine fake@target_tpr if present):
  - acceptance_vs_fake_rate_points.csv

Defaults:
  - iso_dr grid: 0.05..0.50 in steps of 0.05
  - analysis script path: set via --analysis-script

Usage examples:
  python scan_iso_dr_cost.py
  python scan_iso_dr_cost.py --n-events 300 --trk-pt-min 0.75
  python scan_iso_dr_cost.py --auc-max-fpr 0.2
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_roc_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read roc.csv with columns (fpr, tpr) and return arrays sorted by fpr."""
    fpr: List[float] = []
    tpr: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fpr.append(float(row["fpr"]))
                tpr.append(float(row["tpr"]))
            except Exception:
                continue
    if not fpr:
        return np.array([], dtype=float), np.array([], dtype=float)
    f = np.asarray(fpr, dtype=float)
    t = np.asarray(tpr, dtype=float)
    o = np.argsort(f)
    return f[o], t[o]


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float = 1.0) -> float:
    """
    Compute (partial) AUC = ∫ TPR d(FPR) over FPR in [0, max_fpr].

    The input arrays are filtered to finite values. The curve is anchored at (0,0)
    if needed and is clipped/interpolated at max_fpr for partial AUC.
    """
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")

    f = np.asarray(fpr, dtype=float)
    t = np.asarray(tpr, dtype=float)

    m = np.isfinite(f) & np.isfinite(t)
    f = f[m]
    t = t[m]
    if f.size == 0:
        return float("nan")

    max_fpr = float(max_fpr)
    if max_fpr <= 0.0:
        return 0.0

    # ensure the curve starts at (0,0) for integration
    if f[0] > 0.0:
        f = np.concatenate(([0.0], f))
        t = np.concatenate(([0.0], t))

    # clip/extend to max_fpr for partial integration
    if f[-1] < max_fpr:
        f = np.concatenate((f, [max_fpr]))
        t = np.concatenate((t, [t[-1]]))
    else:
        above = f > max_fpr
        if np.any(above):
            i = int(np.argmax(above))
            f0, t0 = f[i - 1], t[i - 1]
            f1, t1 = f[i], t[i]
            if f1 == f0:
                t_at = t0
            else:
                frac = (max_fpr - f0) / (f1 - f0)
                t_at = t0 + frac * (t1 - t0)
            f = np.concatenate((f[:i], [max_fpr]))
            t = np.concatenate((t[:i], [t_at]))

    return float(np.trapezoid(t, f))


def estimate_fake_at_tpr(fpr: np.ndarray, tpr: np.ndarray, target_tpr: float = 0.99) -> float:
    """
    Conservative fake-rate estimate at fixed acceptance:
      fake@target_tpr = first FPR where TPR >= target_tpr

    If the target TPR is never reached, returns NaN.
    """
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    t = np.asarray(tpr, dtype=float)
    f = np.asarray(fpr, dtype=float)
    ok = np.isfinite(t) & np.isfinite(f)
    t = t[ok]
    f = f[ok]
    if t.size == 0:
        return float("nan")
    idx = np.where(t >= float(target_tpr))[0]
    if idx.size == 0:
        return float("nan")
    return float(f[idx[0]])


def read_points_fake(path: str, target_tpr: float) -> Optional[float]:
    """
    Try to read fake@target_tpr from acceptance_vs_fake_rate_points.csv.

    Supported formats:
      - columns: target_tpr, fpr (choose the closest target_tpr row)
      - columns: tpr, fpr (choose the first row where tpr >= target_tpr)

    Returns:
      fake rate (float) if available, otherwise None.
    """
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None

    keys = rows[0].keys()

    def fget(row, name):
        try:
            return float(row[name])
        except Exception:
            return None

    if "target_tpr" in keys and "fpr" in keys:
        best = None
        for row in rows:
            tt = fget(row, "target_tpr")
            fp = fget(row, "fpr")
            if tt is None or fp is None:
                continue
            d = abs(tt - float(target_tpr))
            if best is None or d < best[0]:
                best = (d, fp)
        return best[1] if best else None

    if "tpr" in keys and "fpr" in keys:
        for row in rows:
            tt = fget(row, "tpr")
            fp = fget(row, "fpr")
            if tt is None or fp is None:
                continue
            if tt >= float(target_tpr):
                return fp
        return None

    return None


def read_ntracks_csv(path: str) -> Dict[str, float]:
    """
    Read ntracks_*.csv (one row per object).

    Expected columns:
      ntracks_iso, ntracks_total, frac_kept

    Derived summaries:
      - mean/median/p90 of ntracks_iso (tracks inside the iso cone)
      - mean/median of frac_kept (mean of per-object ratios)
      - global_frac_kept = sum(ntracks_iso)/sum(ntracks_total) (preferred normalisation)
      - sums and mean total tracks (useful for sanity checks)

    Returns an empty dict if the file contains no valid rows.
    """
    iso: List[int] = []
    tot: List[int] = []
    frac: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ni = int(row["ntracks_iso"])
                nt = int(row["ntracks_total"])
                iso.append(ni)
                tot.append(nt)
                frac.append(float(row["frac_kept"]))
            except Exception:
                continue

    if not iso:
        return {}

    iso_a = np.asarray(iso, dtype=float)
    tot_a = np.asarray(tot, dtype=float)
    frac_a = np.asarray(frac, dtype=float)

    tot_sum = float(np.sum(tot_a))
    iso_sum = float(np.sum(iso_a))
    global_frac = iso_sum / tot_sum if tot_sum > 0 else float("nan")

    return {
        "mean_ntracks_iso": float(np.mean(iso_a)),
        "median_ntracks_iso": float(np.median(iso_a)),
        "p90_ntracks_iso": float(np.percentile(iso_a, 90)),
        "mean_frac_kept": float(np.mean(frac_a)),
        "median_frac_kept": float(np.median(frac_a)),
        "global_frac_kept": float(global_frac),
        "mean_ntracks_total": float(np.mean(tot_a)),
        "sum_ntracks_iso": float(iso_sum),
        "sum_ntracks_total": float(tot_sum),
    }


def default_iso_grid() -> List[float]:
    """Default scan grid: iso_dr = 0.05..0.50 in steps of 0.05."""
    return [round(x, 2) for x in np.arange(0.05, 0.50 + 1e-9, 0.05).tolist()]


def run_one(
    analysis_script: str,
    data_dir: str,
    n_events: int,
    out_root: str,
    iso_dr: float,
    pass_args: List[str],
) -> Tuple[int, str]:
    """
    Run the analysis script once at a fixed iso_dr and return (returncode, out_dir).

    On failure, stdout/stderr and the command are written to scan_error.txt in the run folder.
    """
    out_dir = os.path.join(out_root, f"iso_{iso_dr:.3f}".replace(".", "p"))
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        analysis_script,
        "--data-dir",
        data_dir,
        "--n-events",
        str(n_events),
        "--out-dir",
        out_dir,
        "--iso-dr",
        str(iso_dr),
    ] + pass_args

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        err_path = os.path.join(out_dir, "scan_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(
                "CMD:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr + "\n"
            )
    return p.returncode, out_dir


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, title: str, out_png: str, x_max: float = 1.0) -> None:
    """Plot a single ROC curve (acceptance vs fake rate)."""
    plt.figure()
    plt.plot(fpr, tpr, label=None)
    plt.xlabel("Fake rate (jets passing cut)")
    plt.ylabel("Acceptance (photons passing cut)")
    plt.title(title)
    if x_max is not None and np.isfinite(float(x_max)):
        plt.xlim(0, float(x_max))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis-script",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/week6/analysis_v9.py",
        help="path to the analysis script",
    )
    ap.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/1k_ev",
    )
    ap.add_argument("--n-events", type=int, default=300)
    ap.add_argument(
        "--out-root",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/week6/scan_iso_dr6",
        help="root folder for scan outputs (default: <data-dir>/../scan_iso_dr_cost)",
    )
    ap.add_argument(
        "--iso-dr",
        type=float,
        nargs="*",
        default=None,
        help="explicit iso_dr values. If omitted, uses 0.05..0.50 step 0.05.",
    )
    ap.add_argument(
        "--auc-max-fpr",
        type=float,
        default=1.0,
        help="passed through to analysis script; also used when computing AUC from roc.csv",
    )
    ap.add_argument(
        "--target-tpr",
        type=float,
        default=0.99,
        help="for fake@target_tpr",
    )
    ap.add_argument(
        "--trk-pt-min",
        type=float,
        default=0.75,
        help="optional: pass a fixed track pT min to every run",
    )
    args = ap.parse_args()

    analysis_script = os.path.abspath(args.analysis_script)
    data_dir = os.path.abspath(args.data_dir)
    out_root = os.path.abspath(args.out_root) if args.out_root is not None else os.path.abspath(
        os.path.join(data_dir, "..", "scan_iso_dr_cost")
    )
    os.makedirs(out_root, exist_ok=True)

    iso_vals = args.iso_dr if (args.iso_dr is not None and len(args.iso_dr) > 0) else default_iso_grid()

    # arguments passed through to each analysis run
    pass_args = ["--auc-max-fpr", str(float(args.auc_max_fpr))]
    if args.trk_pt_min is not None:
        pass_args += ["--trk-pt-min", str(float(args.trk_pt_min))]

    rows: List[Dict[str, object]] = []
    roc_curves: List[Tuple[float, np.ndarray, np.ndarray]] = []

    for iso_dr in iso_vals:
        rc, out_dir = run_one(
            analysis_script=analysis_script,
            data_dir=data_dir,
            n_events=int(args.n_events),
            out_root=out_root,
            iso_dr=float(iso_dr),
            pass_args=pass_args,
        )

        row: Dict[str, object] = {
            "iso_dr": float(iso_dr),
            "returncode": int(rc),
            "out_dir": out_dir,
        }

        roc_path = os.path.join(out_dir, "roc.csv")
        points_path = os.path.join(out_dir, "acceptance_vs_fake_rate_points.csv")

        # cost proxy inputs
        nph_path = os.path.join(out_dir, "ntracks_photons.csv")
        nj_path = os.path.join(out_dir, "ntracks_jets.csv")
        ncomb_path = os.path.join(out_dir, "ntracks.csv")

        # performance metrics from ROC
        if os.path.exists(roc_path):
            fpr, tpr = read_roc_csv(roc_path)
            row["auc"] = auc_from_roc(fpr, tpr, max_fpr=float(args.auc_max_fpr))
            row["fake_at_target_tpr"] = estimate_fake_at_tpr(fpr, tpr, target_tpr=float(args.target_tpr))

            # prefer precomputed points file if available
            if os.path.exists(points_path):
                fp = read_points_fake(points_path, float(args.target_tpr))
                if fp is not None:
                    row["fake_at_target_tpr"] = float(fp)

            # per-run ROC plot (kept in the run directory)
            per_png = os.path.join(out_dir, "roc_iso_overlayable.png")
            plot_roc(
                fpr,
                tpr,
                title=rf"Acceptance vs Fake rate (iso), $\Delta R < {float(iso_dr):.2f}$",
                out_png=per_png,
                x_max=float(args.auc_max_fpr),
            )

            roc_curves.append((float(iso_dr), fpr, tpr))
        else:
            row["auc"] = float("nan")
            row["fake_at_target_tpr"] = float("nan")

        # cost summaries from ntracks_*.csv
        if os.path.exists(nph_path):
            s = read_ntracks_csv(nph_path)
            for k, v in s.items():
                row[f"ph_{k}"] = v

        if os.path.exists(nj_path):
            s = read_ntracks_csv(nj_path)
            for k, v in s.items():
                row[f"jet_{k}"] = v

        if os.path.exists(ncomb_path):
            s = read_ntracks_csv(ncomb_path)
            for k, v in s.items():
                row[f"comb_{k}"] = v

        rows.append(row)

    # write summary CSV (fixed column order for easier comparison across scans)
    summary_path = os.path.join(out_root, "iso_dr_scan_summary_with_cost.csv")
    cols = [
        "iso_dr",
        "auc",
        "fake_at_target_tpr",
        "comb_mean_ntracks_iso",
        "comb_median_ntracks_iso",
        "comb_p90_ntracks_iso",
        "comb_mean_frac_kept",
        "comb_global_frac_kept",
        "comb_sum_ntracks_iso",
        "comb_sum_ntracks_total",
        "ph_mean_ntracks_iso",
        "jet_mean_ntracks_iso",
        "returncode",
        "out_dir",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # overlay ROC plot (all iso_dr values on one figure)
    if roc_curves:
        plt.figure()
        for iso_dr, fpr, tpr in sorted(roc_curves, key=lambda x: x[0]):
            plt.plot(fpr, tpr, label=f"{iso_dr:.2f}")
        plt.xlabel("Fake rate (jets passing cut)")
        plt.ylabel("Acceptance (photons passing cut)")
        plt.title("Acceptance vs Fake rate (iso): overlay of iso_dr scan")
        plt.legend(title="iso_dr", fontsize=8, title_fontsize=9)
        plt.xlim(0, float(args.auc_max_fpr))
        plt.tight_layout()
        overlay_path = os.path.join(out_root, "roc_overlay_all_iso_dr.png")
        plt.savefig(overlay_path, dpi=200)
        plt.close()

    # summary plots: metrics vs iso_dr
    def _get_arr(key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Collect a metric from rows and return (x=iso_dr, y=metric) sorted by iso_dr."""
        xs: List[float] = []
        ys: List[float] = []
        for r in rows:
            try:
                x = float(r["iso_dr"])
                y = float(r.get(key, float("nan")))
            except Exception:
                continue
            xs.append(x)
            ys.append(y)
        xarr = np.asarray(xs, dtype=float)
        yarr = np.asarray(ys, dtype=float)
        o = np.argsort(xarr)
        return xarr[o], yarr[o]

    x_auc, y_auc = _get_arr("auc")
    x_fake, y_fake = _get_arr("fake_at_target_tpr")
    x_cost, y_cost = _get_arr("comb_mean_ntracks_iso")
    x_frac_mean, y_frac_mean = _get_arr("comb_mean_frac_kept")
    x_frac_glob, y_frac_glob = _get_arr("comb_global_frac_kept")

    if x_auc.size:
        plt.figure()
        plt.plot(x_auc, y_auc, marker="o")
        plt.xlabel("iso_dr")
        plt.ylabel(f"AUC (max_fpr={float(args.auc_max_fpr):g})")
        plt.title("AUC vs iso_dr")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "summary_auc_vs_iso_dr.png"), dpi=200)
        plt.close()

    if x_fake.size:
        plt.figure()
        plt.plot(x_fake, y_fake, marker="o")
        plt.xlabel("iso_dr")
        plt.ylabel(f"fake@TPR={float(args.target_tpr):.2f}")
        plt.title("Fake rate at fixed acceptance vs iso_dr")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "summary_fake_at_tpr_vs_iso_dr.png"), dpi=200)
        plt.close()

    if x_cost.size:
        plt.figure()
        plt.plot(x_cost, y_cost, marker="o")
        plt.xlabel("iso_dr")
        plt.ylabel("mean # tracks in iso cone (combined)")
        plt.title("Cost proxy: mean tracks-in-cone vs iso_dr")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "summary_mean_tracks_in_cone_vs_iso_dr.png"), dpi=200)
        plt.close()

    # preferred normalisation: global fraction kept = sum(n_iso)/sum(n_total)
    if x_frac_glob.size:
        plt.figure()
        plt.plot(x_frac_glob, y_frac_glob, marker="o")
        plt.xlabel("iso_dr")
        plt.ylabel("global kept fraction = sum(n_iso)/sum(n_total) (combined)")
        plt.title("Fraction of tracks kept vs iso_dr (global)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "summary_global_frac_kept_vs_iso_dr.png"), dpi=200)
        plt.close()

    # alternative: mean of per-object ratios
    if x_frac_mean.size:
        plt.figure()
        plt.plot(x_frac_mean, y_frac_mean, marker="o")
        plt.xlabel("iso_dr")
        plt.ylabel("mean (ntracks_iso / ntracks_total) (combined)")
        plt.title("Normalised cone occupancy vs iso_dr (mean of ratios)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "summary_mean_frac_kept_vs_iso_dr.png"), dpi=200)
        plt.close()

    # ranking table (performance-first)
    def _key_perf(r):
        auc = r.get("auc")
        fake = r.get("fake_at_target_tpr")
        auc = float(auc) if auc is not None and np.isfinite(float(auc)) else -1e9
        fake = float(fake) if fake is not None and np.isfinite(float(fake)) else 1e9
        return (-auc, fake)

    rows_sorted = sorted(rows, key=_key_perf)

    print("\niso_dr scan results (+cost proxies):")
    print("  (sorted by AUC desc, then fake@target_tpr asc)\n")
    hdr = f"{'iso_dr':>6}  {'AUC':>10}  {'fake@tpr':>10}  {'meanNcone':>10}  {'globFrac':>9}  {'rc':>3}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_sorted:
        iso = float(r.get("iso_dr", float("nan")))
        auc = float(r.get("auc", float("nan")))
        fake = float(r.get("fake_at_target_tpr", float("nan")))
        meanN = float(r.get("comb_mean_ntracks_iso", float("nan")))
        globF = float(r.get("comb_global_frac_kept", float("nan")))
        rc = int(r.get("returncode", -1))
        print(f"{iso:6.2f}  {auc:10.6f}  {fake:10.6f}  {meanN:10.3f}  {globF:9.4f}  {rc:3d}")

    # ranking table (cost-aware)
    def _key_cost(r):
        fake = r.get("fake_at_target_tpr")
        meanN = r.get("comb_mean_ntracks_iso")
        auc = r.get("auc")
        fake = float(fake) if fake is not None and np.isfinite(float(fake)) else 1e9
        meanN = float(meanN) if meanN is not None and np.isfinite(float(meanN)) else 1e9
        auc = float(auc) if auc is not None and np.isfinite(float(auc)) else -1e9
        return (fake, meanN, -auc)

    rows_cost = sorted(rows, key=_key_cost)

    print("\niso_dr scan results (sorted by fake@tpr, then cost, then AUC):\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_cost:
        iso = float(r.get("iso_dr", float("nan")))
        auc = float(r.get("auc", float("nan")))
        fake = float(r.get("fake_at_target_tpr", float("nan")))
        meanN = float(r.get("comb_mean_ntracks_iso", float("nan")))
        globF = float(r.get("comb_global_frac_kept", float("nan")))
        rc = int(r.get("returncode", -1))
        print(f"{iso:6.2f}  {auc:10.6f}  {fake:10.6f}  {meanN:10.3f}  {globF:9.4f}  {rc:3d}")

    # final file list (kept as prints for quick scan checks)
    print(f"\nwrote: {summary_path}")
    if roc_curves:
        print(f"wrote: {os.path.join(out_root, 'roc_overlay_all_iso_dr.png')}")
    print(f"wrote: {os.path.join(out_root, 'summary_auc_vs_iso_dr.png')}")
    print(f"wrote: {os.path.join(out_root, 'summary_fake_at_tpr_vs_iso_dr.png')}")
    print(f"wrote: {os.path.join(out_root, 'summary_mean_tracks_in_cone_vs_iso_dr.png')}")
    print(f"wrote: {os.path.join(out_root, 'summary_global_frac_kept_vs_iso_dr.png')}")
    print(f"wrote: {os.path.join(out_root, 'summary_mean_frac_kept_vs_iso_dr.png')}")


if __name__ == "__main__":
    main()