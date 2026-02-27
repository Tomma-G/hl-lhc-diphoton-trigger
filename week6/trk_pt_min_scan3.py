#!/usr/bin/env python3
"""
Scan over --trk-pt-min for the analysis script.

For each chosen track pT threshold (trk_pt_min), this script:
  - runs the analysis script once
  - reads the run outputs and extracts summary metrics
  - saves per-point and overlaid ROC plots
  - writes a single summary CSV and prints a compact ranking table

Metrics extracted per run:
  - AUC from roc.csv (optionally partial AUC using --auc-max-fpr)
  - fake@target_tpr from acceptance_vs_fake_rate_points.csv if present,
    otherwise estimated from roc.csv
  - track kept fraction from track_ptmin_reduction.csv

Expected analysis outputs in each run directory:
  - roc.csv
  - track_ptmin_reduction.csv
Optional (preferred for fake@target_tpr):
  - acceptance_vs_fake_rate_points.csv

Usage examples:
  python scan_ptmin.py --analysis-script analysis_v9.py --data-dir "C:/.../1k_ev" --n-events 300

  python scan_ptmin.py --analysis-script analysis_v9.py --data-dir "C:/.../1k_ev" \
      --n-events 300 --ptmin 0.5 0.75 1.0 1.25 1.5 --auc-max-fpr 0.2

Notes:
  - If acceptance_vs_fake_rate_points.csv is not produced, fake@target_tpr is estimated
    conservatively from roc.csv (first FPR where TPR >= target).
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

    The curve is anchored at (0,0) if needed and clipped/interpolated at max_fpr
    for partial AUC.
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

    # ensure starting at (0,0) for integration
    if f[0] > 0.0:
        f = np.concatenate(([0.0], f))
        t = np.concatenate(([0.0], t))

    # clip/extend to max_fpr
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
    Conservative estimate: take the first FPR where TPR >= target_tpr.
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


def read_track_ptmin_reduction(path: str) -> Dict[str, float]:
    """Read track_ptmin_reduction.csv and coerce numeric fields to floats."""
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        row = next(r, None)
    if not row:
        return {}
    out: Dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, title: str, out_png: str) -> None:
    """Plot a single ROC curve (acceptance vs fake rate)."""
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("Fake rate (jets passing cut)")
    plt.ylabel("Acceptance (photons passing cut)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def read_points_fake99(path: str) -> Optional[float]:
    """
    Read fake@0.99 from acceptance_vs_fake_rate_points.csv if present.

    Supported formats:
      - columns: target_tpr, fpr (use the row with target_tpr closest to 0.99)
      - columns: tpr, fpr (use the first row where tpr >= 0.99)

    Returns:
      fpr value if available, otherwise None.
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
            d = abs(tt - 0.99)
            if best is None or d < best[0]:
                best = (d, fp)
        return best[1] if best else None

    if "tpr" in keys and "fpr" in keys:
        for row in rows:
            tt = fget(row, "tpr")
            fp = fget(row, "fpr")
            if tt is None or fp is None:
                continue
            if tt >= 0.99:
                return fp
        return None

    return None


def run_one(
    analysis_script: str,
    data_dir: str,
    n_events: int,
    out_root: str,
    ptmin: float,
    pass_args: List[str],
) -> Tuple[int, str]:
    """
    Run the analysis script once at a fixed trk_pt_min and return (returncode, out_dir).

    On failure, stdout/stderr and the command are written to scan_error.txt in the run folder.
    """
    out_dir = os.path.join(out_root, f"ptmin_{ptmin:.3f}".replace(".", "p"))
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
        "--trk-pt-min",
        str(ptmin),
    ] + pass_args

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        err_path = os.path.join(out_dir, "scan_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("CMD:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr + "\n")
    return p.returncode, out_dir


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
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/week6/scan_trk_pt_min3",
        help="root folder for scan outputs (default: <data-dir>/../scan_ptmin)",
    )
    ap.add_argument(
        "--ptmin",
        type=float,
        nargs="*",
        default=None,
        help="explicit ptmin values in GeV. If omitted, uses a default grid.",
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
    args = ap.parse_args()

    analysis_script = os.path.abspath(args.analysis_script)
    data_dir = os.path.abspath(args.data_dir)
    out_root = (
        os.path.abspath(args.out_root)
        if args.out_root is not None
        else os.path.abspath(os.path.join(data_dir, "..", "scan_ptmin"))
    )
    os.makedirs(out_root, exist_ok=True)

    ptmins = (
        args.ptmin
        if (args.ptmin is not None and len(args.ptmin) > 0)
        else [0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    )

    # arguments passed through to each analysis run
    pass_args = ["--auc-max-fpr", str(float(args.auc_max_fpr))]

    rows: List[Dict[str, object]] = []
    roc_curves: List[Tuple[float, np.ndarray, np.ndarray]] = []

    for ptmin in ptmins:
        rc, out_dir = run_one(
            analysis_script=analysis_script,
            data_dir=data_dir,
            n_events=int(args.n_events),
            out_root=out_root,
            ptmin=float(ptmin),
            pass_args=pass_args,
        )

        row: Dict[str, object] = {
            "trk_pt_min": float(ptmin),
            "returncode": int(rc),
            "out_dir": out_dir,
        }

        roc_path = os.path.join(out_dir, "roc.csv")
        ptred_path = os.path.join(out_dir, "track_ptmin_reduction.csv")
        points_path = os.path.join(out_dir, "acceptance_vs_fake_rate_points.csv")

        if os.path.exists(roc_path):
            fpr, tpr = read_roc_csv(roc_path)
            roc_curves.append((float(ptmin), fpr, tpr))

            # per-point ROC plot
            plot_roc(
                fpr,
                tpr,
                title=f"Acceptance vs Fake rate (iso), ptmin={ptmin:.2f} GeV",
                out_png=os.path.join(out_dir, "roc_ptmin.png"),
            )

            row["auc"] = auc_from_roc(fpr, tpr, max_fpr=float(args.auc_max_fpr))
            row["fake_at_target_tpr"] = estimate_fake_at_tpr(fpr, tpr, target_tpr=float(args.target_tpr))
        else:
            row["auc"] = float("nan")
            row["fake_at_target_tpr"] = float("nan")

        # if points file exists, prefer that value for fake@target_tpr
        if os.path.exists(points_path):
            fp = read_points_fake99(points_path)
            if fp is not None:
                row["fake_at_target_tpr"] = float(fp)

        # track reduction summary from the analysis output
        if os.path.exists(ptred_path):
            d = read_track_ptmin_reduction(ptred_path)
            for k in ["tracks_before_pt", "tracks_after_pt", "kept_fraction_pt"]:
                if k in d:
                    row[k] = d[k]

        rows.append(row)

    # ROC overlay plot across all ptmin values
    if roc_curves:
        plt.figure()
        for pt, fpr, tpr in sorted(roc_curves, key=lambda x: x[0]):
            plt.plot(fpr, tpr, label=f"{pt:.2f}")
        plt.xlabel("Fake rate (jets passing cut)")
        plt.ylabel("Acceptance (photons passing cut)")
        plt.title("ROC overlay: trk_pt_min scan")
        plt.legend(title="trk_pt_min [GeV]", fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "roc_overlay_all_ptmin.png"), dpi=200)
        plt.close()

    # write summary CSV
    summary_path = os.path.join(out_root, "ptmin_scan_summary.csv")
    cols = [
        "trk_pt_min",
        "auc",
        "fake_at_target_tpr",
        "kept_fraction_pt",
        "tracks_before_pt",
        "tracks_after_pt",
        "returncode",
        "out_dir",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # print compact table (sorted by auc desc, then fake asc)
    def _key(r):
        auc = r.get("auc")
        fake = r.get("fake_at_target_tpr")
        auc = float(auc) if auc is not None and np.isfinite(auc) else -1e9
        fake = float(fake) if fake is not None and np.isfinite(fake) else 1e9
        return (-auc, fake)

    rows_sorted = sorted(rows, key=_key)

    print("\nptmin scan results:")
    print("  (sorted by AUC desc, then fake@target_tpr asc)\n")
    hdr = f"{'ptmin':>6}  {'AUC':>10}  {'fake@tpr':>10}  {'keep(frac)':>10}  {'rc':>3}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_sorted:
        pt = r.get("trk_pt_min", float("nan"))
        auc = r.get("auc", float("nan"))
        fake = r.get("fake_at_target_tpr", float("nan"))
        kf = r.get("kept_fraction_pt", float("nan"))
        rc = r.get("returncode", -1)
        print(f"{pt:6.2f}  {auc:10.6f}  {fake:10.6f}  {kf:10.4f}  {int(rc):3d}")

    print(f"\nwrote: {summary_path}")


if __name__ == "__main__":
    main()