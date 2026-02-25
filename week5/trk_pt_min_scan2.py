#!/usr/bin/env python3
"""
Super-fine scan of trk_pt_min for the HL-LHC diphoton analysis (analysis_v7.py).

Key change vs stdout-parsing scanners:
  - DO NOT parse AUC / FIXED-TPR from stdout (breaks when --quiet is used).
  - Instead read ROC points from:
      * acceptance_vs_fake_rate_points.csv (preferred)
      * or roc.csv (fallback)
    and compute:
      * AUC (partial) up to AUC_MAX_FPR
      * fake@TARGET_TPR via interpolation FPR(TPR=TARGET_TPR)

Ranking:
  - lowest fake@TARGET_TPR
  - then highest AUC (partial)
  - then highest kept_frac_pt

Outputs (under OUT_BASE):
  - one folder per scan point with run.log
  - trk_pt_min_scan_summary.csv
"""

from __future__ import annotations

import csv
import os
import subprocess
from math import isnan
from typing import List, Tuple


# ---- USER SETTINGS ----
PYTHON = r"E:\Anaconda\python.exe"
ANALYSIS = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\week5\analysis_v7.py"
DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"
OUT_BASE = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_trk_pt_min2"

# fixed analysis knobs (keep constant so you isolate pT-min effects)
N_EVENTS = 1000
FIXED_ISO_DR = 0.20
FIXED_DR_TRACK = 0.10
FIXED_DR_OVERLAP = 0.20

# ROC/AUC settings
AUC_MAX_FPR = 0.05
TARGET_TPR = 0.99

# scan definition
FINE_START = 0.5
FINE_STOP = 1.5
FINE_STEP = 0.05
COARSE_POINTS = [0.0, 0.2, 0.4, 0.5, 1.4, 1.5, 1.7, 2.0]

# run behaviour
USE_QUIET = False
# -----------------------


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt(x: float, ndp: int = 6) -> str:
    if x is None or (isinstance(x, float) and isnan(x)):
        return "nan"
    return f"{x:.{ndp}f}"


def frange_inclusive(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    k = 0
    while True:
        v = start + k * step
        if v > stop + 1e-12:
            break
        vals.append(round(v, 10))
        k += 1
    return vals


def read_single_row_csv(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[0] if rows else {}
    except Exception:
        return {}


def read_roc_points(out_dir: str) -> Tuple[List[float], List[float]]:
    """
    Returns (fpr_list, tpr_list) from either:
      - acceptance_vs_fake_rate_points.csv (preferred)
      - roc.csv (fallback)
    """
    candidates = [
        os.path.join(out_dir, "acceptance_vs_fake_rate_points.csv"),
        os.path.join(out_dir, "roc.csv"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            fprs: List[float] = []
            tprs: List[float] = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                r = csv.DictReader(f)
                for row in r:
                    if "fpr" not in row or "tpr" not in row:
                        continue
                    fpr = safe_float(row["fpr"])
                    tpr = safe_float(row["tpr"])
                    if not isnan(fpr) and not isnan(tpr):
                        fprs.append(fpr)
                        tprs.append(tpr)
            if len(fprs) > 1:
                return fprs, tprs
        except Exception:
            pass

    return [], []


def sort_by_fpr(fpr: List[float], tpr: List[float]) -> Tuple[List[float], List[float]]:
    order = sorted(range(len(fpr)), key=lambda i: fpr[i])
    return [fpr[i] for i in order], [tpr[i] for i in order]


def auc_partial(fpr: List[float], tpr: List[float], max_fpr: float) -> float:
    """
    Trapezoidal integral of TPR d(FPR) over FPR in [0, max_fpr].
    Adds (0,0) if needed and interpolates endpoint at max_fpr if needed.
    """
    if len(fpr) < 2:
        return float("nan")

    f, t = sort_by_fpr(fpr, tpr)

    # ensure start at f=0
    if f[0] > 0.0:
        f = [0.0] + f
        t = [0.0] + t

    # clip / extend to max_fpr
    if f[-1] < max_fpr:
        f = f + [max_fpr]
        t = t + [t[-1]]
    else:
        # truncate and interpolate at max_fpr
        cut_idx = 0
        while cut_idx < len(f) and f[cut_idx] <= max_fpr:
            cut_idx += 1
        # now cut_idx is first index with f > max_fpr (or len)
        if cut_idx < len(f):
            f0, t0 = f[cut_idx - 1], t[cut_idx - 1]
            f1, t1 = f[cut_idx], t[cut_idx]
            if f1 == f0:
                t_at = t0
            else:
                frac = (max_fpr - f0) / (f1 - f0)
                t_at = t0 + frac * (t1 - t0)
            f = f[:cut_idx]  # includes last <= max_fpr
            t = t[:cut_idx]
            f[-1] = max_fpr
            t[-1] = t_at

    # trapezoid
    area = 0.0
    for i in range(1, len(f)):
        df = f[i] - f[i - 1]
        area += 0.5 * (t[i] + t[i - 1]) * df
    return float(area)


def fpr_at_target_tpr(fpr: List[float], tpr: List[float], target_tpr: float) -> float:
    """
    Compute fake@target_tpr by interpolating FPR as a function of TPR.
    Steps:
      - sort by TPR
      - find first segment that crosses target_tpr
      - linear interpolate in that segment
    """
    if len(fpr) < 2:
        return float("nan")

    order = sorted(range(len(tpr)), key=lambda i: tpr[i])
    T = [tpr[i] for i in order]
    F = [fpr[i] for i in order]

    # if target below min or above max, clamp
    if target_tpr <= T[0]:
        return float(F[0])
    if target_tpr >= T[-1]:
        return float(F[-1])

    for i in range(1, len(T)):
        if T[i] >= target_tpr:
            t0, f0 = T[i - 1], F[i - 1]
            t1, f1 = T[i], F[i]
            if t1 == t0:
                return float(f1)
            frac = (target_tpr - t0) / (t1 - t0)
            return float(f0 + frac * (f1 - f0))

    return float("nan")


def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    pts = sorted(set([float(x) for x in (COARSE_POINTS + frange_inclusive(FINE_START, FINE_STOP, FINE_STEP))]))

    print(f"[scan] OUT_BASE: {OUT_BASE}")
    print(f"[scan] points={len(pts)}  fine=[{FINE_START},{FINE_STOP}] step={FINE_STEP}  auc_max_fpr={AUC_MAX_FPR}")
    print(f"[scan] metrics computed from ROC CSVs (works with --quiet={USE_QUIET})")

    rows: List[dict] = []

    for i, ptmin in enumerate(pts, 1):
        tag = f"ptmin_{ptmin:.3f}".replace(".", "p")
        out_dir = os.path.join(OUT_BASE, tag)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")

        cmd = [
            PYTHON, ANALYSIS,
            "--data-dir", DATA_DIR,
            "--n-events", str(N_EVENTS),
            "--out-dir", out_dir,
            "--iso-dr", str(FIXED_ISO_DR),
            "--dr-track", str(FIXED_DR_TRACK),
            "--dr-overlap", str(FIXED_DR_OVERLAP),
            "--trk-pt-min", str(ptmin),
            "--auc-max-fpr", str(AUC_MAX_FPR),
            "--fast-scan",
        ]
        if USE_QUIET:
            cmd.append("--quiet")

        print(f"[scan] ({i}/{len(pts)}) trk_pt_min={ptmin:g}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(combined)

        auc = float("nan")
        fake99 = float("nan")
        kept_frac_pt = float("nan")
        kept_frac_dr = float("nan")

        if proc.returncode == 0:
            # read diagnostics
            tr = read_single_row_csv(os.path.join(out_dir, "track_reduction.csv"))
            if "kept_fraction" in tr:
                kept_frac_dr = safe_float(tr["kept_fraction"])

            ptr = read_single_row_csv(os.path.join(out_dir, "track_ptmin_reduction.csv"))
            if "kept_fraction_pt" in ptr:
                kept_frac_pt = safe_float(ptr["kept_fraction_pt"])

            # read ROC and compute metrics
            fpr, tpr = read_roc_points(out_dir)
            if len(fpr) >= 2:
                auc = auc_partial(fpr, tpr, AUC_MAX_FPR)
                fake99 = fpr_at_target_tpr(fpr, tpr, TARGET_TPR)

            print(
                f"    rc=0  auc={fmt(auc,6)}  fake@{TARGET_TPR:.2f}={fmt(fake99,3)}  "
                f"kept_frac_pt={fmt(kept_frac_pt,3)}  kept_frac_dr={fmt(kept_frac_dr,3)}"
            )
        else:
            snippet = (proc.stderr or proc.stdout or "").strip().splitlines()[:12]
            print(f"    rc={proc.returncode} (analysis failed)")
            for line in snippet:
                print("    " + line)

        rows.append({
            "trk_pt_min": ptmin,
            "returncode": proc.returncode,
            "n_events": N_EVENTS,
            "iso_dr": FIXED_ISO_DR,
            "dr_track": FIXED_DR_TRACK,
            "dr_overlap": FIXED_DR_OVERLAP,
            "auc_max_fpr": AUC_MAX_FPR,
            "auc": auc,
            "fake_at_TPR_0p99": fake99,
            "kept_frac_pt": kept_frac_pt,
            "kept_frac_dr": kept_frac_dr,
            "out_dir": out_dir,
            "log": log_path,
        })

    # write summary CSV
    csv_path = os.path.join(OUT_BASE, "trk_pt_min_scan_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[scan] wrote: {csv_path}")

    # ranking
    valid = [r for r in rows if r["returncode"] == 0]
    valid.sort(
        key=lambda r: (
            r["fake_at_TPR_0p99"] if not isnan(r["fake_at_TPR_0p99"]) else 1e9,
            -(r["auc"] if not isnan(r["auc"]) else -1e9),
            -(r["kept_frac_pt"] if not isnan(r["kept_frac_pt"]) else -1e9),
        )
    )

    print("[scan] top candidates (lowest fake@0.99, then highest AUC, then highest kept_frac_pt):")
    for r in valid[:20]:
        print(
            f"  trk_pt_min={r['trk_pt_min']:.2f}  "
            f"fake@0.99={fmt(r['fake_at_TPR_0p99'],3)}  "
            f"auc={fmt(r['auc'],6)} (max_fpr={r['auc_max_fpr']})  "
            f"kept_frac_pt={fmt(r['kept_frac_pt'],3)}  "
            f"kept_frac_dr={fmt(r['kept_frac_dr'],3)}"
        )

    print("\n[scan] done.")


if __name__ == "__main__":
    main()
