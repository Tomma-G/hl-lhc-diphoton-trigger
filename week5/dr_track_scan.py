#!/usr/bin/env python3
"""
Super-fine scan of dr_track_keep (track conditioning radius around photons) for analysis_v7.py.

Goal:
  - only tracks within dr_track_keep of a photon contribute to isolation / track-based features
  - measure iso ROC sensitivity: partial AUC (FPR<=AUC_MAX_FPR) + fake@TPR

Outputs:
  - dr_track_keep_scan_summary.csv in OUT_BASE
  - per-point out_dir with run.log + the ROC CSV from analysis

Ranking:
  - lowest fake@TPR, then highest AUC, then highest kept_frac_pt (pt-min reduction)

NOTE:
  - This requires analysis_v7.py to accept KEEP_FLAG (default set below).
  - Metrics are computed from acceptance_vs_fake_rate_points.csv to avoid relying on stdout parsing.
"""

from __future__ import annotations

import csv
import os
import subprocess
from math import isnan
from typing import Dict, List, Tuple

# ---- USER SETTINGS ----
PYTHON = r"E:\Anaconda\python.exe"
ANALYSIS = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\week5\analysis_v7.py"
DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"
OUT_BASE = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_dr_track_keep_superfine"

# IMPORTANT: set this to the real CLI flag your analysis uses
KEEP_FLAG = "--dr-track-keep"   # change if needed

# freeze tuned knobs
N_EVENTS = 1000
FIXED_ISO_DR = 0.20
FIXED_DR_TRACK = 0.10
FIXED_DR_OVERLAP = 0.20
FIXED_TRK_PT_MIN = 1.075  # update to your chosen value

# metrics
AUC_MAX_FPR = 0.05
TARGET_TPR = 0.99

# scan range for dr_track_keep
DR_KEEP_MIN = 0.00001
DR_KEEP_MAX = 0.006
DR_KEEP_STEP = 0.0001

# optional anchors
EXTRA_POINTS = [0.03, 0.05, 0.10]

PASS_QUIET = False
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


def arange_inclusive(lo: float, hi: float, step: float) -> List[float]:
    vals = []
    k = 0
    while True:
        v = lo + k * step
        if v > hi + 1e-12:
            break
        vals.append(round(v, 6))
        k += 1
    return vals


def read_first_row_csv(path: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out: Dict[str, float] = {}
                for k, v in row.items():
                    out[k] = safe_float(v) if v is not None else float("nan")
                return out
    except Exception:
        return {}
    return {}


def read_roc_csv(path: str) -> Tuple[List[float], List[float]]:
    fpr: List[float] = []
    tpr: List[float] = []
    if not path or not os.path.exists(path):
        return fpr, tpr
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fx = safe_float(r.get("fpr", "nan"))
                ty = safe_float(r.get("tpr", "nan"))
                if not isnan(fx) and not isnan(ty):
                    fpr.append(fx)
                    tpr.append(ty)
    except Exception:
        return [], []
    return fpr, tpr


def sort_and_dedup_roc(fpr: List[float], tpr: List[float]) -> Tuple[List[float], List[float]]:
    pts = sorted(zip(fpr, tpr), key=lambda x: x[0])
    out_f: List[float] = []
    out_t: List[float] = []
    last_f = None
    for f, t in pts:
        if last_f is None or abs(f - last_f) > 1e-15:
            out_f.append(f)
            out_t.append(t)
            last_f = f
        else:
            out_t[-1] = max(out_t[-1], t)
    return out_f, out_t


def trapezoid_auc(f: List[float], t: List[float], max_fpr: float) -> float:
    if len(f) < 2:
        return float("nan")
    max_fpr = float(max_fpr)
    if max_fpr <= 0.0:
        return 0.0

    if f[0] > 0.0:
        f = [0.0] + f
        t = [0.0] + t

    if f[-1] < max_fpr:
        f = f + [max_fpr]
        t = t + [t[-1]]
    else:
        idx = 0
        while idx < len(f) and f[idx] < max_fpr:
            idx += 1
        if idx == 0:
            return 0.0
        if idx < len(f) and f[idx] == max_fpr:
            f = f[: idx + 1]
            t = t[: idx + 1]
        else:
            f0, t0 = f[idx - 1], t[idx - 1]
            f1, t1 = f[idx], t[idx]
            frac = 0.0 if f1 == f0 else (max_fpr - f0) / (f1 - f0)
            t_at = t0 + frac * (t1 - t0)
            f = f[:idx] + [max_fpr]
            t = t[:idx] + [t_at]

    area = 0.0
    for i in range(1, len(f)):
        df = f[i] - f[i - 1]
        area += 0.5 * (t[i] + t[i - 1]) * df
    return float(area)


def fake_at_fixed_tpr(f: List[float], t: List[float], target_tpr: float) -> float:
    if len(f) == 0:
        return float("nan")
    target = float(target_tpr)
    if max(t) < target:
        return float("nan")

    if t[0] >= target:
        return float(f[0])

    for i in range(1, len(f)):
        t0, t1 = t[i - 1], t[i]
        if (t0 < target) and (t1 >= target):
            f0, f1 = f[i - 1], f[i]
            if t1 == t0:
                return float(f1)
            frac = (target - t0) / (t1 - t0)
            return float(f0 + frac * (f1 - f0))
    return float("nan")


def main() -> None:
    os.makedirs(OUT_BASE, exist_ok=True)

    scan_points = arange_inclusive(DR_KEEP_MIN, DR_KEEP_MAX, DR_KEEP_STEP)
    for x in EXTRA_POINTS:
        scan_points.append(float(x))
    scan_points = sorted(set(round(v, 6) for v in scan_points))

    print(f"[scan] OUT_BASE: {OUT_BASE}")
    print(f"[scan] points={len(scan_points)}  dr_track_keep=[{DR_KEEP_MIN},{DR_KEEP_MAX}] step={DR_KEEP_STEP}  auc_max_fpr={AUC_MAX_FPR}")
    print("[scan] metrics computed from ROC CSVs")

    rows = []
    key_fake = f"fake_at_TPR_{str(TARGET_TPR).replace('.', 'p')}"

    for i, dr_keep in enumerate(scan_points, start=1):
        tag = f"drkeep_{dr_keep:.4f}".replace(".", "p")
        out_dir = os.path.join(OUT_BASE, tag)
        os.makedirs(out_dir, exist_ok=True)

        log_path = os.path.join(out_dir, "run.log")

        cmd = [
            PYTHON,
            ANALYSIS,
            "--data-dir", DATA_DIR,
            "--n-events", str(N_EVENTS),
            "--out-dir", out_dir,
            "--iso-dr", str(FIXED_ISO_DR),
            "--dr-track", str(FIXED_DR_TRACK),
            "--dr-overlap", str(FIXED_DR_OVERLAP),
            "--trk-pt-min", str(FIXED_TRK_PT_MIN),
            "--auc-max-fpr", str(AUC_MAX_FPR),
            "--fast-scan",
            KEEP_FLAG, str(dr_keep),
        ]
        if PASS_QUIET:
            cmd.append("--quiet")

        print(f"[scan] ({i}/{len(scan_points)}) dr_track_keep={dr_keep:g}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(combined)

        roc_path = os.path.join(out_dir, "acceptance_vs_fake_rate_points.csv")
        fpr, tpr = read_roc_csv(roc_path)
        fpr, tpr = sort_and_dedup_roc(fpr, tpr)

        auc = trapezoid_auc(fpr, tpr, AUC_MAX_FPR) if proc.returncode == 0 else float("nan")
        fake_t = fake_at_fixed_tpr(fpr, tpr, TARGET_TPR) if proc.returncode == 0 else float("nan")

        pt_red = read_first_row_csv(os.path.join(out_dir, "track_ptmin_reduction.csv"))
        dr_red = read_first_row_csv(os.path.join(out_dir, "track_reduction.csv"))
        kept_frac_pt = pt_red.get("kept_fraction_pt", float("nan"))
        kept_frac_dr = dr_red.get("kept_fraction", float("nan"))

        print(
            f"    rc={proc.returncode}  auc={fmt(auc, 6)}  fake@{TARGET_TPR:.2f}={fmt(fake_t, 3)}  "
            f"kept_frac_pt={fmt(kept_frac_pt, 3)}  kept_frac_dr={fmt(kept_frac_dr, 3)}"
        )

        rows.append({
            "dr_track_keep": dr_keep,
            "returncode": proc.returncode,
            "n_events": N_EVENTS,
            "iso_dr": FIXED_ISO_DR,
            "dr_track": FIXED_DR_TRACK,
            "dr_overlap": FIXED_DR_OVERLAP,
            "trk_pt_min": FIXED_TRK_PT_MIN,
            "auc_max_fpr": AUC_MAX_FPR,
            "auc": auc,
            key_fake: fake_t,
            "kept_fraction_pt": kept_frac_pt,
            "kept_fraction_drkeep": kept_frac_dr,
            "out_dir": out_dir,
            "log": log_path,
            "roc_csv": roc_path,
        })

    csv_path = os.path.join(OUT_BASE, "dr_track_keep_scan_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[scan] wrote: {csv_path}")

    valid = [r for r in rows if r["returncode"] == 0 and not isnan(r.get(key_fake, float("nan")))]
    valid.sort(
        key=lambda r: (
            r[key_fake],
            -(r["auc"] if not isnan(r["auc"]) else -1e9),
            -(r["kept_fraction_pt"] if not isnan(r["kept_fraction_pt"]) else -1e9),
        )
    )

    print("[scan] top candidates (lowest fake@TPR, then highest AUC, then highest kept_frac_pt):")
    for r in valid[:30]:
        print(
            f"  dr_track_keep={r['dr_track_keep']:.4f}  "
            f"fake@{TARGET_TPR:.2f}={fmt(r[key_fake], 3)}  "
            f"auc={fmt(r['auc'], 6)} (max_fpr={r['auc_max_fpr']})  "
            f"kept_frac_pt={fmt(r['kept_fraction_pt'], 3)}  "
            f"out={r['out_dir']}"
        )

    print("\n[scan] done.")


if __name__ == "__main__":
    main()
