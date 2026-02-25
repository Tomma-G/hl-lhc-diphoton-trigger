#!/usr/bin/env python3
r"""
Scan iso_dr by running the analysis script repeatedly and extracting:

- AUC from the "[info] AUC (max_fpr=...)" line
- fake@TPR=0.99 from the "FIXED TPR=0.99 ..." line

Additionally (if produced by analysis outputs):
- overlays ROC curves from roc*.csv in each per-point out_dir
- overlays Ntracks distributions from ntracks/tracks csv files in each per-point out_dir
  (and saves histograms, a CDF overlay, and a boxplot)

Writes per-point logs and a summary CSV.

Notes:
- This scan script sets --out-dir so each iso_dr point writes outputs into its own folder.
- It also enables --fast-scan for speed (analysis still writes ROC points + AUC in fast-scan mode).
- If you want partial AUC (trigger-relevant region), set AUC_MAX_FPR to e.g. 0.01 or 0.05.
"""

import csv
import glob
import os
import re
import subprocess
from math import isnan

import numpy as np
import matplotlib.pyplot as plt

ISO_DRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

PYTHON = r"E:\Anaconda\python.exe"
ANALYSIS = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\week5\analysis_v8.py"
DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"
OUT_BASE = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_iso_dr"

# Choose full AUC (=1.0) or partial AUC (e.g. 0.01 for 1% fake-rate region)
AUC_MAX_FPR = 1.0

os.makedirs(OUT_BASE, exist_ok=True)

RE_AUC = re.compile(
    r"\bAUC\s*\(max_fpr=(?P<maxfpr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\s*=\s*(?P<auc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

RE_FIXED = re.compile(
    r"FIXED TPR=0\.99:\s*cut=[^,]+,\s*acc=(?P<acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*fake=(?P<fake>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _find_first(patterns, base_dir, recursive=False):
    """
    Finds first matching file among patterns.
    If recursive=True, searches base_dir/**/pattern.
    """
    for pat in patterns:
        if recursive:
            hits = sorted(glob.glob(os.path.join(base_dir, "**", pat), recursive=True))
        else:
            hits = sorted(glob.glob(os.path.join(base_dir, pat)))
        if hits:
            return hits[0]
    return None

def _read_roc_csv(path):
    """
    Tries to read a ROC CSV with headers like:
      fpr,tpr   or   FPR,TPR   or similar.
    Returns (fpr, tpr) numpy arrays, or (None, None) if not readable.
    """
    if not path or not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, None

        fields = {h.strip().lower(): h for h in reader.fieldnames}

        fpr_key = None
        tpr_key = None
        for cand in ["fpr", "false_positive_rate", "fp_rate"]:
            if cand in fields:
                fpr_key = fields[cand]
                break
        for cand in ["tpr", "true_positive_rate", "tp_rate", "eff", "efficiency"]:
            if cand in fields:
                tpr_key = fields[cand]
                break

        if fpr_key is None or tpr_key is None:
            return None, None

        fprs, tprs = [], []
        for row in reader:
            xf = _safe_float(row.get(fpr_key, ""))
            xt = _safe_float(row.get(tpr_key, ""))
            if xf is None or xt is None:
                continue
            fprs.append(xf)
            tprs.append(xt)

    if not fprs:
        return None, None

    fpr = np.array(fprs, dtype=float)
    tpr = np.array(tprs, dtype=float)

    order = np.argsort(fpr)
    return fpr[order], tpr[order]

def _read_ntracks_csv(path):
    """
    Reads your new ntracks*.csv written by analysis_v6.py with headers:
      ntracks_iso, ntracks_total, frac_kept

    Returns:
      (ntracks_iso_array, frac_kept_array)  OR  (None, None)
    """
    if not path or not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, None

        fields = {h.strip().lower(): h for h in reader.fieldnames}

        # prefer the new names
        iso_key = fields.get("ntracks_iso")
        tot_key = fields.get("ntracks_total")
        frac_key = fields.get("frac_kept")

        # fallback (if you ever point it at old files)
        if iso_key is None:
            for cand in ["ntracks", "n_tracks", "ntrk", "tracks", "n_trk", "ntrack"]:
                if cand in fields:
                    iso_key = fields[cand]
                    break

        if iso_key is None:
            return None, None

        iso_vals = []
        frac_vals = []

        for row in reader:
            a = _safe_float(row.get(iso_key, ""))
            if a is None:
                continue

            # if frac_kept present, use it; otherwise compute from total if available
            fval = _safe_float(row.get(frac_key, "")) if frac_key else None
            if fval is None and tot_key is not None:
                b = _safe_float(row.get(tot_key, ""))
                if b is not None and b > 0:
                    fval = a / b

            iso_vals.append(a)
            if fval is not None:
                frac_vals.append(fval)

    if not iso_vals:
        return None, None

    iso_arr = np.array(iso_vals, dtype=float)
    frac_arr = np.array(frac_vals, dtype=float) if frac_vals else None
    return iso_arr, frac_arr


rows = []

roc_curves = {}     # iso -> (fpr, tpr, file)
ntracks_dists = {}     # iso -> (ntracks_iso_arr, file)
frac_kept_dists = {}   # iso -> (frac_kept_arr, file)

for iso in ISO_DRS:
    tag = f"iso_dr_{iso:.2f}".replace(".", "p")
    out_dir = os.path.join(OUT_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON,
        ANALYSIS,
        "--data-dir", DATA_DIR,
        "--out-dir", out_dir,
        "--iso-dr", str(iso),
        "--auc-max-fpr", str(AUC_MAX_FPR),
        "--fast-scan",
    ]

    print(f"[scan] running iso_dr={iso:.2f}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path = os.path.join(out_dir, "run.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(combined)

    if proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout or "").strip().splitlines()[:12]
        print(f"    returncode={proc.returncode}  (analysis failed)")
        if snippet:
            print("    --- error snippet ---")
            for line in snippet:
                print("    " + line)
        print(f"    log: {log_path}")
        rows.append({
            "iso_dr": iso,
            "returncode": proc.returncode,
            "auc_max_fpr": AUC_MAX_FPR,
            "auc": float("nan"),
            "fixed_fake_TPR99": float("nan"),
            "ntracks_mean": float("nan"),
            "ntracks_median": float("nan"),
            "ntracks_p95": float("nan"),
            "out_dir": out_dir,
            "log": log_path,
        })
        continue

    auc = float("nan")
    fixed_fake = float("nan")

    for line in combined.splitlines():
        m = RE_AUC.search(line)
        if m:
            auc = float(m.group("auc"))
        m = RE_FIXED.search(line)
        if m:
            fixed_fake = float(m.group("fake"))

    # ROC file (try local then recursive)
    roc_patterns = ["roc*.csv", "ROC*.csv", "*roc*.csv", "metrics_roc*.csv", "*_roc_points*.csv"]
    roc_path = _find_first(roc_patterns, out_dir, recursive=False) or _find_first(roc_patterns, out_dir, recursive=True)
    if roc_path:
        fpr, tpr = _read_roc_csv(roc_path)
        if fpr is not None and tpr is not None:
            roc_curves[iso] = (fpr, tpr, roc_path)
            print(f"    [scan] ROC file: {os.path.relpath(roc_path, out_dir)}")
        else:
            print(f"    [scan] ROC file found but unreadable: {os.path.relpath(roc_path, out_dir)}")
    else:
        print("    [scan] ROC file: (none found)")

    # Ntracks file (try local then recursive)
    ntrack_patterns = ["ntracks*.csv", "tracks*.csv", "*ntracks*.csv", "*tracks_per*.csv", "*_ntracks*.csv"]
    ntracks_path = _find_first(ntrack_patterns, out_dir, recursive=False) or _find_first(ntrack_patterns, out_dir, recursive=True)

    n_iso, frac_kept = _read_ntracks_csv(ntracks_path) if ntracks_path else (None, None)

    n_mean = float("nan")
    n_median = float("nan")
    n_p95 = float("nan")

    f_mean = float("nan")
    f_median = float("nan")
    f_p95 = float("nan")

    if n_iso is not None and len(n_iso) > 0:
        n_mean = float(np.mean(n_iso))
        n_median = float(np.median(n_iso))
        n_p95 = float(np.percentile(n_iso, 95))
        ntracks_dists[iso] = (n_iso, ntracks_path)

        if frac_kept is not None and len(frac_kept) > 0:
            f_mean = float(np.mean(frac_kept))
            f_median = float(np.median(frac_kept))
            f_p95 = float(np.percentile(frac_kept, 95))
            frac_kept_dists[iso] = (frac_kept, ntracks_path)

        print(f"    [scan] Ntracks file: {os.path.relpath(ntracks_path, out_dir)}  (N={len(n_iso)})")

        plt.figure()
        plt.hist(n_iso, bins=60)
        plt.xlabel("Ntracks (iso cone)")
        plt.ylabel("Entries")
        plt.title(f"Ntracks distribution (iso_dr={iso:.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ntracks_hist_iso_dr_{iso:.2f}".replace(".", "p") + ".png"), dpi=160)
        plt.close()
    else:
        if ntracks_path:
            print(f"    [scan] Ntracks file found but unreadable/empty: {os.path.relpath(ntracks_path, out_dir)}")
        else:
            print("    [scan] Ntracks file: (none found)")


    print(
        f"    returncode={proc.returncode}  auc={auc}  fake@99={fixed_fake}  "
        f"ntracks_mean={n_mean if not isnan(n_mean) else 'nan'}  "
        f"frac_kept_mean={f_mean if not isnan(f_mean) else 'nan'}  out={out_dir}"
    )


    rows.append({
        "iso_dr": iso,
        "returncode": proc.returncode,
        "auc_max_fpr": AUC_MAX_FPR,
        "auc": auc,
        "fixed_fake_TPR99": fixed_fake,
        "ntracks_mean": n_mean,
        "ntracks_median": n_median,
        "ntracks_p95": n_p95,
        "frac_kept_mean": f_mean,
        "frac_kept_median": f_median,
        "frac_kept_p95": f_p95,
        "out_dir": out_dir,
        "log": log_path,
    })

csv_path = os.path.join(OUT_BASE, "iso_dr_scan_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

if roc_curves:
    plt.figure()
    for iso in sorted(roc_curves.keys()):
        fpr, tpr, _ = roc_curves[iso]
        plt.plot(fpr, tpr, label=f"iso_dr={iso:.2f}")
    plt.xlabel("Fake rate (FPR)")
    plt.ylabel("Acceptance (TPR)")
    plt.title("ROC overlay for iso_dr scan")
    plt.legend(fontsize=8)
    plt.tight_layout()
    roc_fig = os.path.join(OUT_BASE, "roc_overlay.png")
    plt.savefig(roc_fig, dpi=180)
    plt.close()
    print(f"[scan] wrote: {roc_fig}")
else:
    print("[scan] no ROC CSVs found; skipped ROC overlay plot")

if ntracks_dists:
    plt.figure()
    for iso in sorted(ntracks_dists.keys()):
        arr, _ = ntracks_dists[iso]
        x = np.sort(arr)
        y = np.linspace(0.0, 1.0, len(x), endpoint=True)
        plt.plot(x, y, label=f"iso_dr={iso:.2f}")
    plt.xlabel("Ntracks")
    plt.ylabel("CDF")
    plt.title("Ntracks CDF overlay for iso_dr scan")
    plt.legend(fontsize=8)
    plt.tight_layout()
    cdf_fig = os.path.join(OUT_BASE, "ntracks_cdf_overlay.png")
    plt.savefig(cdf_fig, dpi=180)
    plt.close()
    print(f"[scan] wrote: {cdf_fig}")

    plt.figure()
    isos = sorted(ntracks_dists.keys())
    data = [ntracks_dists[iso][0] for iso in isos]
    plt.boxplot(data, labels=[f"{iso:.2f}" for iso in isos], showfliers=False)
    plt.xlabel("iso_dr")
    plt.ylabel("Ntracks")
    plt.title("Ntracks boxplot (outliers hidden)")
    plt.tight_layout()
    box_fig = os.path.join(OUT_BASE, "ntracks_boxplot.png")
    plt.savefig(box_fig, dpi=180)
    plt.close()
    print(f"[scan] wrote: {box_fig}")
else:
    print("[scan] no Ntracks CSVs found; skipped Ntracks plots")

valid = [r for r in rows if r["returncode"] == 0 and not isnan(r["auc"])]
valid.sort(key=lambda r: (-r["auc"], r["fixed_fake_TPR99"] if not isnan(r["fixed_fake_TPR99"]) else 1e9))

print("\n[scan] wrote:", csv_path)
print("[scan] top candidates (highest AUC, then lowest fake@TPR=0.99):")
for r in valid[:5]:
    fake99 = r["fixed_fake_TPR99"]
    fake99_s = f"{fake99:.6f}" if not isnan(fake99) else "nan"
    nmean = r["ntracks_mean"]
    nmean_s = f"{nmean:.3f}" if not isnan(nmean) else "nan"
    print(
        f"  iso_dr={r['iso_dr']:.2f}  auc={r['auc']:.6f} (max_fpr={r['auc_max_fpr']})  "
        f"fake@99={fake99_s}  ntracks_mean={nmean_s}  out={r['out_dir']}"
    )

print("\n[scan] done.")