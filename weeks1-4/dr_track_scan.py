#!/usr/bin/env python3
"""
Scan dr_track for the HL-LHC diphoton analysis (analysis_v6.py style outputs).

For each dr_track value:
  - runs analysis script in --fast-scan mode into a per-point out_dir
  - extracts:
      * best track-count separation (eff/fake/rej) from the analysis stdout
      * fake rate at fixed TPR (default 0.99) from the iso ROC summary line
      * AUC (optional; full or partial via AUC_MAX_FPR)
  - writes a summary CSV
  - produces optional ROC overlay plot (if ROC CSVs exist)

Notes:
  - analysis_v6.py does NOT print "best cut ... J=..." (your old regex won't match).
  - Instead, it prints a track-count cut line:
      "track-count cut (dR<...): photon if n_tracks<=t, t=..., eff=..., fake=..., rej=..."
    We parse that and compute Youden J = eff - fake.
"""

import csv
import os
import re
import subprocess
from math import isnan

import numpy as np
import matplotlib.pyplot as plt

# ---- USER SETTINGS ----
PYTHON = r"E:\Anaconda\python.exe"
ANALYSIS = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\analysis_v6.py"

DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"
OUT_BASE = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_dr_track"

DR_TRACK_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

N_EVENTS = 1000
FIXED_ISO_DR = 0.30          # set to your chosen best iso_dr (0.30 from your earlier decision)
FIXED_DR_OVERLAP = 0.40      # your jet–photon overlap removal radius
TARGET_TPR = 0.99

# Choose full AUC (=1.0) or partial AUC (e.g. 0.01 for trigger region)
AUC_MAX_FPR = 1.0

# -----------------------

os.makedirs(OUT_BASE, exist_ok=True)

# AUC line from analysis_v6.py:
#   [info] AUC (max_fpr=1) = 0.913714
RE_AUC = re.compile(
    r"\bAUC\s*\(max_fpr=(?P<maxfpr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\s*=\s*(?P<auc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Fixed-TPR line:
#   [info] FIXED TPR=0.99: cut=..., acc=0.990, fake=0.437, rej=0.563
RE_FIXED = re.compile(
    rf"FIXED TPR={TARGET_TPR:.2f}:\s*cut=[^,]+,\s*acc=(?P<acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*fake=(?P<fake>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Track-count cut line:
#   [info] track-count cut (dR<0.1): photon if n_tracks<=t, t=7, eff=0.812, fake=0.421, rej=0.579
RE_NTRK_CUT = re.compile(
    r"track-count cut \(dR<(?P<dr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\):\s*"
    r"photon if n_tracks<=t,\s*t=(?P<t>\d+),\s*eff=(?P<eff>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
    r"fake=(?P<fake>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*rej=(?P<rej>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

def read_roc_csv(path):
    import csv as _csv
    if not path or (not os.path.exists(path)):
        return None, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        r = _csv.DictReader(f)
        if not r.fieldnames:
            return None, None
        fields = {h.strip().lower(): h for h in r.fieldnames}
        if "fpr" not in fields or "tpr" not in fields:
            return None, None
        fpr_key = fields["fpr"]
        tpr_key = fields["tpr"]
        xs, ys = [], []
        for row in r:
            xf = safe_float(row.get(fpr_key, None), default=None)
            yt = safe_float(row.get(tpr_key, None), default=None)
            if xf is None or yt is None:
                continue
            xs.append(xf)
            ys.append(yt)
    if not xs:
        return None, None
    fpr = np.asarray(xs, dtype=float)
    tpr = np.asarray(ys, dtype=float)
    order = np.argsort(fpr)
    return fpr[order], tpr[order]

rows = []
roc_curves = {}  # dr_track -> (fpr, tpr)

for dr in DR_TRACK_VALUES:
    tag = f"dr_track_{dr:.2f}".replace(".", "p")
    out_dir = os.path.join(OUT_BASE, tag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run.log")

    cmd = [
        PYTHON, ANALYSIS,
        "--data-dir", DATA_DIR,
        "--n-events", str(N_EVENTS),
        "--iso-dr", str(FIXED_ISO_DR),
        "--dr-overlap", str(FIXED_DR_OVERLAP),
        "--dr-track", str(dr),
        "--auc-max-fpr", str(AUC_MAX_FPR),
        "--out-dir", out_dir,
    ]

    print(f"[scan] running dr_track={dr:.2f}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(combined)

    auc = float("nan")
    fake99 = float("nan")
    acc99 = float("nan")

    # track-count metrics
    ntrk_t = float("nan")
    ntrk_eff = float("nan")
    ntrk_fake = float("nan")
    ntrk_rej = float("nan")
    youdenJ = float("nan")

    if proc.returncode == 0:
        for line in combined.splitlines():
            m = RE_AUC.search(line)
            if m:
                auc = safe_float(m.group("auc"))
            m = RE_FIXED.search(line)
            if m:
                acc99 = safe_float(m.group("acc"))
                fake99 = safe_float(m.group("fake"))
            m = RE_NTRK_CUT.search(line)
            if m:
                ntrk_t = safe_float(m.group("t"))
                ntrk_eff = safe_float(m.group("eff"))
                ntrk_fake = safe_float(m.group("fake"))
                ntrk_rej = safe_float(m.group("rej"))
                if np.isfinite(ntrk_eff) and np.isfinite(ntrk_fake):
                    youdenJ = float(ntrk_eff - ntrk_fake)

        # grab roc.csv if present (analysis_v6 writes roc.csv even in fast-scan)
        roc_path = os.path.join(out_dir, "roc.csv")
        fpr, tpr = read_roc_csv(roc_path)
        if fpr is not None and tpr is not None:
            roc_curves[dr] = (fpr, tpr)

        print(
            f"    auc={auc if not isnan(auc) else float('nan'):.6f}  "
            f"fake@{TARGET_TPR:.2f}={fake99 if not isnan(fake99) else float('nan'):.3f}  "
            f"ntrkJ={youdenJ if not isnan(youdenJ) else float('nan'):.3f}"
        )
    else:
        print(f"    analysis failed (returncode={proc.returncode}); log: {log_path}")

    rows.append({
        "dr_track": dr,
        "returncode": proc.returncode,
        "n_events": N_EVENTS,
        "iso_dr": FIXED_ISO_DR,
        "dr_overlap": FIXED_DR_OVERLAP,
        "auc_max_fpr": AUC_MAX_FPR,
        "auc": auc,
        "acc_at_TPR_0p99": acc99,
        "fake_at_TPR_0p99": fake99,
        "ntrk_best_t": ntrk_t,
        "ntrk_eff": ntrk_eff,
        "ntrk_fake": ntrk_fake,
        "ntrk_rej": ntrk_rej,
        "ntrk_youdenJ": youdenJ,
        "out_dir": out_dir,
        "log": log_path,
    })

# ---- write summary CSV ----
csv_path = os.path.join(OUT_BASE, "dr_track_scan_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[scan] wrote: {csv_path}")

# ---- ROC overlay ----
if roc_curves:
    plt.figure()
    for dr in sorted(roc_curves.keys()):
        fpr, tpr = roc_curves[dr]
        plt.plot(fpr, tpr, label=f"dr_track={dr:.2f}")
    plt.xlabel("Fake rate (FPR)")
    plt.ylabel("Acceptance (TPR)")
    plt.title(f"ROC overlay (iso_dr={FIXED_ISO_DR:.2f}, dr_overlap={FIXED_DR_OVERLAP:.2f})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    roc_fig = os.path.join(OUT_BASE, "roc_overlay_dr_track.png")
    plt.savefig(roc_fig, dpi=180)
    plt.close()
    print(f"[scan] wrote: {roc_fig}")

# ---- ranking ----
# Primary: lowest fake@TPR=0.99 (iso)
# Secondary: highest ntrk_youdenJ (track-count separation), then highest AUC
valid = [r for r in rows if r["returncode"] == 0]
valid.sort(
    key=lambda r: (
        r["fake_at_TPR_0p99"] if not isnan(r["fake_at_TPR_0p99"]) else 1e9,
        -(r["ntrk_youdenJ"] if not isnan(r["ntrk_youdenJ"]) else -1e9),
        -(r["auc"] if not isnan(r["auc"]) else -1e9),
    )
)

print("[scan] top candidates (lowest fake@99, then highest ntrk Youden J, then highest AUC):")
for r in valid[:10]:
    print(
        f"  dr_track={r['dr_track']:.2f}  "
        f"fake@99={r['fake_at_TPR_0p99']:.3f}  "
        f"ntrkJ={r['ntrk_youdenJ']:.3f} (t={int(r['ntrk_best_t']) if not isnan(r['ntrk_best_t']) else 'nan'})  "
        f"auc={r['auc']:.6f}"
    )

print("\n[scan] done.")