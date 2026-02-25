#!/usr/bin/env python3
r"""
Scan iso_dr by running the analysis script repeatedly and extracting:

- AUC from the "[info] AUC (max_fpr=...)" line
- fake@TPR=0.99 from the "FIXED TPR=0.99 ..." line

Writes per-point logs and a summary CSV.

Notes:
- This scan script sets --out-dir so each iso_dr point writes outputs into its own folder.
- It also enables --fast-scan for speed (analysis still writes ROC points + AUC in fast-scan mode).
- If you want partial AUC (trigger-relevant region), set AUC_MAX_FPR to e.g. 0.01 or 0.05.
"""

import csv
import os
import re
import subprocess
from math import isnan

ISO_DRS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

PYTHON = r"E:\Anaconda\python.exe"
ANALYSIS = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\analysis_v6.py"
DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"
OUT_BASE = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_iso_dr"

# Choose full AUC (=1.0) or partial AUC (e.g. 0.01 for 1% fake-rate region)
AUC_MAX_FPR = 1.0

os.makedirs(OUT_BASE, exist_ok=True)

# AUC line printed by updated analysis script:
# [info] AUC (max_fpr=1) = 0.873421
RE_AUC = re.compile(
    r"\bAUC\s*\(max_fpr=(?P<maxfpr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\s*=\s*(?P<auc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# fixed operating point line still printed by analysis script
RE_FIXED = re.compile(
    r"FIXED TPR=0\.99:\s*cut=[^,]+,\s*acc=(?P<acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*fake=(?P<fake>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

rows = []

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

    # If analysis failed, show why immediately
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
            "out_dir": out_dir,
            "log": log_path,
        })
        continue

    auc = float("nan")
    fixed_fake = float("nan")

    for line in combined.splitlines():
        m = RE_AUC.search(line)
        if m:
            # take the last seen AUC line (should only be one)
            auc = float(m.group("auc"))

        m = RE_FIXED.search(line)
        if m:
            fixed_fake = float(m.group("fake"))

    print(
        f"    returncode={proc.returncode}  auc={auc}  fake@99={fixed_fake}  out={out_dir}"
    )

    rows.append({
        "iso_dr": iso,
        "returncode": proc.returncode,
        "auc_max_fpr": AUC_MAX_FPR,
        "auc": auc,
        "fixed_fake_TPR99": fixed_fake,
        "out_dir": out_dir,
        "log": log_path,
    })

csv_path = os.path.join(OUT_BASE, "iso_dr_scan_summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Ranking:
# Primary: maximise AUC (or partial AUC if AUC_MAX_FPR < 1)
# Secondary: minimise fake@TPR=0.99 (still a useful trigger-like operating point)
valid = [r for r in rows if r["returncode"] == 0 and not isnan(r["auc"])]
valid.sort(key=lambda r: (-r["auc"], r["fixed_fake_TPR99"] if not isnan(r["fixed_fake_TPR99"]) else 1e9))

print("\n[scan] wrote:", csv_path)
print("[scan] top candidates (highest AUC, then lowest fake@TPR=0.99):")
for r in valid[:5]:
    fake99 = r["fixed_fake_TPR99"]
    fake99_s = f"{fake99:.6f}" if not isnan(fake99) else "nan"
    print(
        f"  iso_dr={r['iso_dr']:.2f}  auc={r['auc']:.6f} (max_fpr={r['auc_max_fpr']})  "
        f"fake@99={fake99_s}  out={r['out_dir']}"
    )