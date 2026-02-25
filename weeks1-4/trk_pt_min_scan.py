#!/usr/bin/env python3
"""
Scan trk_pt_min for the HL-LHC diphoton analysis (analysis_v7.py).

For each trk_pt_min value:
  - runs analysis script
  - extracts:
      * AUC (from "[info] AUC (max_fpr=...)=..." line)
      * fake rate at fixed TPR=0.99 (from "FIXED TPR=0.99 ..." line)
  - reads:
      * track_ptmin_reduction.csv (effect of trk_pt_min on total track multiplicity)
      * track_reduction.csv (diagnostic-only: tracks within dr_track_keep of any photon)
  - writes a summary CSV
  - ranks by: lowest fake@0.99, then highest AUC, then highest kept_fraction_pt

Notes:
  - Converted photons are excluded by default in analysis_v7.py (do NOT pass --include-converted).
  - Uses --fast-scan to speed up (still writes ROC points + AUC + fixed-TPR line).
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from math import isnan


# ---- USER SETTINGS ----
PYTHON = r"E:\Anaconda\python.exe"
ANALYSIS = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\week5\analysis_v7.py"
DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"
OUT_BASE = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_trk_pt_min"

# scan points (GeV)
PT_MINS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]

# fixed analysis knobs (keep constant so you isolate pT-min effects)
N_EVENTS = 1000
FIXED_ISO_DR = 0.20
FIXED_DR_TRACK = 0.10
FIXED_DR_OVERLAP = 0.20

# AUC integration region: use 1.0 for full AUC or e.g. 0.05 for trigger-relevant region
AUC_MAX_FPR = 0.05

TARGET_TPR = 0.99
# -----------------------


# ---- Regexes ----
RE_AUC = re.compile(
    r"\bAUC\s*\(max_fpr=(?P<maxfpr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\s*=\s*(?P<auc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
RE_FIXED = re.compile(
    rf"FIXED TPR={TARGET_TPR:.2f}:\s*cut=[^,]+,\s*acc=(?P<acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*fake=(?P<fake>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_track_reduction_csv(path: str):
    """
    Reads out_dir/track_reduction.csv written by analysis_v7.py.
    This is the *dR-to-photons* diagnostic: tracks within dr_track_keep of any photon.
    Expected columns: kept_tracks, all_tracks, kept_fraction, dr_track_keep
    """
    out = {
        "kept_tracks": float("nan"),
        "all_tracks": float("nan"),
        "kept_fraction": float("nan"),
        "dr_track_keep": float("nan"),
    }
    if not path or not os.path.exists(path):
        return out

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return out

        r0 = rows[0]
        if "kept_tracks" in r0:
            out["kept_tracks"] = safe_float(r0["kept_tracks"])
        if "all_tracks" in r0:
            out["all_tracks"] = safe_float(r0["all_tracks"])
        if "kept_fraction" in r0:
            out["kept_fraction"] = safe_float(r0["kept_fraction"])
        if "dr_track_keep" in r0:
            out["dr_track_keep"] = safe_float(r0["dr_track_keep"])
        return out
    except Exception:
        return out


def read_track_ptmin_reduction_csv(path: str):
    """
    Reads out_dir/track_ptmin_reduction.csv written by analysis_v7.py.
    This is the *trk_pt_min* diagnostic: how many tracks survive the pT cut.
    Expected columns: trk_pt_min, tracks_before_pt, tracks_after_pt, kept_fraction_pt
    """
    out = {
        "trk_pt_min": float("nan"),
        "tracks_before_pt": float("nan"),
        "tracks_after_pt": float("nan"),
        "kept_fraction_pt": float("nan"),
    }
    if not path or not os.path.exists(path):
        return out

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return out

        r0 = rows[0]
        if "trk_pt_min" in r0:
            out["trk_pt_min"] = safe_float(r0["trk_pt_min"])
        if "tracks_before_pt" in r0:
            out["tracks_before_pt"] = safe_float(r0["tracks_before_pt"])
        if "tracks_after_pt" in r0:
            out["tracks_after_pt"] = safe_float(r0["tracks_after_pt"])
        if "kept_fraction_pt" in r0:
            out["kept_fraction_pt"] = safe_float(r0["kept_fraction_pt"])
        return out
    except Exception:
        return out


def fmt(x: float, ndp: int = 6) -> str:
    if x is None or (isinstance(x, float) and isnan(x)):
        return "nan"
    return f"{x:.{ndp}f}"


def main():
    os.makedirs(OUT_BASE, exist_ok=True)

    rows = []

    for ptmin in PT_MINS:
        tag = f"ptmin_{ptmin:.2f}".replace(".", "p")
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
            "--trk-pt-min", str(ptmin),
            "--auc-max-fpr", str(AUC_MAX_FPR),
            "--fast-scan",
        ]

        print(f"[scan] running trk_pt_min={ptmin:g}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(combined)

        auc = float("nan")
        fake99 = float("nan")
        acc99 = float("nan")

        if proc.returncode == 0:
            for line in combined.splitlines():
                m = RE_AUC.search(line)
                if m:
                    auc = safe_float(m.group("auc"))
                m = RE_FIXED.search(line)
                if m:
                    acc99 = safe_float(m.group("acc"))
                    fake99 = safe_float(m.group("fake"))

            # diagnostic-only: dr_track_keep near photons
            tr_dr_path = os.path.join(out_dir, "track_reduction.csv")
            tr_dr = read_track_reduction_csv(tr_dr_path)

            # trk_pt_min diagnostic: how many tracks survive pt cut
            tr_pt_path = os.path.join(out_dir, "track_ptmin_reduction.csv")
            tr_pt = read_track_ptmin_reduction_csv(tr_pt_path)

            print(
                f"    returncode=0  auc={fmt(auc, 6)}  fake@0.99={fmt(fake99, 3)}  "
                f"kept_frac_pt={fmt(tr_pt['kept_fraction_pt'], 3)}  "
                f"kept_frac_dr={fmt(tr_dr['kept_fraction'], 3)}  out={out_dir}"
            )

            rows.append({
                "trk_pt_min": ptmin,
                "returncode": proc.returncode,
                "n_events": N_EVENTS,
                "iso_dr": FIXED_ISO_DR,
                "dr_track": FIXED_DR_TRACK,
                "dr_overlap": FIXED_DR_OVERLAP,
                "auc_max_fpr": AUC_MAX_FPR,
                "auc": auc,
                "acc_at_TPR_0p99": acc99,
                "fake_at_TPR_0p99": fake99,

                # pt-min diagnostic (this is the one you should use for “tracks kept”)
                "tracks_before_pt": tr_pt["tracks_before_pt"],
                "tracks_after_pt": tr_pt["tracks_after_pt"],
                "kept_fraction_pt": tr_pt["kept_fraction_pt"],

                # dr diagnostic (useful but *not* “tracks kept by ptmin”)
                "kept_tracks_dr": tr_dr["kept_tracks"],
                "all_tracks_dr": tr_dr["all_tracks"],
                "kept_fraction_dr": tr_dr["kept_fraction"],
                "dr_track_keep": tr_dr["dr_track_keep"],

                "out_dir": out_dir,
                "log": log_path,
            })
        else:
            snippet = (proc.stderr or proc.stdout or "").strip().splitlines()[:15]
            print(f"    returncode={proc.returncode} (analysis failed)")
            if snippet:
                print("    --- error snippet ---")
                for line in snippet:
                    print("    " + line)
            print(f"    log: {log_path}")

            rows.append({
                "trk_pt_min": ptmin,
                "returncode": proc.returncode,
                "n_events": N_EVENTS,
                "iso_dr": FIXED_ISO_DR,
                "dr_track": FIXED_DR_TRACK,
                "dr_overlap": FIXED_DR_OVERLAP,
                "auc_max_fpr": AUC_MAX_FPR,
                "auc": float("nan"),
                "acc_at_TPR_0p99": float("nan"),
                "fake_at_TPR_0p99": float("nan"),
                "tracks_before_pt": float("nan"),
                "tracks_after_pt": float("nan"),
                "kept_fraction_pt": float("nan"),
                "kept_tracks_dr": float("nan"),
                "all_tracks_dr": float("nan"),
                "kept_fraction_dr": float("nan"),
                "dr_track_keep": float("nan"),
                "out_dir": out_dir,
                "log": log_path,
            })

    # ---- write summary CSV ----
    csv_path = os.path.join(OUT_BASE, "trk_pt_min_scan_summary.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[scan] wrote: {csv_path}")

    # ---- ranking ----
    valid = [r for r in rows if r["returncode"] == 0]
    valid.sort(
        key=lambda r: (
            r["fake_at_TPR_0p99"] if not isnan(r["fake_at_TPR_0p99"]) else 1e9,
            -(r["auc"] if not isnan(r["auc"]) else -1e9),
            -(r["kept_fraction_pt"] if not isnan(r["kept_fraction_pt"]) else -1e9),
        )
    )

    print("[scan] top candidates (lowest fake@0.99, then highest AUC, then highest kept_fraction_pt):")
    for r in valid:
        print(
            f"  trk_pt_min={r['trk_pt_min']:.2f}  "
            f"fake@0.99={fmt(r['fake_at_TPR_0p99'], 3)}  "
            f"auc={fmt(r['auc'], 6)} (max_fpr={r['auc_max_fpr']})  "
            f"kept_frac_pt={fmt(r['kept_fraction_pt'], 3)}  "
            f"kept_frac_dr={fmt(r['kept_fraction_dr'], 3)}  "
            f"out={r['out_dir']}"
        )

    print("\n[scan] done.")


if __name__ == "__main__":
    main()