#!/usr/bin/env python3
"""
Scan dr_overlap (jet–photon overlap removal) as a robustness check.

Runs analysis script for a set of dr_overlap values with a fixed iso_dr (e.g. 0.20),
collects:
  - bestJ from: "iso ROC ... J=..."
  - fake@99 from: "FIXED TPR=0.99 ... fake=..."
and writes a summary CSV.

Note: uses ASCII-only output parsing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


# -------- user settings --------
ANALYSIS_SCRIPT = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\analysis_v5.py"
PYTHON_EXE = r"E:\Anaconda\python.exe"

BASE_OUT_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\scan_dr_overlap"
DATA_DIR = r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\1k_ev"

N_EVENTS = 1000
ISO_DR_FIXED = 0.20

DR_OVERLAPS = [0.10, 0.15, 0.20, 0.30, 0.40]

# keep these fixed for fair comparison
DR_TRACK = 0.10
DR_TRACK_KEEP = 0.03
# --------------------------------


RE_BESTJ = re.compile(r"iso ROC .*? J=([0-9]*\.[0-9]+|[0-9]+)")
RE_FAKE99 = re.compile(r"FIXED TPR=0\.99: .*? fake=([0-9]*\.[0-9]+|[0-9]+)")


@dataclass
class ScanResult:
    dr_overlap: float
    returncode: int
    bestJ: float
    fake99: float
    out_dir: str
    log_path: str


def fmt_dir(x: float) -> str:
    # 0.20 -> "0p20"
    s = f"{x:.2f}".replace(".", "p")
    return s


def parse_metrics_from_log(text: str) -> tuple[float, float]:
    bestJ = float("nan")
    fake99 = float("nan")

    mJ = RE_BESTJ.search(text)
    if mJ:
        bestJ = float(mJ.group(1))

    mF = RE_FAKE99.search(text)
    if mF:
        fake99 = float(mF.group(1))

    return bestJ, fake99


def run_one(dr_overlap: float) -> ScanResult:
    tag = fmt_dir(dr_overlap)
    out_dir = os.path.join(BASE_OUT_DIR, f"dr_overlap_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run.log")

    cmd = [
        PYTHON_EXE,
        ANALYSIS_SCRIPT,
        "--data-dir", DATA_DIR,
        "--n-events", str(N_EVENTS),
        "--out-dir", out_dir,
        "--iso-dr", str(ISO_DR_FIXED),
        "--dr-overlap", str(dr_overlap),
        "--dr-track", str(DR_TRACK),
        "--dr-track-keep", str(DR_TRACK_KEEP),
    ]

    # run and capture stdout/stderr into log (UTF-8)
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    # read log back and parse metrics
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    bestJ, fake99 = parse_metrics_from_log(text)

    return ScanResult(
        dr_overlap=dr_overlap,
        returncode=proc.returncode,
        bestJ=bestJ,
        fake99=fake99,
        out_dir=out_dir,
        log_path=log_path,
    )


def main() -> None:
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    results: List[ScanResult] = []

    for dr in DR_OVERLAPS:
        print(f"[scan] running dr_overlap={dr:.2f}")
        res = run_one(dr)
        results.append(res)

        if res.returncode != 0:
            print(f"    returncode={res.returncode}  (analysis failed)")
            print(f"    log: {res.log_path}")
        else:
            print(f"    returncode=0  bestJ={res.bestJ:.3f}  fake@99={res.fake99:.3f}")
            print(f"    log: {res.log_path}")

    # write summary CSV
    df = pd.DataFrame([{
        "dr_overlap": r.dr_overlap,
        "iso_dr_fixed": ISO_DR_FIXED,
        "dr_track": DR_TRACK,
        "dr_track_keep": DR_TRACK_KEEP,
        "returncode": r.returncode,
        "bestJ": r.bestJ,
        "fake@99": r.fake99,
        "out_dir": r.out_dir,
        "log_path": r.log_path,
    } for r in results])

    summary_path = os.path.join(BASE_OUT_DIR, "dr_overlap_scan_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n[scan] wrote: {summary_path}")

    # rank: lowest fake@99, then highest bestJ
    ok = df[df["returncode"] == 0].copy()
    ok = ok.dropna(subset=["fake@99", "bestJ"])
    if len(ok) > 0:
        ok = ok.sort_values(["fake@99", "bestJ"], ascending=[True, False])
        print("[scan] top candidates (lowest fake@99, then highest J):")
        for _, row in ok.head(5).iterrows():
            print(
                f"  dr_overlap={row['dr_overlap']:.2f}  fake@99={row['fake@99']:.3f}  "
                f"bestJ={row['bestJ']:.3f}  log={row['log_path']}"
            )
    else:
        print("[scan] no successful runs with parsable metrics.")


if __name__ == "__main__":
    main()
