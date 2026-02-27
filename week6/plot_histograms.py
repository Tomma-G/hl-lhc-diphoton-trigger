#!/usr/bin/env python3
"""
Histogram plotting utility.

This script reads histogram-style CSV files (or ROOT TH1.Print() dumps)
and produces PNG plots.

Supported input formats:

1) Standard CSV histograms
   - 1D with bin edges:
       bin_low, bin_high, count/value
   - 1D with bin centres:
       bin_center, count/value
   - Binned XY ("vs" plots):
       x_low, x_high, y (+ optional yerr)

2) ROOT TH1.Print() dumps saved as text/CSV-like files:
   TH1.Print Name  = ..., Entries= ...
    fSumw[0]=..., x=..., error=...
    ...

Outputs:
  <out_dir>/*.png

Usage:
  python plot_histograms.py --in-dir path/to/csvs --out-dir plots

Notes:
  - Column name matching is case-insensitive.
  - Empty bins (y <= 0) are excluded when parsing ROOT dumps.
  - Files are located recursively under --in-dir.
"""

import argparse
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Expected input filenames (edit if filenames differ).
# Keys define internal plot names; values are filenames searched
# recursively under --in-dir.
# ----------------------------------------------------------------------
FILES: Dict[str, str] = {
    "num_offl_all": "num_offl_all.csv",
    "num_offl_selected": "num_offl_selected.csv",
    "avgNum_offl_selected_vs_actualMu": "avgNum_offl_selected_vs_actualMu.csv",
    "offl_d0": "offl_d0.csv",
    "truth_d0": "truth_d0.csv",
    "resolution_d0_vs_truth_pt": "resolution_d0_vs_truth_pt.csv",
    "resolution_d0_vs_truth_eta": "resolution_d0_vs_truth_eta.csv",
}

# ----------------------------------------------------------------------
# Plot labels (modify as required for presentation or publication).
# ----------------------------------------------------------------------
TITLES = {
    "num_offl_all": "Track multiplicity: all offline tracks",
    "num_offl_selected": "Track multiplicity: offline selected tracks",
    "avgNum_offl_selected_vs_actualMu": "Average # offline selected tracks vs pile-up (μ)",
    "offl_d0": "Reco track d0 (offline)",
    "truth_d0": "Truth track d0",
    "resolution_d0_vs_truth_pt": "d0 resolution vs truth pT",
    "resolution_d0_vs_truth_eta": "d0 resolution vs truth η",
}

XLABELS = {
    "num_offl_all": "# tracks",
    "num_offl_selected": "# tracks",
    "avgNum_offl_selected_vs_actualMu": "actual μ",
    "offl_d0": "d0 [units as in CSV]",
    "truth_d0": "d0 [units as in CSV]",
    "resolution_d0_vs_truth_pt": "truth pT [units as in CSV]",
    "resolution_d0_vs_truth_eta": "truth η",
}

YLABELS = {
    "num_offl_all": "Entries",
    "num_offl_selected": "Entries",
    "avgNum_offl_selected_vs_actualMu": "avg # selected tracks",
    "offl_d0": "Entries",
    "truth_d0": "Entries",
    "resolution_d0_vs_truth_pt": "resolution (width) [units as in CSV]",
    "resolution_d0_vs_truth_eta": "resolution (width) [units as in CSV]",
}


def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    """
    Return the first matching column name (case-insensitive)
    from a list of candidate names.
    """
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def is_root_th1_print_file(path: str) -> bool:
    """
    Determine whether a file appears to be a ROOT TH1.Print() dump
    based on its first line.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        return ("TH1.Print" in first) or first.startswith("[TH1.Print]") or first.startswith("TH1.Print")
    except Exception:
        return False


def read_root_th1_print(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Parse a ROOT TH1.Print() style dump.

    Extracted quantities:
      - x : bin centres
      - y : bin contents (fSumw)
      - e : bin errors (if present)

    Empty bins (y <= 0) are excluded.
    Output is sorted by increasing x.
    """
    xs = []
    ys = []
    es = []

    rx_x = re.compile(r"\bx\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    rx_y = re.compile(r"\bfSumw\[\d+\]\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    rx_e = re.compile(r"\berror\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            mx = rx_x.search(s)
            my = rx_y.search(s)
            if not (mx and my):
                continue

            xval = float(mx.group(1))
            yval = float(my.group(1))
            if not np.isfinite(xval) or not np.isfinite(yval):
                continue

            if yval <= 0.0:
                continue

            me = rx_e.search(s)
            eval_ = float(me.group(1)) if me else np.nan

            xs.append(xval)
            ys.append(yval)
            es.append(eval_)

    if not xs:
        raise ValueError(f"No (x, fSumw) rows found in ROOT TH1.Print dump: {path}")

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    e = np.asarray(es, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    e = e[order]

    if not np.any(np.isfinite(e)):
        return x, y, None

    return x, y, e


def read_1d_hist(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a 1D histogram file.

    Returns:
      x : bin centres
      y : bin contents

    Supports standard CSV formats and ROOT TH1.Print dumps.
    """
    if is_root_th1_print_file(path):
        x, y, _ = read_root_th1_print(path)
        return x, y

    df = pd.read_csv(path)

    c_lo = _find_col(df, ["bin_low", "low", "x_low", "lo", "left"])
    c_hi = _find_col(df, ["bin_high", "high", "x_high", "hi", "right"])
    c_y = _find_col(df, ["count", "counts", "entries", "y", "value", "val"])

    if c_lo and c_hi and c_y:
        lo = df[c_lo].to_numpy(dtype=float)
        hi = df[c_hi].to_numpy(dtype=float)
        y = df[c_y].to_numpy(dtype=float)
        x = 0.5 * (lo + hi)
        return x, y

    c_x = _find_col(df, ["bin_center", "center", "x", "bin", "x_center"])
    if c_x and c_y:
        x = df[c_x].to_numpy(dtype=float)
        y = df[c_y].to_numpy(dtype=float)
        return x, y

    raise ValueError(f"Unrecognised 1D histogram format in {path}. Columns: {list(df.columns)}")


def read_binned_xy(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Read a binned XY histogram (e.g. resolution vs variable).

    Returns:
      x : bin centres
      y : values
      e : optional uncertainties (or None)
    """
    if is_root_th1_print_file(path):
        return read_root_th1_print(path)

    df = pd.read_csv(path)

    c_lo = _find_col(df, ["x_low", "bin_low", "low", "lo", "left"])
    c_hi = _find_col(df, ["x_high", "bin_high", "high", "hi", "right"])
    c_x = _find_col(df, ["x", "bin_center", "center", "x_center", "mu", "pt", "eta"])
    c_y = _find_col(df, ["y", "value", "mean", "avg", "resolution"])
    c_e = _find_col(df, ["yerr", "y_err", "err", "error", "sigma", "stderr"])

    if c_lo and c_hi and c_y:
        lo = df[c_lo].to_numpy(dtype=float)
        hi = df[c_hi].to_numpy(dtype=float)
        x = 0.5 * (lo + hi)
        y = df[c_y].to_numpy(dtype=float)
        e = df[c_e].to_numpy(dtype=float) if c_e else None
        return x, y, e

    if c_x and c_y:
        x = df[c_x].to_numpy(dtype=float)
        y = df[c_y].to_numpy(dtype=float)
        e = df[c_e].to_numpy(dtype=float) if c_e else None
        return x, y, e

    raise ValueError(f"Unrecognised binned XY format in {path}. Columns: {list(df.columns)}")


def plot_1d(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, out_png: str) -> None:
    """Produce a step-style 1D histogram plot."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    plt.figure()
    plt.step(x, y, where="mid")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_xy(x: np.ndarray, y: np.ndarray, e: Optional[np.ndarray], title: str, xlabel: str, ylabel: str, out_png: str) -> None:
    """Produce a binned XY plot with optional error bars."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if e is not None:
        e = np.asarray(e, dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
        x, y, e = x[m], y[m], e[m]
    else:
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

    plt.figure()
    if e is not None and e.size > 0:
        plt.errorbar(x, y, yerr=e, fmt="o", capsize=2)
    else:
        plt.plot(x, y, marker="o")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if x.size > 0 and x.size <= 6:
        plt.suptitle("NOTE: only a few populated bins in this file", y=0.98, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    """Locate input files and generate PNG plots."""
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--in-dir",
        default=r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\initial_data\histograms",
        help="folder containing the histogram CSVs",
    )

    ap.add_argument(
        "--out-dir",
        default=r"C:\Users\Tom Greenwood\Desktop\University\Year 4\SEM 2\SH Project\week6\hist_plots",
        help="folder to write PNGs",
    )

    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    one_d = {"num_offl_all", "num_offl_selected", "offl_d0", "truth_d0"}

    for key, fname in FILES.items():
        path = None
        for root, _, files in os.walk(in_dir):
            if fname in files:
                path = os.path.join(root, fname)
                break

        if path is None:
            print(f"[warn] missing: {fname}")
            continue

        out_png = os.path.join(out_dir, f"{key}.png")

        try:
            if key in one_d:
                x, y = read_1d_hist(path)
                plot_1d(x, y, TITLES[key], XLABELS[key], YLABELS[key], out_png)
            else:
                x, y, e = read_binned_xy(path)
                plot_xy(x, y, e, TITLES[key], XLABELS[key], YLABELS[key], out_png)

            print(f"[ok] wrote {out_png}")

        except Exception as ex:
            print(f"[error] failed on {path}: {ex}")


if __name__ == "__main__":
    main()