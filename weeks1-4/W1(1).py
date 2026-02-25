#!/usr/bin/env python3
"""
PRE-REMOVAL version (Tasks 1–2 only) for HL-LHC diphoton toy data.

Task 1:
  - take ~N events (files) from a directory containing photons_<event>.csv etc.

Task 2:
  - in events with exactly 2 photons, compute and plot m_yy (m_gg).

Assumed file formats (as observed):
  photons_<ev>.csv: pT, eta, phi, e, conversionType   (comma-separated, no header)

Notes:
  - This file intentionally DOES NOT apply photon–jet overlap removal.
  - Jets are not read, and no ΔR diagnostics are computed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")


def find_event_ids(data_dir: str) -> List[int]:
    """Return sorted event IDs that have a photons_<id>.csv file."""
    ids: List[int] = []
    for name in os.listdir(data_dir):
        m = PHOTON_RE.match(name)
        if m:
            ids.append(int(m.group(1)))
    ids.sort()
    return ids


def photons_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"photons_{event_id}.csv")


def photon_fourvec_from_row(row: pd.Series) -> np.ndarray:
    """Return (E, px, py, pz) for a photon row using (pT, eta, phi, e)."""
    pt = float(row["pT"])
    eta = float(row["eta"])
    phi = float(row["phi"])
    E = float(row["e"])

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    return np.array([E, px, py, pz], dtype=float)


def invariant_mass(fourvecs: np.ndarray) -> float:
    """fourvecs: shape (N,4) with columns (E,px,py,pz)."""
    total = np.sum(fourvecs, axis=0)
    E, px, py, pz = total
    m2 = E * E - (px * px + py * py + pz * pz)
    return float(np.sqrt(max(m2, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRE-REMOVAL: m_gg for 2-photon events (no photon–jet overlap removal)."
    )
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/100_ev",
        help="Directory containing photons_<event>.csv files",
    )
    parser.add_argument("--n-events", type=int, default=100, help="How many photon files (events) to scan")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument("--out-png", default="mgg_hist_before.png", help="Output histogram filename")
    parser.add_argument("--out-csv", default="mgg_values_before.csv", help="Output CSV filename with per-event m_gg")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[error] data-dir does not exist or is not a directory: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    event_ids = find_event_ids(args.data_dir)
    if not event_ids:
        print(f"[error] no photons_<id>.csv files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    chosen = event_ids[: args.n_events]
    print(f"Found {len(event_ids)} photon files. Processing first {len(chosen)} events.")

    mgg_rows: List[Dict[str, float]] = []
    n_read = 0
    n_two_photon = 0
    n_bad_ph = 0

    example_printed = False

    for ev in chosen:
        ph_path = photons_path(args.data_dir, ev)
        if not os.path.exists(ph_path) or os.path.getsize(ph_path) == 0:
            n_bad_ph += 1
            continue

        try:
            df_ph = pd.read_csv(
                ph_path,
                header=None,
                names=["pT", "eta", "phi", "e", "conversionType"],
            )[["pT", "eta", "phi", "e"]]
            n_read += 1
        except Exception as e:
            print(f"[warn] failed to read {ph_path}: {e}")
            n_bad_ph += 1
            continue

        if (not example_printed) and (len(df_ph) > 0):
            print(f"[debug] example event {ev}")
            print("[debug] photon row:", df_ph.iloc[0].to_dict())
            example_printed = True

        if len(df_ph) != 2:
            continue

        n_two_photon += 1
        fv = np.stack(
            [photon_fourvec_from_row(df_ph.iloc[0]), photon_fourvec_from_row(df_ph.iloc[1])],
            axis=0,
        )
        mgg = invariant_mass(fv)
        mgg_rows.append({"event_id": ev, "m_gg": mgg})

    if not mgg_rows:
        print("[error] no valid 2-photon events found in the selected files.", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame(mgg_rows)
    print(f"Read photon files: {n_read}")
    print(f"2-photon events:   {n_two_photon}")
    if n_bad_ph:
        print(f"Skipped photons (bad/empty): {n_bad_ph}")

    out.to_csv(args.out_csv, index=False)
    print(f"Saved per-event m_gg to {args.out_csv}")

    plt.figure()
    plt.hist(out["m_gg"].to_numpy(dtype=float), bins=args.bins)
    plt.xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    plt.ylabel("Entries")
    plt.title(r"Di-photon invariant mass for events with exactly 2 photons (pre-removal)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved histogram to {args.out_png}")


if __name__ == "__main__":
    main()