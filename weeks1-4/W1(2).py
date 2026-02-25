#!/usr/bin/env python3
"""
Tasks 1–3 for HL-LHC diphoton toy data.

Task 1:
  - take ~N events (files) from a directory containing photons_<event>.csv etc.

Task 2:
  - in events with exactly 2 photons, compute and plot m_yy (m_gg).

Task 3:
  - photon–jet overlap removal (remove jets within ΔR < 0.4 of any photon),
    and monitor efficiency (fraction of jets removed).
  - diagnostic: record min ΔR(jet, photon) per event and summarise.

Assumed file formats (as observed):
  photons_<ev>.csv: pT, eta, phi, e, conversionType   (comma-separated, no header)
  jets_<ev>.csv:    pT, eta, phi, e                   (comma-separated, no header)
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


def jets_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"jets_{event_id}.csv")


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2.0 * np.pi
    while dphi < -np.pi:
        dphi += 2.0 * np.pi
    return dphi


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return float(np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2))


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


def read_object_csv(path: str, names: List[str]) -> pd.DataFrame:
    """Read comma-separated, no-header CSV; return empty df if missing/empty/unreadable."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=names)
    try:
        return pd.read_csv(path, header=None, names=names)
    except Exception:
        return pd.DataFrame(columns=names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tasks 1–3: m_gg for 2-photon events + photon–jet overlap removal + diagnostics."
    )
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/100_ev",
        help="Directory containing photons_<event>.csv and jets_<event>.csv files",
    )
    parser.add_argument("--n-events", type=int, default=100, help="How many photon files (events) to scan")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument("--out-png", default="mgg_hist.png", help="Output histogram filename")
    parser.add_argument("--out-csv", default="mgg_values.csv", help="Output CSV filename with per-event m_gg")
    parser.add_argument("--dr-overlap", type=float, default=0.4, help="ΔR cut for photon–jet overlap removal")
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

    n_jets_before = 0
    n_jets_after = 0
    n_bad_jets = 0

    # diagnostics (before and after overlap removal)
    min_dr_before_per_event: List[float] = []
    min_dr_after_per_event: List[float] = []

    example_printed = False

    for ev in chosen:
        ph_path = photons_path(args.data_dir, ev)
        if not os.path.exists(ph_path) or os.path.getsize(ph_path) == 0:
            n_bad_ph += 1
            continue

        try:
            # photons have 5 columns; keep only kinematics for downstream code
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

        jet_path = jets_path(args.data_dir, ev)
        if os.path.exists(jet_path) and os.path.getsize(jet_path) > 0:
            try:
                df_j = pd.read_csv(jet_path, header=None, names=["pT", "eta", "phi", "e"])
            except Exception as e:
                print(f"[warn] failed to read {jet_path}: {e}")
                n_bad_jets += 1
                df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])
        else:
            df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])

        n_jets_before += len(df_j)

        if (not example_printed) and (len(df_ph) > 0) and (len(df_j) > 0):
            print(f"[debug] example event {ev}")
            print("[debug] photon row:", df_ph.iloc[0].to_dict())
            print("[debug] jet row:   ", df_j.iloc[0].to_dict())
            example_printed = True

        kept = []

        # diagnostics + overlap removal in one pass
        min_dr_before = np.inf
        for _, jet in df_j.iterrows():
            overlap = False
            for _, pho in df_ph.iterrows():
                dr = delta_r(jet["eta"], jet["phi"], pho["eta"], pho["phi"])
                if dr < min_dr_before:
                    min_dr_before = dr
                if dr < args.dr_overlap:
                    overlap = True
            if not overlap:
                kept.append(jet)

        if np.isfinite(min_dr_before):
            min_dr_before_per_event.append(float(min_dr_before))

        # min ΔR after removal (only over kept jets)
        if len(kept) > 0 and len(df_ph) > 0:
            min_dr_after = np.inf
            for jet in kept:
                for _, pho in df_ph.iterrows():
                    dr = delta_r(jet["eta"], jet["phi"], pho["eta"], pho["phi"])
                    if dr < min_dr_after:
                        min_dr_after = dr
            if np.isfinite(min_dr_after):
                min_dr_after_per_event.append(float(min_dr_after))

        n_jets_after += len(kept)

        if len(df_ph) != 2:
            continue

        n_two_photon += 1
        fv = np.stack([photon_fourvec_from_row(df_ph.iloc[0]), photon_fourvec_from_row(df_ph.iloc[1])], axis=0)
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
    plt.title(r"Di-photon invariant mass for events with exactly 2 photons")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved histogram to {args.out_png}")

    if n_jets_before > 0:
        frac_removed = 1.0 - (n_jets_after / n_jets_before)
        print("Jet–photon overlap removal (ΔR < {:.2f}):".format(args.dr_overlap))
        print(f"  jets before: {n_jets_before}")
        print(f"  jets after:  {n_jets_after}")
        print(f"  fraction removed: {frac_removed:.3f}")
        if n_bad_jets:
            print(f"  skipped jets files (bad): {n_bad_jets}")
    else:
        print("Jet–photon overlap removal: no jets found (n_jets_before = 0).")

    if len(min_dr_before_per_event) > 0:
        arr = np.array(min_dr_before_per_event, dtype=float)
        print("Min ΔR(jet, photon) per event (before removal):")
        print(f"  n events with jets+photons: {len(arr)}")
        print(f"  min / median / max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
        print(f"  fraction with min ΔR < 0.4: {np.mean(arr < 0.4):.3f}")
    else:
        print("Min ΔR(jet, photon) diagnostic (before): no events had both jets and photons.")

    if len(min_dr_after_per_event) > 0:
        arr = np.array(min_dr_after_per_event, dtype=float)
        print("Min ΔR(jet, photon) per event (after removal):")
        print(f"  n events with kept jets+photons: {len(arr)}")
        print(f"  min / median / max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
        print(f"  fraction with min ΔR < 0.4: {np.mean(arr < 0.4):.3f}")
    else:
        print("Min ΔR(jet, photon) diagnostic (after): no events had kept jets and photons.")


if __name__ == "__main__":
    main()