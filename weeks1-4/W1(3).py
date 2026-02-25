#!/usr/bin/env python3
"""
Tasks 1–4 for HL-LHC diphoton toy data.

Task 1:
  - take ~N events (files) from a directory containing photons_<event>.csv etc.

Task 2:
  - in events with exactly 2 photons, compute and plot m_yy (m_gg).

Task 3:
  - photon–jet overlap removal (remove jets within ΔR < 0.4 of any photon),
    and monitor efficiency (fraction of jets removed).
  - diagnostic: record min ΔR(jet, photon) per event and summarise.

Task 4:
  - count how many tracks are close to a) photons and b) jets
  - plot #entries vs #tracks for photons and jets
  - report a simple separation “efficiency” by scanning a threshold on #tracks

Assumed file formats (as observed):
  photons_<ev>.csv: pT, eta, phi, e, conversionType   (comma-separated, no header)
  jets_<ev>.csv:    pT, eta, phi, e                   (comma-separated, no header)
  tracks_<ev>.csv:  pT, eta, phi, eTot, z0, d0         (comma-separated, no header)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")


def find_event_ids(data_dir: str) -> List[int]:
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


def tracks_path(data_dir: str, event_id: int) -> str:
    return os.path.join(data_dir, f"tracks_{event_id}.csv")


def read_csv_maybe_gzip(path: str, *, names: List[str]) -> pd.DataFrame:
    """
    Read a headerless CSV that may actually be gzipped (sometimes mislabelled .csv).
    Detect gzip by magic bytes 0x1f 0x8b.
    """
    p = Path(path)
    with p.open("rb") as f:
        magic = f.read(2)

    is_gzip = (magic == b"\x1f\x8b")
    return pd.read_csv(
        path,
        header=None,
        names=names,
        compression="gzip" if is_gzip else None,
    )


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
    pt = float(row["pT"])
    eta = float(row["eta"])
    phi = float(row["phi"])
    E = float(row["e"])
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.array([E, px, py, pz], dtype=float)


def invariant_mass(fourvecs: np.ndarray) -> float:
    total = np.sum(fourvecs, axis=0)
    E, px, py, pz = total
    m2 = E * E - (px * px + py * py + pz * pz)
    return float(np.sqrt(max(m2, 0.0)))


def count_tracks_near_object(
    obj_eta: float,
    obj_phi: float,
    df_trk: pd.DataFrame,
    dr_max: float,
) -> int:
    if len(df_trk) == 0:
        return 0
    n = 0
    for _, trk in df_trk.iterrows():
        if delta_r(obj_eta, obj_phi, float(trk["eta"]), float(trk["phi"])) < dr_max:
            n += 1
    return n


def best_threshold_separation(
    photon_counts: List[int],
    jet_counts: List[int],
) -> Tuple[int, float, float, float]:
    """
    Scan integer threshold t, classify as 'photon-like' if n_tracks <= t.
    Returns: (best_t, photon_eff, jet_misid, accuracy)
    """
    if len(photon_counts) == 0 or len(jet_counts) == 0:
        return 0, float("nan"), float("nan"), float("nan")

    p = np.array(photon_counts, dtype=int)
    j = np.array(jet_counts, dtype=int)
    max_c = int(max(p.max(), j.max()))
    best = (-1.0, 0, 0.0, 1.0)  # (accuracy, t, photon_eff, jet_misid)

    for t in range(0, max_c + 1):
        photon_eff = float(np.mean(p <= t))
        jet_misid = float(np.mean(j <= t))
        accuracy = float((np.sum(p <= t) + np.sum(j > t)) / (len(p) + len(j)))
        if accuracy > best[0]:
            best = (accuracy, t, photon_eff, jet_misid)

    accuracy, t, photon_eff, jet_misid = best
    return int(t), float(photon_eff), float(jet_misid), float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tasks 1–4: m_gg (2-photon events), photon–jet overlap removal, and track counts near photons/jets."
    )
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/1k_ev",
        help="Directory containing photons_<event>.csv, jets_<event>.csv, tracks_<event>.csv",
    )
    parser.add_argument("--n-events", type=int, default=1000, help="How many photon files (events) to scan")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins for mgg")
    parser.add_argument("--out-png", default="mgg_hist.png", help="Output mgg histogram filename")
    parser.add_argument("--out-csv", default="mgg_values.csv", help="Output CSV filename with per-event m_gg")
    parser.add_argument("--dr-overlap", type=float, default=0.4, help="ΔR cut for photon–jet overlap removal")

    # task 4
    parser.add_argument("--dr-track", type=float, default=0.1, help="ΔR to count tracks near photons/jets")
    parser.add_argument("--tracks-photons-png", default="tracks_near_photons.png", help="Histogram: tracks near photons")
    parser.add_argument("--tracks-jets-png", default="tracks_near_jets.png", help="Histogram: tracks near jets")
    parser.add_argument("--tracks-out-csv", default="track_counts.csv", help="Output CSV with per-object track counts")
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

    # task 2
    mgg_rows: List[Dict[str, float]] = []
    n_read = 0
    n_two_photon = 0
    n_bad_ph = 0

    # task 3
    n_jets_before = 0
    n_jets_after = 0
    n_bad_jets = 0
    min_dr_before_per_event: List[float] = []
    min_dr_after_per_event: List[float] = []

    # task 4
    photon_track_counts: List[int] = []
    jet_track_counts: List[int] = []
    track_rows: List[Dict[str, float]] = []

    example_printed = False

    for ev in chosen:
        ph_path = photons_path(args.data_dir, ev)
        if not os.path.exists(ph_path) or os.path.getsize(ph_path) == 0:
            n_bad_ph += 1
            continue

        try:
            df_ph = read_csv_maybe_gzip(
                ph_path,
                names=["pT", "eta", "phi", "e", "conversionType"],
            )[["pT", "eta", "phi", "e"]]
            n_read += 1
        except Exception as e:
            print(f"[warn] failed to read {ph_path}: {e}")
            n_bad_ph += 1
            continue

        # jets
        jet_path = jets_path(args.data_dir, ev)
        if os.path.exists(jet_path) and os.path.getsize(jet_path) > 0:
            try:
                df_j = read_csv_maybe_gzip(
                    jet_path,
                    names=["pT", "eta", "phi", "e"],
                )
            except Exception as e:
                print(f"[warn] failed to read {jet_path}: {e}")
                n_bad_jets += 1
                df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])
        else:
            df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])

        # tracks
        trk_path = tracks_path(args.data_dir, ev)
        if os.path.exists(trk_path) and os.path.getsize(trk_path) > 0:
            try:
                df_trk = read_csv_maybe_gzip(
                    trk_path,
                    names=["pT", "eta", "phi", "eTot", "z0", "d0"],
                )
            except Exception as e:
                print(f"[warn] failed to read {trk_path}: {e}")
                df_trk = pd.DataFrame(columns=["pT", "eta", "phi", "eTot", "z0", "d0"])
        else:
            df_trk = pd.DataFrame(columns=["pT", "eta", "phi", "eTot", "z0", "d0"])

        n_jets_before += len(df_j)

        if (not example_printed) and (len(df_ph) > 0) and (len(df_j) > 0):
            print(f"[debug] example event {ev}")
            print("[debug] photon row:", df_ph.iloc[0].to_dict())
            print("[debug] jet row:   ", df_j.iloc[0].to_dict())
            example_printed = True

        # task 3: overlap removal + diagnostics
        kept = []
        min_dr_before = np.inf
        for _, jet in df_j.iterrows():
            overlap = False
            for _, pho in df_ph.iterrows():
                dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                if dr < min_dr_before:
                    min_dr_before = dr
                if dr < args.dr_overlap:
                    overlap = True
            if not overlap:
                kept.append(jet)

        if np.isfinite(min_dr_before):
            min_dr_before_per_event.append(float(min_dr_before))

        if len(kept) > 0 and len(df_ph) > 0:
            min_dr_after = np.inf
            for jet in kept:
                for _, pho in df_ph.iterrows():
                    dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                    if dr < min_dr_after:
                        min_dr_after = dr
            if np.isfinite(min_dr_after):
                min_dr_after_per_event.append(float(min_dr_after))

        n_jets_after += len(kept)

        # task 4: track counts near photons and near jets (use kept jets)
        for i, pho in df_ph.iterrows():
            ntrk = count_tracks_near_object(float(pho["eta"]), float(pho["phi"]), df_trk, args.dr_track)
            photon_track_counts.append(ntrk)
            track_rows.append({"event_id": ev, "object": "photon", "index": int(i), "n_tracks": int(ntrk)})

        for i, jet in enumerate(kept):
            ntrk = count_tracks_near_object(float(jet["eta"]), float(jet["phi"]), df_trk, args.dr_track)
            jet_track_counts.append(ntrk)
            track_rows.append({"event_id": ev, "object": "jet", "index": int(i), "n_tracks": int(ntrk)})

        # task 2: mgg for exactly 2 photons
        if len(df_ph) != 2:
            continue

        n_two_photon += 1
        fv = np.stack(
            [photon_fourvec_from_row(df_ph.iloc[0]), photon_fourvec_from_row(df_ph.iloc[1])],
            axis=0,
        )
        mgg = invariant_mass(fv)
        mgg_rows.append({"event_id": ev, "m_gg": mgg})

    # task 2 outputs
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

    # task 3 summary
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
        print(f"  fraction with min ΔR < {args.dr_overlap:.1f}: {np.mean(arr < args.dr_overlap):.3f}")

    if len(min_dr_after_per_event) > 0:
        arr = np.array(min_dr_after_per_event, dtype=float)
        print("Min ΔR(jet, photon) per event (after removal):")
        print(f"  n events with kept jets+photons: {len(arr)}")
        print(f"  min / median / max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
        print(f"  fraction with min ΔR < {args.dr_overlap:.1f}: {np.mean(arr < args.dr_overlap):.3f}")

    # task 4 outputs: histograms + simple separation scan
    track_df = pd.DataFrame(track_rows)
    track_df.to_csv(args.tracks_out_csv, index=False)
    print(f"Saved per-object track counts to {args.tracks_out_csv}")

    if len(photon_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(photon_track_counts, dtype=int), bins=range(0, max(photon_track_counts) + 2))
        plt.xlabel(f"# tracks within ΔR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("Tracks near photons")
        plt.tight_layout()
        plt.savefig(args.tracks_photons_png, dpi=200)
        print(f"Saved histogram to {args.tracks_photons_png}")
    else:
        print("[warn] no photon objects found for track counting")

    if len(jet_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(jet_track_counts, dtype=int), bins=range(0, max(jet_track_counts) + 2))
        plt.xlabel(f"# tracks within ΔR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("Tracks near jets (after overlap removal)")
        plt.tight_layout()
        plt.savefig(args.tracks_jets_png, dpi=200)
        print(f"Saved histogram to {args.tracks_jets_png}")
    else:
        print("[warn] no jet objects found for track counting")

    t, ph_eff, jet_misid, acc = best_threshold_separation(photon_track_counts, jet_track_counts)
    if np.isfinite(acc):
        print(f"Track-count separation (ΔR_track < {args.dr_track}): classify photon if n_tracks <= t")
        print(f"  best threshold t = {t}")
        print(f"  photon efficiency (kept as photon): {ph_eff:.3f}")
        print(f"  jet mis-id rate (jet tagged as photon): {jet_misid:.3f}")
        print(f"  jet rejection: {1.0 - jet_misid:.3f}")
        print(f"  overall accuracy: {acc:.3f}")
    else:
        print("[warn] could not compute separation threshold (need both photon and jet counts)")


if __name__ == "__main__":
    main()
