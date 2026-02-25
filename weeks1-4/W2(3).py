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

Step 2 (this week):
  - LHC-style track isolation:
      I(R) = (sum pT of tracks within ΔR < R) / pT(object)
  - plot acceptance vs fake rate (ROC) by scanning a cut on I(R).
  - scan R and choose best cone.

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
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")


# -----------------------------
# step 0: central configuration
# -----------------------------
CFG: Dict[str, object] = {
    "dr_overlap_default": 0.40,  # jet–photon overlap removal (ΔR)
    "dr_track_default": 0.10,    # track counting cone (ΔR)
    "iso_dr_default": 0.20,      # LHC-style isolation cone (ΔR)
}


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


# -----------------------------------------
# step 1: robust ΔR utilities (single truth)
# -----------------------------------------
def delta_phi(phi1: float, phi2: float) -> float:
    """
    Smallest signed separation in φ in [-π, π).
    """
    dphi = phi1 - phi2
    return float((dphi + np.pi) % (2.0 * np.pi) - np.pi)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return float(np.sqrt(deta * deta + dphi * dphi))


# -----------------------------------------
# step 2: invariant mass using 4-momenta
# -----------------------------------------
def pxyz_from_ptetaphi(pt: float, eta: float, phi: float) -> Tuple[float, float, float]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return float(px), float(py), float(pz)


def inv_mass_from_two_objects(obj1: Dict[str, float], obj2: Dict[str, float]) -> float:
    """
    Invariant mass of two objects given dict-like payloads:
      keys: 'pT', 'eta', 'phi', 'e'
    """
    px1, py1, pz1 = pxyz_from_ptetaphi(obj1["pT"], obj1["eta"], obj1["phi"])
    px2, py2, pz2 = pxyz_from_ptetaphi(obj2["pT"], obj2["eta"], obj2["phi"])

    E = obj1["e"] + obj2["e"]
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

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


def track_iso_scalar_sum_pt(
    obj_eta: float,
    obj_phi: float,
    obj_pt: float,
    df_trk: pd.DataFrame,
    dr_max: float,
) -> float:
    """
    LHC-style track isolation:
      I = (sum pT of tracks within ΔR < dr_max) / pT(object)
    """
    if obj_pt <= 0.0:
        return float("nan")
    if len(df_trk) == 0:
        return 0.0

    sum_pt = 0.0
    for _, trk in df_trk.iterrows():
        if delta_r(obj_eta, obj_phi, float(trk["eta"]), float(trk["phi"])) < dr_max:
            sum_pt += float(trk["pT"])
    return float(sum_pt / obj_pt)


def roc_from_iso(ph_iso: np.ndarray, jet_iso: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Scan a cut on isolation; classify photon-like if iso < cut.

    Returns:
      fake_rate (FPR), acceptance (TPR), best_cut, best_J (Youden's J = TPR - FPR)
    """
    ph = ph_iso[np.isfinite(ph_iso)]
    jt = jet_iso[np.isfinite(jet_iso)]
    if len(ph) == 0 or len(jt) == 0:
        return np.array([]), np.array([]), float("nan"), float("nan")

    # use a manageable number of cuts so it's not insane for big samples
    max_iso = float(max(ph.max(), jt.max()))
    cuts = np.linspace(0.0, max_iso, 400)

    tpr = np.zeros_like(cuts, dtype=float)
    fpr = np.zeros_like(cuts, dtype=float)

    for k, c in enumerate(cuts):
        tpr[k] = float(np.mean(ph < c))
        fpr[k] = float(np.mean(jt < c))

    J = tpr - fpr
    best_idx = int(np.argmax(J))
    return fpr, tpr, float(cuts[best_idx]), float(J[best_idx])


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


def _parse_iso_dr_list(arg: Optional[str], fallback_single: float) -> List[float]:
    if arg is None or str(arg).strip() == "":
        return [float(fallback_single)]
    out = []
    for tok in str(arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        return [float(fallback_single)]
    return out


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
    parser.add_argument("--out-dir", default=None, help="Directory to write all outputs (default: <data-dir>/../results_week1)")

    # task 3
    parser.add_argument(
        "--dr-overlap",
        type=float,
        default=float(CFG["dr_overlap_default"]),
        help="ΔR cut for photon–jet overlap removal",
    )

    # task 4 (track counting)
    parser.add_argument(
        "--dr-track",
        type=float,
        default=float(CFG["dr_track_default"]),
        help="ΔR to count tracks near photons/jets",
    )
    parser.add_argument("--tracks-photons-png", default="tracks_near_photons.png", help="Histogram: tracks near photons")
    parser.add_argument("--tracks-jets-png", default="tracks_near_jets.png", help="Histogram: tracks near jets")
    parser.add_argument("--tracks-out-csv", default="track_counts.csv", help="Output CSV with per-object track counts")

    # diagnostics
    parser.add_argument("--mindr-before-png", default="min_dr_jet_photon_before.png")
    parser.add_argument("--mindr-after-png", default="min_dr_jet_photon_after.png")
    parser.add_argument("--tracks-photons-split-png", default="tracks_near_photons_split_conversion.png")

    # LHC strategy isolation + ROC
    parser.add_argument(
        "--iso-dr",
        type=float,
        default=float(CFG["iso_dr_default"]),
        help="Isolation cone ΔR for LHC-style track isolation (ignored if --iso-dr-list is set)",
    )
    parser.add_argument(
        "--iso-dr-list",
        default=None,
        help="Comma-separated isolation ΔR values to scan, e.g. '0.05,0.1,0.2,0.3' (overrides --iso-dr)",
    )
    parser.add_argument("--iso-out-csv", default="iso_values.csv", help="Output CSV with per-object isolation values")
    parser.add_argument("--iso-photons-png", default="iso_photons.png", help="Histogram of isolation for photons (best R)")
    parser.add_argument("--iso-jets-png", default="iso_jets.png", help="Histogram of isolation for jets (best R)")
    parser.add_argument("--roc-png", default="acceptance_vs_fake_rate.png", help="Acceptance vs fake rate (ROC) plot (best R)")
    parser.add_argument(
        "--save-roc-per-r",
        action="store_true",
        help="If set, also write one ROC png per ΔR value (can be a lot).",
    )

    args = parser.parse_args()

    iso_dr_list = _parse_iso_dr_list(args.iso_dr_list, float(args.iso_dr))

    # output folder (default: sibling of data-dir called results_week1)
    if args.out_dir is None:
        out_dir = os.path.abspath(os.path.join(args.data_dir, "..", "results_week1"))
    else:
        out_dir = os.path.abspath(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] writing outputs to: {out_dir}")

    out_csv_path = os.path.join(out_dir, args.out_csv)
    out_png_path = os.path.join(out_dir, args.out_png)

    tracks_out_csv_path = os.path.join(out_dir, args.tracks_out_csv)
    tracks_photons_png_path = os.path.join(out_dir, args.tracks_photons_png)
    tracks_jets_png_path = os.path.join(out_dir, args.tracks_jets_png)

    mindr_before_png_path = os.path.join(out_dir, args.mindr_before_png)
    mindr_after_png_path = os.path.join(out_dir, args.mindr_after_png)
    tracks_photons_split_png_path = os.path.join(out_dir, args.tracks_photons_split_png)

    iso_out_csv_path = os.path.join(out_dir, args.iso_out_csv)
    iso_photons_png_path = os.path.join(out_dir, args.iso_photons_png)
    iso_jets_png_path = os.path.join(out_dir, args.iso_jets_png)
    roc_png_path = os.path.join(out_dir, args.roc_png)

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

    # isolation scan storage
    iso_rows: List[Dict[str, float]] = []
    iso_photons_by_r: Dict[float, List[float]] = {R: [] for R in iso_dr_list}
    iso_jets_by_r: Dict[float, List[float]] = {R: [] for R in iso_dr_list}

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
            )[["pT", "eta", "phi", "e", "conversionType"]]
            n_read += 1
        except Exception as e:
            print(f"[warn] failed to read {ph_path}: {e}")
            n_bad_ph += 1
            continue

        # jets
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

        # tracks
        trk_path = tracks_path(args.data_dir, ev)
        if os.path.exists(trk_path) and os.path.getsize(trk_path) > 0:
            try:
                df_trk = pd.read_csv(
                    trk_path,
                    header=None,
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
            pho_eta = float(pho["eta"])
            pho_phi = float(pho["phi"])
            pho_pt = float(pho["pT"])

            ntrk = count_tracks_near_object(pho_eta, pho_phi, df_trk, args.dr_track)
            photon_track_counts.append(ntrk)
            track_rows.append({
                "event_id": ev,
                "object": "photon",
                "index": int(i),
                "n_tracks": int(ntrk),
                "conversionType": int(pho["conversionType"]),
            })

            # isolation scan
            for R in iso_dr_list:
                iso = track_iso_scalar_sum_pt(pho_eta, pho_phi, pho_pt, df_trk, R)
                iso_photons_by_r[R].append(iso)
                iso_rows.append({
                    "event_id": ev,
                    "object": "photon",
                    "index": int(i),
                    "iso_dr": float(R),
                    "iso": float(iso),
                    "conversionType": int(pho["conversionType"]),
                    "obj_pt": pho_pt,
                })

        for i, jet in enumerate(kept):
            jet_eta = float(jet["eta"])
            jet_phi = float(jet["phi"])
            jet_pt = float(jet["pT"])

            ntrk = count_tracks_near_object(jet_eta, jet_phi, df_trk, args.dr_track)
            jet_track_counts.append(ntrk)
            track_rows.append({
                "event_id": ev,
                "object": "jet",
                "index": int(i),
                "n_tracks": int(ntrk),
                "conversionType": np.nan,
            })

            # isolation scan
            for R in iso_dr_list:
                iso = track_iso_scalar_sum_pt(jet_eta, jet_phi, jet_pt, df_trk, R)
                iso_jets_by_r[R].append(iso)
                iso_rows.append({
                    "event_id": ev,
                    "object": "jet",
                    "index": int(i),
                    "iso_dr": float(R),
                    "iso": float(iso),
                    "conversionType": np.nan,
                    "obj_pt": jet_pt,
                })

        # task 2: mgg for exactly 2 photons
        if len(df_ph) != 2:
            continue

        n_two_photon += 1

        pho1 = {
            "pT": float(df_ph.iloc[0]["pT"]),
            "eta": float(df_ph.iloc[0]["eta"]),
            "phi": float(df_ph.iloc[0]["phi"]),
            "e": float(df_ph.iloc[0]["e"]),
        }
        pho2 = {
            "pT": float(df_ph.iloc[1]["pT"]),
            "eta": float(df_ph.iloc[1]["eta"]),
            "phi": float(df_ph.iloc[1]["phi"]),
            "e": float(df_ph.iloc[1]["e"]),
        }

        mgg = inv_mass_from_two_objects(pho1, pho2)
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

    out.to_csv(out_csv_path, index=False)
    print(f"Saved per-event m_gg to {out_csv_path}")

    plt.figure()
    plt.hist(out["m_gg"].to_numpy(dtype=float), bins=args.bins)
    plt.xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    plt.ylabel("Entries")
    plt.title(r"Di-photon invariant mass for events with exactly 2 photons")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    print(f"Saved histogram to {out_png_path}")

    # task 3 summary + plots
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
        print(f"  fraction with min ΔR < {args.dr_overlap:.2f}: {np.mean(arr < args.dr_overlap):.3f}")
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(r"min $\Delta R$(jet, photon) per event")
        plt.ylabel("Entries")
        plt.title(r"Per-event min $\Delta R$(jet, photon) before overlap removal")
        plt.tight_layout()
        plt.savefig(mindr_before_png_path, dpi=200)
        print(f"Saved min ΔR (before) histogram to {mindr_before_png_path}")

    if len(min_dr_after_per_event) > 0:
        arr = np.array(min_dr_after_per_event, dtype=float)
        print("Min ΔR(jet, photon) per event (after removal):")
        print(f"  n events with kept jets+photons: {len(arr)}")
        print(f"  min / median / max: {arr.min():.3f} / {np.median(arr):.3f} / {arr.max():.3f}")
        print(f"  fraction with min ΔR < {args.dr_overlap:.2f}: {np.mean(arr < args.dr_overlap):.3f}")
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(r"min $\Delta R$(jet, photon) per event")
        plt.ylabel("Entries")
        plt.title(r"Per-event min $\Delta R$(jet, photon) after overlap removal")
        plt.tight_layout()
        plt.savefig(mindr_after_png_path, dpi=200)
        print(f"Saved min ΔR (after) histogram to {mindr_after_png_path}")

    # task 4 outputs: track-count hists + conversion split
    track_df = pd.DataFrame(track_rows)
    ph_df = track_df[track_df["object"] == "photon"].copy()
    unconv = ph_df[ph_df["conversionType"] == 0]["n_tracks"].to_numpy(dtype=int)
    conv = ph_df[ph_df["conversionType"] != 0]["n_tracks"].to_numpy(dtype=int)

    if (len(unconv) > 0) or (len(conv) > 0):
        max_n = 0
        if len(unconv) > 0:
            max_n = max(max_n, int(unconv.max()))
        if len(conv) > 0:
            max_n = max(max_n, int(conv.max()))
        bins = range(0, max_n + 2)

        plt.figure()
        if len(unconv) > 0:
            plt.hist(unconv, bins=bins, alpha=0.6, label="unconverted (conversionType=0)")
        if len(conv) > 0:
            plt.hist(conv, bins=bins, alpha=0.6, label="converted (conversionType!=0)")
        plt.xlabel(f"# tracks within ΔR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("Tracks near photons split by conversionType")
        plt.legend()
        plt.tight_layout()
        plt.savefig(tracks_photons_split_png_path, dpi=200)
        print(f"Saved conversion-split photon histogram to {tracks_photons_split_png_path}")

    track_df.to_csv(tracks_out_csv_path, index=False)
    print(f"Saved per-object track counts to {tracks_out_csv_path}")

    if len(photon_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(photon_track_counts, dtype=int), bins=range(0, max(photon_track_counts) + 2))
        plt.xlabel(f"# tracks within ΔR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("Tracks near photons")
        plt.tight_layout()
        plt.savefig(tracks_photons_png_path, dpi=200)
        print(f"Saved histogram to {tracks_photons_png_path}")
    else:
        print("[warn] no photon objects found for track counting")

    if len(jet_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(jet_track_counts, dtype=int), bins=range(0, max(jet_track_counts) + 2))
        plt.xlabel(f"# tracks within ΔR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("Tracks near jets (after overlap removal)")
        plt.tight_layout()
        plt.savefig(tracks_jets_png_path, dpi=200)
        print(f"Saved histogram to {tracks_jets_png_path}")
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

    # Step 2: isolation outputs + choose best R by max J
    iso_df = pd.DataFrame(iso_rows)
    iso_df.to_csv(iso_out_csv_path, index=False)
    print(f"Saved per-object isolation values to {iso_out_csv_path}")

    best_R = None
    best_cut = float("nan")
    best_J = -np.inf
    best_fpr = None
    best_tpr = None

    for R in iso_dr_list:
        ph_iso = np.array(iso_photons_by_r[R], dtype=float)
        jt_iso = np.array(iso_jets_by_r[R], dtype=float)

        fpr, tpr, cut, J = roc_from_iso(ph_iso, jt_iso)
        if len(fpr) == 0:
            continue

        if args.save_roc_per_r:
            roc_path = os.path.join(out_dir, f"roc_iso_dr_{R:.3f}.png")
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("Fake rate (jets passing cut)")
            plt.ylabel("Acceptance (photons passing cut)")
            plt.title(rf"ROC (iso cut), $\Delta R<{R}$")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=200)

        if J > best_J:
            best_J = float(J)
            best_R = float(R)
            best_cut = float(cut)
            best_fpr = fpr
            best_tpr = tpr

    if best_R is None or best_fpr is None or best_tpr is None:
        print("[warn] could not compute isolation ROC for any R (need finite iso values for both photons and jets)")
        return

    # save BEST ROC
    plt.figure()
    plt.plot(best_fpr, best_tpr)
    plt.xlabel("Fake rate (jets passing cut)")
    plt.ylabel("Acceptance (photons passing cut)")
    plt.title(rf"Acceptance vs Fake rate (iso cut), best $\Delta R<{best_R}$")
    plt.tight_layout()
    plt.savefig(roc_png_path, dpi=200)
    print(f"Saved ROC to {roc_png_path}")

    # save BEST histograms
    ph_best = np.array(iso_photons_by_r[best_R], dtype=float)
    jt_best = np.array(iso_jets_by_r[best_R], dtype=float)

    plt.figure()
    plt.hist(ph_best[np.isfinite(ph_best)], bins=60)
    plt.xlabel(rf"$I$ (track isolation), $\Delta R<{best_R}$")
    plt.ylabel("Entries")
    plt.title("Track isolation for photons (best R)")
    plt.tight_layout()
    plt.savefig(iso_photons_png_path, dpi=200)
    print(f"Saved photon isolation histogram to {iso_photons_png_path}")

    plt.figure()
    plt.hist(jt_best[np.isfinite(jt_best)], bins=60)
    plt.xlabel(rf"$I$ (track isolation), $\Delta R<{best_R}$")
    plt.ylabel("Entries")
    plt.title("Track isolation for jets (after overlap removal, best R)")
    plt.tight_layout()
    plt.savefig(iso_jets_png_path, dpi=200)
    print(f"Saved jet isolation histogram to {iso_jets_png_path}")

    # summary numbers at best cut
    ph_acc = float(np.mean(ph_best[np.isfinite(ph_best)] < best_cut))
    jt_fake = float(np.mean(jt_best[np.isfinite(jt_best)] < best_cut))

    print("Iso ROC optimisation (classify photon if iso < cut):")
    print(f"  scanned R: {', '.join([f'{x:g}' for x in iso_dr_list])}")
    print(f"  best R: {best_R:.3f}")
    print(f"  best cut (max TPR-FPR): {best_cut:.6g}")
    print(f"  best J = TPR-FPR: {best_J:.3f}")
    print(f"  at best cut: acceptance = {ph_acc:.3f}, fake rate = {jt_fake:.3f}, jet rejection = {1.0 - jt_fake:.3f}")


if __name__ == "__main__":
    main()

