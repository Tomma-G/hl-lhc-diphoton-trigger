#!/usr/bin/env python3
"""
HL-LHC diphoton toy analysis.

Inputs per event (no headers):
  photons_<id>.csv: pT, eta, phi, e, conversionType
  jets_<id>.csv:    pT, eta, phi, e
  tracks_<id>.csv:  pT, eta, phi, eTot, z0, d0

Outputs:
  - m_gg CSV + histogram (2-photon events only)
  - jet–photon overlap removal summary + min-dR histograms
  - per-object track counts + histograms
  - track isolation (scalar sum pT in cone / object pT) + ROC
  - track_reduction.csv (Step C): kept_tracks / all_tracks
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PHOTON_RE = re.compile(r"^photons_(\d+)\.csv$")

CFG: Dict[str, float] = {
    "dr_overlap_default": 0.40,
    "dr_track_default": 0.10,
    "iso_dr_default": 0.30,
    "dr_track_keep_default": 0.40,
}


def info(msg: str) -> None:
    # avoid Windows cp1252 console crashes if Unicode sneaks in
    try:
        print(f"[info] {msg}")
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(f"[info] {safe}")


def warn(msg: str) -> None:
    # keep warn ASCII-only by convention, but make it safe anyway
    try:
        print(f"[warn] {msg}")
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(f"[warn] {safe}")


def err(msg: str) -> None:
    try:
        print(f"[error] {msg}", file=sys.stderr)
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8")
        print(f"[error] {safe}", file=sys.stderr)


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


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    return float((dphi + np.pi) % (2.0 * np.pi) - np.pi)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return float(np.sqrt(deta * deta + dphi * dphi))


def pxyz_from_ptetaphi(pt: float, eta: float, phi: float) -> Tuple[float, float, float]:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return float(px), float(py), float(pz)


def inv_mass_from_two_objects(obj1: Dict[str, float], obj2: Dict[str, float]) -> float:
    px1, py1, pz1 = pxyz_from_ptetaphi(obj1["pT"], obj1["eta"], obj1["phi"])
    px2, py2, pz2 = pxyz_from_ptetaphi(obj2["pT"], obj2["eta"], obj2["phi"])

    E = obj1["e"] + obj2["e"]
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = E * E - (px * px + py * py + pz * pz)
    return float(np.sqrt(max(m2, 0.0)))


def count_tracks_near_object(obj_eta: float, obj_phi: float, df_trk: pd.DataFrame, dr_max: float) -> int:
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
    if obj_pt <= 0.0:
        return float("nan")
    if len(df_trk) == 0:
        return 0.0

    sum_pt = 0.0
    for _, trk in df_trk.iterrows():
        if delta_r(obj_eta, obj_phi, float(trk["eta"]), float(trk["phi"])) < dr_max:
            sum_pt += float(trk["pT"])
    return float(sum_pt / obj_pt)


def filter_tracks_close_to_photons(df_trk: pd.DataFrame, df_ph: pd.DataFrame, dr_keep: float) -> pd.DataFrame:
    """keep tracks with min dR(track, any photon) < dr_keep."""
    if len(df_trk) == 0 or len(df_ph) == 0:
        return df_trk

    trk_eta = df_trk["eta"].to_numpy(dtype=float)
    trk_phi = df_trk["phi"].to_numpy(dtype=float)

    keep_mask = np.zeros(len(df_trk), dtype=bool)
    for _, pho in df_ph.iterrows():
        pho_eta = float(pho["eta"])
        pho_phi = float(pho["phi"])
        deta = trk_eta - pho_eta
        dphi = (trk_phi - pho_phi + np.pi) % (2.0 * np.pi) - np.pi
        dr = np.sqrt(deta * deta + dphi * dphi)
        keep_mask |= (dr < float(dr_keep))

    return df_trk.loc[keep_mask].reset_index(drop=True)


def roc_from_iso(ph_iso: np.ndarray, jet_iso: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """scan cut on iso; classify photon-like if iso < cut. returns fpr, tpr, best_cut, best_J."""
    ph = ph_iso[np.isfinite(ph_iso)].astype(float)
    jt = jet_iso[np.isfinite(jet_iso)].astype(float)
    if ph.size == 0 or jt.size == 0:
        return np.array([]), np.array([]), float("nan"), float("nan")

    vals = np.unique(np.concatenate([ph, jt]))
    vals.sort()
    if vals.size == 1:
        c = float(vals[0])
        tpr = float(np.mean(ph < c))
        fpr = float(np.mean(jt < c))
        return np.array([fpr]), np.array([tpr]), c, float(tpr - fpr)

    mids = (vals[:-1] + vals[1:]) * 0.5
    cuts_plot = np.concatenate(([vals[0] - 1.0], mids, [vals[-1] + 1.0]))

    tpr_plot = np.array([float(np.mean(ph < c)) for c in cuts_plot], dtype=float)
    fpr_plot = np.array([float(np.mean(jt < c)) for c in cuts_plot], dtype=float)

    tpr_m = np.array([float(np.mean(ph < c)) for c in mids], dtype=float)
    fpr_m = np.array([float(np.mean(jt < c)) for c in mids], dtype=float)
    J = tpr_m - fpr_m
    best_idx = int(np.argmax(J))

    return fpr_plot, tpr_plot, float(mids[best_idx]), float(J[best_idx])


def best_threshold_separation(photon_counts: List[int], jet_counts: List[int]) -> Tuple[int, float, float, float]:
    """classify photon-like if n_tracks <= t. returns t, photon_eff, jet_misid, accuracy."""
    if len(photon_counts) == 0 or len(jet_counts) == 0:
        return 0, float("nan"), float("nan"), float("nan")

    p = np.array(photon_counts, dtype=int)
    j = np.array(jet_counts, dtype=int)
    max_c = int(max(p.max(), j.max()))

    best = (-1.0, 0, 0.0, 1.0)
    for t in range(0, max_c + 1):
        photon_eff = float(np.mean(p <= t))
        jet_misid = float(np.mean(j <= t))
        accuracy = float((np.sum(p <= t) + np.sum(j > t)) / (len(p) + len(j)))
        if accuracy > best[0]:
            best = (accuracy, t, photon_eff, jet_misid)

    accuracy, t, photon_eff, jet_misid = best
    return int(t), float(photon_eff), float(jet_misid), float(accuracy)


def read_csv_maybe_gzip(path: str, **kwargs) -> pd.DataFrame:
    """read csv; auto-detect gzip via magic bytes."""
    with open(path, "rb") as f:
        head = f.read(2)
    is_gz = (len(head) == 2 and head[0] == 0x1F and head[1] == 0x8B)
    if is_gz:
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/1k_ev",
    )
    parser.add_argument("--n-events", type=int, default=100)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--out-png", default="mgg_hist.png")
    parser.add_argument("--out-csv", default="mgg_values.csv")
    parser.add_argument("--out-dir", default=None)

    parser.add_argument(
        "--include-converted",
        action="store_true",
        default=False,
        help="include converted photons (default: exclude them)",
    )

    parser.add_argument("--dr-overlap", type=float, default=float(CFG["dr_overlap_default"]))
    parser.add_argument("--dr-track", type=float, default=float(CFG["dr_track_default"]))
    parser.add_argument("--dr-track-keep", type=float, default=float(CFG["dr_track_keep_default"]))

    parser.add_argument("--tracks-photons-png", default="tracks_near_photons.png")
    parser.add_argument("--tracks-jets-png", default="tracks_near_jets.png")
    parser.add_argument("--tracks-out-csv", default="track_counts.csv")

    parser.add_argument("--mindr-before-png", default="min_dr_jet_photon_before.png")
    parser.add_argument("--mindr-after-png", default="min_dr_jet_photon_after.png")
    parser.add_argument("--tracks-photons-split-png", default="tracks_near_photons_split_conversion.png")

    parser.add_argument("--iso-dr", type=float, default=float(CFG["iso_dr_default"]))
    parser.add_argument("--iso-out-csv", default="iso_values.csv")
    parser.add_argument("--iso-photons-png", default="iso_photons.png")
    parser.add_argument("--iso-jets-png", default="iso_jets.png")
    parser.add_argument("--roc-png", default="acceptance_vs_fake_rate.png")

    args = parser.parse_args()
    exclude_converted = not args.include_converted

    out_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir is not None
        else os.path.abspath(os.path.join(args.data_dir, "..", "results_week1"))
    )
    os.makedirs(out_dir, exist_ok=True)
    info(f"writing outputs to: {out_dir}")

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
        err(f"data-dir does not exist: {args.data_dir}")
        sys.exit(1)

    event_ids = find_event_ids(args.data_dir)
    if not event_ids:
        err(f"no photons_<id>.csv found in {args.data_dir}")
        sys.exit(1)

    chosen = event_ids[: args.n_events]
    info(f"found {len(event_ids)} photon files; processing {len(chosen)} events")
    info(f"photon conversion handling: {'excluding' if exclude_converted else 'including'} converted photons")
    info(f"track preselection: keep tracks with min dR(track, photon) < {args.dr_track_keep}")

    mgg_rows: List[Dict[str, float]] = []

    n_read = 0
    n_two_photon = 0
    n_bad_ph = 0

    n_jets_before = 0
    n_jets_after = 0
    n_bad_jets = 0

    min_dr_before_per_event: List[float] = []
    min_dr_after_per_event: List[float] = []

    photon_track_counts: List[int] = []
    jet_track_counts: List[int] = []
    track_rows: List[Dict[str, float]] = []

    iso_rows: List[Dict[str, float]] = []
    photon_iso_vals: List[float] = []
    jet_iso_vals: List[float] = []

    n_trk_all_total = 0
    n_trk_phclose_total = 0

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
            warn(f"failed to read {ph_path}: {e}")
            n_bad_ph += 1
            continue

        if exclude_converted:
            conv_int = pd.to_numeric(df_ph["conversionType"], errors="coerce").fillna(-999).astype(int)
            df_ph = df_ph[conv_int == 0].reset_index(drop=True)

        jet_p = jets_path(args.data_dir, ev)
        if os.path.exists(jet_p) and os.path.getsize(jet_p) > 0:
            try:
                df_j = read_csv_maybe_gzip(jet_p, header=None, names=["pT", "eta", "phi", "e"])
            except Exception as e:
                warn(f"failed to read {jet_p}: {e}")
                n_bad_jets += 1
                df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])
        else:
            df_j = pd.DataFrame(columns=["pT", "eta", "phi", "e"])

        trk_p = tracks_path(args.data_dir, ev)
        if os.path.exists(trk_p) and os.path.getsize(trk_p) > 0:
            try:
                df_trk_all = pd.read_csv(
                    trk_p,
                    header=None,
                    names=["pT", "eta", "phi", "eTot", "z0", "d0"],
                )
            except Exception as e:
                warn(f"failed to read {trk_p}: {e}")
                df_trk_all = pd.DataFrame(columns=["pT", "eta", "phi", "eTot", "z0", "d0"])
        else:
            df_trk_all = pd.DataFrame(columns=["pT", "eta", "phi", "eTot", "z0", "d0"])

        n_trk_all_total += len(df_trk_all)
        df_trk_phclose = filter_tracks_close_to_photons(df_trk_all, df_ph, float(args.dr_track_keep))
        n_trk_phclose_total += len(df_trk_phclose)

        n_jets_before += len(df_j)

        if (not example_printed) and (len(df_ph) > 0) and (len(df_j) > 0):
            info(f"example event {ev}")
            info(f"photon row: {df_ph.iloc[0].to_dict()}")
            info(f"jet row:    {df_j.iloc[0].to_dict()}")
            example_printed = True

        kept = []
        min_dr_before = np.inf
        for _, jet in df_j.iterrows():
            overlap = False
            for _, pho in df_ph.iterrows():
                dr = delta_r(float(jet["eta"]), float(jet["phi"]), float(pho["eta"]), float(pho["phi"]))
                min_dr_before = min(min_dr_before, dr)
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
                    min_dr_after = min(min_dr_after, dr)
            if np.isfinite(min_dr_after):
                min_dr_after_per_event.append(float(min_dr_after))

        n_jets_after += len(kept)

        # photons: use photon-close tracks
        for i, pho in df_ph.iterrows():
            pho_eta = float(pho["eta"])
            pho_phi = float(pho["phi"])
            pho_pt = float(pho["pT"])

            ntrk = count_tracks_near_object(pho_eta, pho_phi, df_trk_phclose, args.dr_track)
            photon_track_counts.append(ntrk)

            conv_val = pd.to_numeric(pho["conversionType"], errors="coerce")
            conv_int = int(conv_val) if pd.notna(conv_val) else -999

            track_rows.append(
                {"event_id": ev, "object": "photon", "index": int(i), "n_tracks": int(ntrk), "conversionType": conv_int}
            )

            iso = track_iso_scalar_sum_pt(pho_eta, pho_phi, pho_pt, df_trk_phclose, args.iso_dr)
            photon_iso_vals.append(iso)
            iso_rows.append(
                {
                    "event_id": ev,
                    "object": "photon",
                    "index": int(i),
                    "iso_dr": float(args.iso_dr),
                    "iso": float(iso),
                    "conversionType": conv_int,
                    "obj_pt": pho_pt,
                }
            )

        # jets: use all tracks
        for i, jet in enumerate(kept):
            jet_eta = float(jet["eta"])
            jet_phi = float(jet["phi"])
            jet_pt = float(jet["pT"])

            ntrk = count_tracks_near_object(jet_eta, jet_phi, df_trk_all, args.dr_track)
            jet_track_counts.append(ntrk)
            track_rows.append(
                {"event_id": ev, "object": "jet", "index": int(i), "n_tracks": int(ntrk), "conversionType": np.nan}
            )

            iso = track_iso_scalar_sum_pt(jet_eta, jet_phi, jet_pt, df_trk_all, args.iso_dr)
            jet_iso_vals.append(iso)
            iso_rows.append(
                {
                    "event_id": ev,
                    "object": "jet",
                    "index": int(i),
                    "iso_dr": float(args.iso_dr),
                    "iso": float(iso),
                    "conversionType": np.nan,
                    "obj_pt": jet_pt,
                }
            )

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
        mgg_rows.append({"event_id": ev, "m_gg": inv_mass_from_two_objects(pho1, pho2)})

    # Step C summary + write track_reduction.csv (ONCE)
    if n_trk_all_total > 0:
        frac = float(n_trk_phclose_total) / float(n_trk_all_total)
        info(f"track preselection kept {n_trk_phclose_total}/{n_trk_all_total} = {frac:.3f}")

        red_path = os.path.join(out_dir, "track_reduction.csv")
        pd.DataFrame(
            [
                {
                    "kept_tracks": int(n_trk_phclose_total),
                    "all_tracks": int(n_trk_all_total),
                    "kept_fraction": float(frac),
                    "dr_track_keep": float(args.dr_track_keep),
                }
            ]
        ).to_csv(red_path, index=False)
        info(f"wrote {red_path}")

    if not mgg_rows:
        err("no valid 2-photon events found")
        sys.exit(1)

    out = pd.DataFrame(mgg_rows)
    info(f"read photon files: {n_read} (skipped bad/empty: {n_bad_ph})")
    info(f"2-photon events: {n_two_photon}")

    out.to_csv(out_csv_path, index=False)
    info(f"wrote {out_csv_path}")

    plt.figure()
    plt.hist(out["m_gg"].to_numpy(dtype=float), bins=args.bins)
    plt.xlabel(r"$m_{\gamma\gamma}$ [GeV]")
    plt.ylabel("Entries")
    plt.title(r"Di-photon invariant mass (events with exactly 2 photons)")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    info(f"wrote {out_png_path}")

    if n_jets_before > 0:
        frac_removed = 1.0 - (n_jets_after / n_jets_before)
        info(
            f"jet-photon overlap removal (dR<{args.dr_overlap:.2f}): "
            f"{n_jets_before} -> {n_jets_after} (removed {frac_removed:.3f})"
        )
        if n_bad_jets:
            info(f"bad jets files: {n_bad_jets}")
    else:
        info("jet-photon overlap removal: no jets found")

    if len(min_dr_before_per_event) > 0:
        arr = np.array(min_dr_before_per_event, dtype=float)
        info(
            f"min dR(jet,photon) before: n={len(arr)} min/med/max="
            f"{arr.min():.3f}/{np.median(arr):.3f}/{arr.max():.3f}"
        )
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(r"min $\Delta R$(jet, photon) per event")
        plt.ylabel("Entries")
        plt.title(r"min $\Delta R$(jet, photon) before overlap removal")
        plt.tight_layout()
        plt.savefig(mindr_before_png_path, dpi=200)
        info(f"wrote {mindr_before_png_path}")

    if len(min_dr_after_per_event) > 0:
        arr = np.array(min_dr_after_per_event, dtype=float)
        info(
            f"min dR(jet,photon) after: n={len(arr)} min/med/max="
            f"{arr.min():.3f}/{np.median(arr):.3f}/{arr.max():.3f}"
        )
        plt.figure()
        plt.hist(arr, bins=60)
        plt.xlabel(r"min $\Delta R$(jet, photon) per event")
        plt.ylabel("Entries")
        plt.title(r"min $\Delta R$(jet, photon) after overlap removal")
        plt.tight_layout()
        plt.savefig(mindr_after_png_path, dpi=200)
        info(f"wrote {mindr_after_png_path}")

    track_df = pd.DataFrame(track_rows)

    if args.include_converted:
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
            plt.xlabel(f"# tracks within dR < {args.dr_track}")
            plt.ylabel("Entries")
            plt.title("tracks near photons split by conversionType")
            plt.legend()
            plt.tight_layout()
            plt.savefig(tracks_photons_split_png_path, dpi=200)
            info(f"wrote {tracks_photons_split_png_path}")

    track_df.to_csv(tracks_out_csv_path, index=False)
    info(f"wrote {tracks_out_csv_path}")

    if len(photon_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(photon_track_counts, dtype=int), bins=range(0, max(photon_track_counts) + 2))
        plt.xlabel(f"# tracks within dR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("tracks near photons")
        plt.tight_layout()
        plt.savefig(tracks_photons_png_path, dpi=200)
        info(f"wrote {tracks_photons_png_path}")

    if len(jet_track_counts) > 0:
        plt.figure()
        plt.hist(np.array(jet_track_counts, dtype=int), bins=range(0, max(jet_track_counts) + 2))
        plt.xlabel(f"# tracks within dR < {args.dr_track}")
        plt.ylabel("Entries")
        plt.title("tracks near jets (after overlap removal)")
        plt.tight_layout()
        plt.savefig(tracks_jets_png_path, dpi=200)
        info(f"wrote {tracks_jets_png_path}")

    t, ph_eff, jet_misid, acc = best_threshold_separation(photon_track_counts, jet_track_counts)
    if np.isfinite(acc):
        info(
            f"track-count cut (dR<{args.dr_track}): photon if n_tracks<=t, "
            f"t={t}, eff={ph_eff:.3f}, fake={jet_misid:.3f}, rej={1.0-jet_misid:.3f}"
        )

    iso_df = pd.DataFrame(iso_rows)
    iso_df.to_csv(iso_out_csv_path, index=False)
    info(f"wrote {iso_out_csv_path}")

    ph_iso = np.array(photon_iso_vals, dtype=float)
    jt_iso = np.array(jet_iso_vals, dtype=float)

    if np.any(np.isfinite(ph_iso)):
        plt.figure()
        plt.hist(ph_iso[np.isfinite(ph_iso)], bins=60)
        plt.xlabel(rf"$I$ (track isolation), $\Delta R<{args.iso_dr}$")
        plt.ylabel("Entries")
        plt.title("track isolation for photons")
        plt.tight_layout()
        plt.savefig(iso_photons_png_path, dpi=200)
        info(f"wrote {iso_photons_png_path}")

    if np.any(np.isfinite(jt_iso)):
        plt.figure()
        plt.hist(jt_iso[np.isfinite(jt_iso)], bins=60)
        plt.xlabel(rf"$I$ (track isolation), $\Delta R<{args.iso_dr}$")
        plt.ylabel("Entries")
        plt.title("track isolation for jets")
        plt.tight_layout()
        plt.savefig(iso_jets_png_path, dpi=200)
        info(f"wrote {iso_jets_png_path}")

    ph_f = ph_iso[np.isfinite(ph_iso)]
    jt_f = jt_iso[np.isfinite(jt_iso)]
    info(f"iso stats: photons n={ph_f.size}, jets n={jt_f.size}")

    fpr, tpr, best_cut, best_J = roc_from_iso(ph_iso, jt_iso)
    if len(fpr) > 0:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("Fake rate (jets passing cut)")
        plt.ylabel("Acceptance (photons passing cut)")
        plt.title(rf"Acceptance vs Fake rate (iso), $\Delta R<{args.iso_dr}$")
        plt.tight_layout()
        plt.savefig(roc_png_path, dpi=200)
        info(f"wrote {roc_png_path}")

        acc_best = float(np.mean(ph_f < best_cut)) if ph_f.size else float("nan")
        fake_best = float(np.mean(jt_f < best_cut)) if jt_f.size else float("nan")
        info(
            f"iso ROC (dR<{args.iso_dr}): best cut={best_cut:.6g}, J={best_J:.3f}, "
            f"acc={acc_best:.3f}, fake={fake_best:.3f}, rej={1.0-fake_best:.3f}"
        )
    else:
        warn("could not compute iso ROC (need finite iso for both photons and jets)")


if __name__ == "__main__":
    main()