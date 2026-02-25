# eta_plot.py
#
# Click Run → produces:
#   - eta_tracks_vs_photons.png
#   - eta_jets_vs_photons.png
#
# Assumes files live directly in BASE_DIR with names:
#   photons_*.csv, jets_*.csv, tracks_*.csv
# If your track files are named differently, change TRACK_GLOB accordingly.

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------

BASE_DIR = r"C:/Users/Tom Greenwood/Desktop/University/Year 4/SEM 2/SH Project/initial_data/1k_ev"

PHOTON_GLOB = BASE_DIR + r"/photons_*.csv"
JET_GLOB    = BASE_DIR + r"/jets_*.csv"
TRACK_GLOB  = BASE_DIR + r"/tracks_*.csv"

# ---------------- HELPERS ----------------

def load_eta_from_file(path):
    """
    Load eta values from one CSV.

    Supports:
      (a) header row with an 'eta'-like column
      (b) no header, numeric columns where eta is column index 1
          (matches your shown layout: pT, eta, phi, e, ...)

    Returns: 1D numpy array of eta values (possibly empty).
    """
    # try headered
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 0:
            for c in df.columns:
                if "eta" in str(c).lower():
                    eta = pd.to_numeric(df[c], errors="coerce").to_numpy()
                    return eta[np.isfinite(eta)]
    except Exception:
        pass

    # fall back: no header
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] < 2:
            return np.array([])
        eta = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        return eta[np.isfinite(eta)]
    except Exception:
        return np.array([])


def load_eta_from_glob(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[warn] no files found for {label} with pattern: {pattern}")
        return np.array([]), []

    etas = []
    skipped = 0
    for f in files:
        eta = load_eta_from_file(f)
        if eta.size == 0:
            skipped += 1
            continue
        etas.append(eta)

    if not etas:
        print(f"[warn] found {len(files)} {label} files but none had usable eta")
        return np.array([]), files

    etas = np.concatenate(etas)
    print(f"[info] {label}: found {len(files)} files, loaded {etas.size} eta entries, skipped {skipped}")
    return etas, files


def plot_two_normalised(ref_eta, other_eta, ref_label, other_label, outname, title,
                        eta_min=-5.0, eta_max=5.0, n_bins=60):
    bins = np.linspace(eta_min, eta_max, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])

    ref_counts, _ = np.histogram(ref_eta, bins=bins)
    oth_counts, _ = np.histogram(other_eta, bins=bins)

    ref_norm = ref_counts / np.sum(ref_counts) if np.sum(ref_counts) else ref_counts
    oth_norm = oth_counts / np.sum(oth_counts) if np.sum(oth_counts) else oth_counts

    plt.figure(figsize=(10, 5))
    plt.step(centres, ref_norm, where="mid", linewidth=2, label=f"{ref_label} (normalised)")
    plt.step(centres, oth_norm, where="mid", linewidth=2, label=f"{other_label} (normalised)")

    # guide lines useful for discussion
    for x, ls in [(2.5, "--"), (-2.5, "--"), (4.0, ":"), (-4.0, ":")]:
        plt.axvline(x, linewidth=1, linestyle=ls)

    plt.xlabel(r"$\eta$")
    plt.ylabel("normalised entries")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.show()
    print(f"[info] saved {outname}")


def print_tail_fractions(eta, label):
    if eta.size == 0:
        print(f"[warn] {label}: no eta values, cannot compute fractions")
        return
    print(f"\n[info] {label}: tail fractions")
    print(f"  fraction |eta| > 2.5 : {np.mean(np.abs(eta) > 2.5):.3f}")
    print(f"  fraction |eta| > 4.0 : {np.mean(np.abs(eta) > 4.0):.3f}")


# ---------------- MAIN ----------------

def main():
    photons_eta, photon_files = load_eta_from_glob(PHOTON_GLOB, "photons")
    jets_eta, jet_files       = load_eta_from_glob(JET_GLOB, "jets")
    tracks_eta, track_files   = load_eta_from_glob(TRACK_GLOB, "tracks")

    if photons_eta.size == 0:
        raise RuntimeError("No photon eta values loaded. Check PHOTON_GLOB / file format.")

    # print fractions
    print_tail_fractions(photons_eta, "photons")
    if jets_eta.size:
        print_tail_fractions(jets_eta, "jets")
    if tracks_eta.size:
        print_tail_fractions(tracks_eta, "tracks")

    # plot jets vs photons
    if jets_eta.size:
        plot_two_normalised(
            photons_eta, jets_eta,
            ref_label="photons", other_label="jets",
            outname="eta_jets_vs_photons.png",
            title="Normalised η distributions: jets vs photons",
        )

    # plot tracks vs photons
    if tracks_eta.size:
        plot_two_normalised(
            photons_eta, tracks_eta,
            ref_label="photons", other_label="tracks",
            outname="eta_tracks_vs_photons.png",
            title="Normalised η distributions: tracks vs photons",
        )
    else:
        print("\n[warn] No tracks loaded. If your track files have a different name, change TRACK_GLOB.")


if __name__ == "__main__":
    main()
