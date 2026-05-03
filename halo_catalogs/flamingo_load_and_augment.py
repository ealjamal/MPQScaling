"""
Load FLAMINGO halo/subhalo CSVs from Great Lakes and augment with derived quantities.

Usage:
    python flamingo_load_and_augment.py <simulation> <snap> <min_sub_M_star> [num_bootstrap] [variation]

    simulation: L1000N3600, L1000N1800
    snap:       integer snapshot number (zero-padded to 4 digits internally, e.g. 38 -> 0038)
    variation:  (L1000N1800 only) fgas_m2sig, fgas_m4sig, fgas_m8sig, fgas_p2sig, jets,
                jets_fgas_m4sig, mstar_m1sig, mstar_m1sig_fgas_m4sig, adiabatic, L1m9.
                Omit or pass '_' for the fiducial (no-variation) run.

Examples:
    python flamingo_load_and_augment.py L1000N3600 38 9.5 1000
    python flamingo_load_and_augment.py L1000N1800 77 9.5 1000 fgas_m4sig
    python flamingo_load_and_augment.py L1000N1800 77 9.5 1000 _
"""

import logging
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from utils import (BOX_SIZE_COMOVING, SNAP_TO_Z, gapper_vel_disp_bootstrap,
                   log_stellar_mass_gap, periodic_dist, save_h5)

DATA_BASE = Path("/nfs/turbo/lsa-evrard/MPQ/halo-subhalo-catalogs")

RESULT_COLUMNS = [
    "halo_cat_ID", "host_ID", "M_200c", "R_200c", "M_500c", "R_500c",
    "M_dm_500c", "M_hot_gas_500c", "M_cold_gas_500c",
    "T_sl_500c", "T_sl_wo_recent_AGN_500c",
    "T_mw_hot_gas_500c", "T_mw_hot_gas_wo_recent_AGN_500c",
    "L_X_ROSAT_obs_500c", "L_X_ROSAT_obs_wo_recent_AGN_500c",
    "L_X_ROSAT_obs_ce_500c", "L_X_ROSAT_obs_wo_recent_AGN_ce_500c",
    "Y_X_500c", "Y_SZ_500c",
    "M_star_500c", "M_star_BCG_30kpc", "M_star_BCG_100kpc", "M_bh_500c",
    "M_star_sat_500c", "M_star_ICL_30kpc_500c", "M_star_ICL_100kpc_500c",
    "M_star_12", "M_star_14", "SFR_500c", "sSFR_500c",
    "N_sat_500c", "sat_vel_disp_x_500c", "sat_vel_disp_y_500c",
    "sat_vel_disp_z_500c", "sat_vel_disp_500c",
]

# Supported L1000N1800 variation names and their label suffixes
VARIATION_LABEL: dict[str, str] = {
    "fgas_m2sig":               "_fgas_m2sig",
    "fgas_m4sig":               "_fgas_m4sig",
    "fgas_m8sig":               "_fgas_m8sig",
    "fgas_p2sig":               "_fgas_p2sig",
    "jets":                     "_jets",
    "jets_fgas_m4sig":          "_jets_fgas_m4sig",
    "mstar_m1sig":              "_mstar_m1sig",
    "mstar_m1sig_fgas_m4sig":   "_mstar_m1sig_fgas_m4sig",
    "adiabatic":                "_adiabatic",
    "L1m9":                     "_L1m9",
}


def resolve_paths(simulation: str, snap_str: str, variation: str | None) -> tuple[Path, Path, Path, str]:
    """
    Resolve file paths and simulation label for a FLAMINGO run.

    Parameters
    ----------
    simulation : str
        Simulation box name: 'L1000N3600' or 'L1000N1800'.
    snap_str : str
        Zero-padded snapshot string (e.g. '0038').
    variation : str or None
        L1000N1800 physics variation key (e.g. 'fgas_m4sig'), or None for the
        fiducial run. Must be a key in VARIATION_LABEL or None.

    Returns
    -------
    tuple of (Path, Path, Path, str)
        halo_csv, subhalo_csv, out_h5, label — where label encodes the
        simulation and variation (e.g. 'L1000N1800_fgas_m4sig').

    Raises
    ------
    SystemExit
        If `variation` is not None and not a recognised key in VARIATION_LABEL.
    """
    if simulation == "L1000N1800":
        if variation is not None and variation not in VARIATION_LABEL:
            raise SystemExit(
                f"Unknown variation '{variation}'. Supported: {', '.join(sorted(VARIATION_LABEL))}.\n"
                "Use '_' (or omit) for the fiducial run."
            )
        var_dir = "no_variation" if variation is None else variation
        label_suffix = "" if variation is None else VARIATION_LABEL[variation]
        label = f"{simulation}{label_suffix}"
        sim_data_dir = DATA_BASE / simulation / var_dir
    else:
        label = simulation
        sim_data_dir = DATA_BASE / simulation

    stem = f"FLAMINGO_{label}_halo_catalog_snap{snap_str}"
    halo_csv = sim_data_dir / f"{stem}.csv"
    sub_stem = f"FLAMINGO_{label}_subhalo_catalog_snap{snap_str}"
    subhalo_csv = sim_data_dir / f"{sub_stem}.csv"
    out_h5 = sim_data_dir / f"{stem}.h5"

    return halo_csv, subhalo_csv, out_h5, label


def load_and_augment(halo_file, subhalo_file, min_sub_M_star, boxsize_physical,
                     result_columns=None, num_bootstrap=1000, random_seed=0):
    """
    Load FLAMINGO halo/subhalo CSVs and augment with derived quantities.

    Computes subhalo distances from the halo center using periodic boundary
    conditions to determine which subhalos lie within R_500c, then computes
    stellar mass gaps, satellite counts, and per-axis gapper velocity
    dispersions (log10-scaled) for satellites within R_500c above the
    stellar mass threshold.

    Parameters
    ----------
    halo_file : str or Path
        Path to the halo catalog CSV.
    subhalo_file : str or Path
        Path to the subhalo catalog CSV.
    min_sub_M_star : float
        Minimum log10 stellar mass (log10 Msol) for a subhalo to enter the
        satellite pool used for gaps and velocity dispersions.
    boxsize_physical : float
        Physical side length of the simulation box (Mpc), used for periodic
        boundary wrapping when computing subhalo–halo distances.
    result_columns : list of str or None
        Columns to retain in the returned DataFrame. If None, all columns
        are kept.
    num_bootstrap : int
        Number of bootstrap samples for the gapper velocity dispersion.
    random_seed : int
        Seed for the bootstrap random number generator.

    Returns
    -------
    pandas.DataFrame
        Augmented halo catalog sorted by M_500c descending, restricted to
        `result_columns`, with integer halo_cat_ID and host_ID.
    """
    
    halos = pd.read_csv(halo_file)
    subs = pd.read_csv(subhalo_file)

    # Rename physical radii to drop the "_physical" suffix
    halos.rename(columns={"R_500c_physical": "R_500c", "R_200c_physical": "R_200c"}, inplace=True)

    # Derived quantities from halo CSV alone
    # M_star_sat_500c, M_star_BCG_100kpc/30kpc, M_star_ICL_100kpc/30kpc_500c are already in halo CSV
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_wo_recent_AGN_500c"]
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_wo_recent_AGN_500c"]
    halos["sSFR_500c"] = halos["SFR_500c"] - halos["M_star_500c"]

    # Compute subhalo distance from halo center and in_R_500c flag
    subs = subs.merge(
        halos[["host_ID", "halo_pos_physical_x", "halo_pos_physical_y", "halo_pos_physical_z", "R_500c"]],
        on="host_ID", how="left"
    )
    dx = subs["sub_pos_physical_x"] - subs["halo_pos_physical_x"]
    dy = subs["sub_pos_physical_y"] - subs["halo_pos_physical_y"]
    dz = subs["sub_pos_physical_z"] - subs["halo_pos_physical_z"]
    subs["in_R_500c"] = periodic_dist(dx, dy, dz, boxsize_physical) <= subs["R_500c"]

    # Stellar mass gaps: all subhalos (incl. BCG) in R_500c above min stellar mass
    pool = subs[(subs["in_R_500c"]) & (subs["sub_M_star"] >= min_sub_M_star)]
    M_star_12 = (
        pool.groupby("host_ID")
        .apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["sub_num_star_particles"], 1, 2),
               include_groups=False)
        .rename("M_star_12").reset_index()
    )
    M_star_14 = (
        pool.groupby("host_ID")
        .apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["sub_num_star_particles"], 1, 4),
               include_groups=False)
        .rename("M_star_14").reset_index()
    )

    # Satellites in R_500c above min stellar mass with BCG-relative velocities
    bcg_vels = (
        subs[subs["is_central"] == 1][["host_ID", "sub_com_vel_x", "sub_com_vel_y", "sub_com_vel_z"]]
        .rename(columns={"sub_com_vel_x": "BCG_vel_x",
                         "sub_com_vel_y": "BCG_vel_y",
                         "sub_com_vel_z": "BCG_vel_z"})
    )
    sats_pool = (
        subs[(subs["in_R_500c"]) & (subs["is_central"] == 0) & (subs["sub_M_star"] >= min_sub_M_star)]
        .copy()
        .merge(bcg_vels, on="host_ID", how="left")
    )
    sats_pool["sub_rel_vel_x"] = sats_pool["sub_com_vel_x"] - sats_pool["BCG_vel_x"]
    sats_pool["sub_rel_vel_y"] = sats_pool["sub_com_vel_y"] - sats_pool["BCG_vel_y"]
    sats_pool["sub_rel_vel_z"] = sats_pool["sub_com_vel_z"] - sats_pool["BCG_vel_z"]

    N_sat = sats_pool.groupby("host_ID").apply("size").rename("N_sat_500c").reset_index()

    print(f"\nCalculating gapper velocity dispersion with {num_bootstrap} bootstrap realizations.")
    sigma_G_x = (
        sats_pool.groupby("host_ID")["sub_rel_vel_x"]
        .apply(gapper_vel_disp_bootstrap, num_bootstrap=num_bootstrap, random_seed=random_seed)
        .rename("sat_vel_disp_x_500c").reset_index()
    )
    sigma_G_y = (
        sats_pool.groupby("host_ID")["sub_rel_vel_y"]
        .apply(gapper_vel_disp_bootstrap, num_bootstrap=num_bootstrap, random_seed=random_seed)
        .rename("sat_vel_disp_y_500c").reset_index()
    )
    sigma_G_z = (
        sats_pool.groupby("host_ID")["sub_rel_vel_z"]
        .apply(gapper_vel_disp_bootstrap, num_bootstrap=num_bootstrap, random_seed=random_seed)
        .rename("sat_vel_disp_z_500c").reset_index()
    )

    halos = (halos
             .merge(M_star_12, on="host_ID", how="left")
             .merge(M_star_14, on="host_ID", how="left")
             .merge(N_sat, on="host_ID", how="left")
             .merge(sigma_G_x, on="host_ID", how="left")
             .merge(sigma_G_y, on="host_ID", how="left")
             .merge(sigma_G_z, on="host_ID", how="left"))
    halos["sat_vel_disp_500c"] = np.sqrt(
        halos["sat_vel_disp_x_500c"]**2 +
        halos["sat_vel_disp_y_500c"]**2 +
        halos["sat_vel_disp_z_500c"]**2
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        halos["N_sat_500c"] = np.log10(halos["N_sat_500c"])
        halos["sat_vel_disp_x_500c"] = np.log10(halos["sat_vel_disp_x_500c"])
        halos["sat_vel_disp_y_500c"] = np.log10(halos["sat_vel_disp_y_500c"])
        halos["sat_vel_disp_z_500c"] = np.log10(halos["sat_vel_disp_z_500c"])
        halos["sat_vel_disp_500c"] = np.log10(halos["sat_vel_disp_500c"])

    if result_columns is None:
        result_columns = list(halos.columns)

    halos_final = halos[result_columns].sort_values("M_500c", ascending=False).reset_index(drop=True)
    halos_final["halo_cat_ID"] = halos_final["halo_cat_ID"].astype(np.int64)
    halos_final["host_ID"] = halos_final["host_ID"].astype(np.int64)
    
    return halos_final


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python flamingo_load_and_augment.py <simulation> <snap> <min_sub_M_star> [num_bootstrap] [variation]\n"
            "  simulation: L1000N3600, L1000N1800\n"
            "  snap:       integer (e.g. 38, 58, 68, 78)\n"
            "  variation:  (L1000N1800 only) e.g. fgas_m4sig, or '_' for fiducial\n"
            "Example: python flamingo_load_and_augment.py L1000N3600 38 9.5 1000"
        )

    simulation = sys.argv[1]
    snap_int = int(sys.argv[2])
    snap_str = f"{snap_int:04d}"
    min_sub_M_star = float(sys.argv[3])
    num_bootstrap = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    variation = None
    if len(sys.argv) > 5:
        v = sys.argv[5].strip()
        if v not in ("", "-", "_", "None", "none"):
            variation = v

    halo_file, subhalo_file, out_path, label = resolve_paths(simulation, snap_str, variation)

    redshift = SNAP_TO_Z[simulation][snap_int]
    scale_factor = 1 / (1 + redshift)
    boxsize_physical = BOX_SIZE_COMOVING[simulation] * scale_factor

    logging.info(f"Processing FLAMINGO {label} snapshot {snap_str} (z={redshift}, a={scale_factor:.4f})")
    catalog = load_and_augment(
        halo_file, subhalo_file, min_sub_M_star, boxsize_physical,
        result_columns=RESULT_COLUMNS,
        num_bootstrap=num_bootstrap,
    )

    save_h5(catalog, out_path, label, snap_str, min_sub_M_star, num_bootstrap)
    logging.info(f"Saved {out_path}")


if __name__ == "__main__":
    start = time()
    main()
    elapsed = time() - start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\nDone! Total time elapsed {h:02d}:{m:02d}:{s:02d}")
