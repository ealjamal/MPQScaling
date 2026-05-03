"""
Load TNG halo/subhalo CSVs from Great Lakes and augment with derived quantities.

Usage:
    python tng_load_and_augment.py <simulation> <snap> <min_sub_M_star> [num_bootstrap]

    simulation: TNG300-1, TNG300-2, TNG-Cluster
    snap:       33, 50, 67, 99

Example:
    python tng_load_and_augment.py TNG300-1 99 9.5 1000
"""

import logging
import sys
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from utils import gapper_vel_disp_bootstrap, log_stellar_mass_gap, save_h5

DATA_BASE = Path("/Volumes/external-hard/cosmo-research/stellar-properties-paper/halo_subhalo_catalogs")

RESULT_COLUMNS = [
    "halo_ID", "M_200c", "R_200c", "M_500c", "R_500c",
    "M_dm_500c", "M_hot_gas_500c", "M_cold_gas_500c", "T_sl_500c", "T_mw_hot_gas_500c",
    "L_X_ROSAT_obs_500c", "L_X_ROSAT_obs_ce_500c", "Y_X_500c", "Y_SZ_500c",
    "M_star_500c", "M_star_BCG_30kpc", "M_star_BCG_100kpc", "M_bh_500c",
    "M_star_sat_500c", "M_star_ICL_30kpc_500c", "M_star_ICL_100kpc_500c",
    "M_star_12", "M_star_14", "SFR_500c", "sSFR_500c",
    "N_sat_500c", "sat_vel_disp_x_500c", "sat_vel_disp_y_500c",
    "sat_vel_disp_z_500c", "sat_vel_disp_500c",
]


def load_and_augment(halo_file, subhalo_file, min_sub_M_star,
                     result_columns=None, exclude_subhalo_flag=True,
                     exclude_hierarchical_subhalos=False, num_bootstrap=1000,
                     random_seed=0):
    """
    Load TNG halo/subhalo CSVs and augment with derived quantities.

    Merges BCG properties from the subhalo catalog into the halo catalog,
    optionally filters out halos with SubhaloFlag=0 or hierarchical subhalos
    (SubhaloParent!=0), then computes stellar mass gaps, satellite counts, and
    per-axis gapper velocity dispersions (log10-scaled) for satellites within
    R_500c above the stellar mass threshold.

    Parameters
    ----------
    halo_file : str or Path
        Path to the halo catalog CSV.
    subhalo_file : str or Path
        Path to the subhalo catalog CSV.
    min_sub_M_star : float
        Minimum log10 stellar mass (log10 Msol) for a subhalo to enter the
        satellite pool used for gaps and velocity dispersions.
    result_columns : list of str or None
        Columns to retain in the returned DataFrame. If None, all columns
        are kept.
    exclude_subhalo_flag : bool
        If True, drop halos whose BCG has SubhaloFlag=0 and subhalos with
        SubhaloFlag=0 from the satellite pool.
    exclude_hierarchical_subhalos : bool
        If True, drop halos whose BCG has SubhaloParent!=0 and subhalos with
        SubhaloParent!=0 from the satellite pool.
    num_bootstrap : int
        Number of bootstrap samples for the gapper velocity dispersion.
    random_seed : int
        Seed for the bootstrap random number generator.

    Returns
    -------
    pandas.DataFrame
        Augmented halo catalog sorted by M_500c descending, restricted to
        `result_columns`, with integer halo_ID.
    """
    
    halos = pd.read_csv(halo_file)
    subs = pd.read_csv(subhalo_file)

    # Derived quantities from halo CSV alone
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_500c"]
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_500c"]
    halos["sSFR_500c"] = halos["SFR_500c"] - halos["M_star_500c"]

    # Pull BCG properties (flag, parent, velocities, aperture masses) from subhalo CSV
    bcg_cols = {
        "sub_flag": "BCG_flag",
        "sub_parent_sub": "BCG_parent_sub",
        "sub_com_vel_x": "BCG_vel_x",
        "sub_com_vel_y": "BCG_vel_y",
        "sub_com_vel_z": "BCG_vel_z",
    }
    bcg_passthrough = ["M_star_BCG_100kpc", "M_star_BCG_30kpc"]
    bcgs = subs[["sub_ID"] + list(bcg_cols.keys()) + bcg_passthrough].rename(columns=bcg_cols)
    halos = halos.merge(bcgs, left_on="BCG_ID", right_on="sub_ID", how="left").drop(columns="sub_ID")

    print(f"Number of centrals with SubhaloFlag = 0: {len(halos[halos['BCG_flag'] == 0])}")
    print(f"Number of centrals with SubhaloParent != 0: {len(halos[halos['BCG_parent_sub'] != 0])}")
    if exclude_subhalo_flag:
        halos = halos[halos["BCG_flag"] == 1].copy()
        print("Dropped all BCGs with SubhaloFlag = 0.")
    if exclude_hierarchical_subhalos:
        halos = halos[halos["BCG_parent_sub"] == 0].copy()
        print("Dropped all BCGs with SubhaloParent != 0.")

    # Restrict subhalos to valid halos and apply satellite-pool exclusions
    subs_valid = subs[subs["host_ID"].isin(halos["halo_ID"])].copy()
    if exclude_subhalo_flag:
        subs_valid = subs_valid[subs_valid["sub_flag"] == 1]
    if exclude_hierarchical_subhalos:
        subs_valid = subs_valid[subs_valid["sub_parent_sub"] == 0]

    # Mark centrals so we can separate satellites
    subs_valid = subs_valid.merge(
        halos[["halo_ID", "BCG_ID"]], left_on="host_ID", right_on="halo_ID", how="left"
    ).drop(columns="halo_ID")
    subs_valid["is_central"] = subs_valid["sub_ID"] == subs_valid["BCG_ID"]
    subs_valid = subs_valid.drop(columns="BCG_ID")

    # Stellar mass gaps: all subhalos (incl. BCG) in R_500c above min stellar mass
    pool = subs_valid[subs_valid["in_R_500c"] & (subs_valid["sub_M_star"] >= min_sub_M_star)]
    M_star_12 = (
        pool.groupby("host_ID")
        .apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["num_star_particles_catalog"], 1, 2),
               include_groups=False)
        .rename("M_star_12").reset_index()
        .rename(columns={"host_ID": "halo_ID"})
    )
    M_star_14 = (
        pool.groupby("host_ID")
        .apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["num_star_particles_catalog"], 1, 4),
               include_groups=False)
        .rename("M_star_14").reset_index()
        .rename(columns={"host_ID": "halo_ID"})
    )

    # Satellites in R_500c above min stellar mass with BCG-relative velocities
    sats_pool = subs_valid[
        ~subs_valid["is_central"] & subs_valid["in_R_500c"] & (subs_valid["sub_M_star"] >= min_sub_M_star)
    ].copy()
    sats_pool = sats_pool.merge(
        halos[["halo_ID", "BCG_vel_x", "BCG_vel_y", "BCG_vel_z"]],
        left_on="host_ID", right_on="halo_ID", how="left"
    ).drop(columns="halo_ID")
    sats_pool["sub_rel_vel_x"] = sats_pool["sub_com_vel_x"] - sats_pool["BCG_vel_x"]
    sats_pool["sub_rel_vel_y"] = sats_pool["sub_com_vel_y"] - sats_pool["BCG_vel_y"]
    sats_pool["sub_rel_vel_z"] = sats_pool["sub_com_vel_z"] - sats_pool["BCG_vel_z"]

    N_sat = (
        sats_pool.groupby("host_ID").apply("size")
        .rename("N_sat_500c").reset_index()
        .rename(columns={"host_ID": "halo_ID"})
    )

    print(f"\nCalculating gapper velocity dispersion with {num_bootstrap} bootstrap realizations.")
    sigma_G_x = (
        sats_pool.groupby("host_ID")["sub_rel_vel_x"]
        .apply(gapper_vel_disp_bootstrap, num_bootstrap=num_bootstrap, random_seed=random_seed)
        .rename("sat_vel_disp_x_500c").reset_index().rename(columns={"host_ID": "halo_ID"})
    )
    sigma_G_y = (
        sats_pool.groupby("host_ID")["sub_rel_vel_y"]
        .apply(gapper_vel_disp_bootstrap, num_bootstrap=num_bootstrap, random_seed=random_seed)
        .rename("sat_vel_disp_y_500c").reset_index().rename(columns={"host_ID": "halo_ID"})
    )
    sigma_G_z = (
        sats_pool.groupby("host_ID")["sub_rel_vel_z"]
        .apply(gapper_vel_disp_bootstrap, num_bootstrap=num_bootstrap, random_seed=random_seed)
        .rename("sat_vel_disp_z_500c").reset_index().rename(columns={"host_ID": "halo_ID"})
    )

    halos = (halos
             .merge(M_star_12, on="halo_ID", how="left")
             .merge(M_star_14, on="halo_ID", how="left")
             .merge(N_sat, on="halo_ID", how="left")
             .merge(sigma_G_x, on="halo_ID", how="left")
             .merge(sigma_G_y, on="halo_ID", how="left")
             .merge(sigma_G_z, on="halo_ID", how="left"))
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
    halos_final["halo_ID"] = halos_final["halo_ID"].astype(np.int64)
    
    return halos_final


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: python tng_load_and_augment.py <simulation> <snap> <min_sub_M_star> [num_bootstrap]\n"
            "  simulation: TNG300-1, TNG300-2, TNG-Cluster\n"
            "  snap:       33, 50, 67, 99\n"
            "Example: python tng_load_and_augment.py TNG300-1 99 9.5 1000"
        )

    simulation = sys.argv[1]
    snap = int(sys.argv[2])
    min_sub_M_star = float(sys.argv[3])
    num_bootstrap = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    sim_dir = DATA_BASE / simulation
    sim_label = simulation.replace("-", "_")
    halo_file = sim_dir / f"{sim_label}_halo_catalog_snap{snap}.csv"
    subhalo_file = sim_dir / f"{sim_label}_subhalo_catalog_snap{snap}.csv"

    logging.info(f"Processing {simulation} snapshot {snap}")
    catalog = load_and_augment(
        halo_file, subhalo_file, min_sub_M_star,
        result_columns=RESULT_COLUMNS,
        exclude_subhalo_flag=True,
        exclude_hierarchical_subhalos=False,
        num_bootstrap=num_bootstrap,
    )

    out_path = sim_dir / f"{sim_label}_halo_catalog_snap{snap}.h5"
    save_h5(catalog, out_path, simulation, snap, min_sub_M_star, num_bootstrap)
    logging.info(f"Saved {out_path}")


if __name__ == "__main__":
    start = time()
    main()
    elapsed = time() - start
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"\nDone! Total time elapsed {h:02d}:{m:02d}:{s:02d}")
