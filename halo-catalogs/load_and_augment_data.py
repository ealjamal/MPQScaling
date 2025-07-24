import pandas as pd
import numpy as np
import h5py
import json
import sys
import warnings
from time import time


def gapper_vel_disp(velocities):
    '''
    Calculate velocity dispersion of given velocities using the gapper method

    '''

    sorted_v = np.sort(velocities)
    n = len(sorted_v)
    if n <= 1:
        return float("-inf")
    gaps = sorted_v[1:] - sorted_v[:-1]

    weights = np.arange(1, n) * (n - np.arange(1, n))
    weighted_gaps = gaps * weights

    sigma_G = np.sqrt(np.pi)/(n * (n - 1)) * np.sum(weighted_gaps)

    return sigma_G


def gapper_vel_disp_bootstrap(velocities, num_bootstrap = 1000, random_seed = 0):
    '''
    Calculate median velocity dispersion using bootstrapping of the gapper velocity
    dispersion calculation.
    
    '''

    sigma_G_bootstrap = []
    dng = np.random.default_rng(seed = random_seed)
    for _ in range(num_bootstrap):
        vels_random = dng.choice(velocities, size = len(velocities), replace = True)
        sigma_G = gapper_vel_disp(vels_random)
        if not np.isnan(sigma_G):
            sigma_G_bootstrap.append(sigma_G)

    if len(sigma_G_bootstrap) > 0:
        sigma_G_median = np.percentile(sigma_G_bootstrap, 50)
    else:
        sigma_G_median = np.nan

    sigma_G_bootstrap = np.array(sigma_G_bootstrap)

    return sigma_G_median


def log_stellar_mass_gap(subhalo_log_stellar_masses, brighter_ind, fainter_ind):
    '''
    Calculate the log stellar mass gap between the ith brightest and jth brightest
    subhalos. Brightest refers to the largest subhalo stellar mass.

    '''
    
    if len(subhalo_log_stellar_masses) < fainter_ind:
        return np.nan
    
    sorted_sub_log_stellar_masses = np.sort(subhalo_log_stellar_masses)[::-1]
    brighter_sub_log_stellar_mass = sorted_sub_log_stellar_masses[brighter_ind - 1]
    fainter_sub_log_stellar_mass = sorted_sub_log_stellar_masses[fainter_ind - 1]

    log_stellar_mass_gap = np.log10(np.power(10, brighter_sub_log_stellar_mass) - 
                                    np.power(10, fainter_sub_log_stellar_mass))
    
    return log_stellar_mass_gap


def TNG_load_and_add_properties(halo_file, subhalo_file, min_sub_M_star,
                                result_columns = None, num_bootstrap = 1000,
                                random_seed = 0):
    '''
    Load TNG halo properties and add new properties to them.

    '''

    # Load halo and subhalo catalogs.
    halos = pd.read_csv(halo_file)
    subs = pd.read_csv(subhalo_file)

    # If result_columns is not provided, use all columns.
    if result_columns is None:
        result_columns = list(halos.columns)

    # Add new properties to halo data frame.
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_500c"]
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_500c"]
    halos["sSFR_500c"] = halos["SFR_500c"] - halos["M_star_500c"]

    # Merge the halo and subhalo catalgues.
    halos_subs_merged = halos.merge(subs, left_on = "halo_ID", right_on = "host_ID", how = "inner", suffixes = ("", "_sub"))
    # Define a column that identifies which subhalo is central subhalo.
    halos_subs_merged["is_central"] = (halos_subs_merged["sub_ID"] == halos_subs_merged["BCG_ID"])
    # Calculate the distance of each subhalo from the halo center.
    halos_subs_merged["sub_dist_from_halo"] = np.linalg.norm([halos_subs_merged["halo_pos_physical_x"] - halos_subs_merged["sub_pos_x"],
                                                              halos_subs_merged["halo_pos_physical_y"] - halos_subs_merged["sub_pos_y"],
                                                              halos_subs_merged["halo_pos_physical_z"] - halos_subs_merged["sub_pos_z"]],
                                                             ord = 2, axis = 0)
    # Only consider subhalos within R_500c and with stellar mass greater than the minimum suhalo stellar mass given.
    halos_subs_merged = halos_subs_merged[(halos_subs_merged["sub_M_star"] >= min_sub_M_star) & 
                                          (halos_subs_merged["sub_dist_from_halo"] <= halos_subs_merged["R_500c"])].copy()

    # Define two new properties: the log stellar mass gap between the first and second highest stellar mass subhalos
    # and the first and fourth highest stellar mass subhalos.
    M_star_12 = halos_subs_merged.groupby(by="halo_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], 1, 2), 
                                                              include_groups=False
                                                             ).rename("M_star_12").reset_index()
    M_star_14 = halos_subs_merged.groupby(by="halo_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], 1, 4), 
                                                              include_groups=False
                                                             ).rename("M_star_14").reset_index()
    # Add the two new properties to original halos data frame.
    halos = halos.merge(M_star_12, on = "halo_ID", how = "left").merge(M_star_14, on = "halo_ID", how = "left")

    # Define a data frame including only central subhalos.
    centrals = halos_subs_merged[halos_subs_merged["is_central"] == 1].copy().rename({"sub_com_vel_x": "BCG_vel_x",
                                                                                      "sub_com_vel_y": "BCG_vel_y",
                                                                                      "sub_com_vel_z": "BCG_vel_z"},
                                                                                     axis = 1)
    # Add BCG properties to halos data frame.                                                                                                                                                         
    halos = halos.merge(centrals[["halo_ID", "M_star_BCG_30kpc", "M_star_BCG_100kpc"]], on = "halo_ID", how = "left")

    # Define satellite (non-central) subhalos.
    sats = halos_subs_merged[halos_subs_merged["is_central"] == 0].copy().merge(centrals[["halo_ID",
                                                                                          "BCG_vel_x",
                                                                                          "BCG_vel_y",
                                                                                          "BCG_vel_z"]], on = "halo_ID")
    # Define the subhalo component-wise relative velocities with respect to the BCG velocity for each satellite subhalo.                                                                                      
    sats["sub_rel_vel_x"] = sats["sub_com_vel_x"] - sats["BCG_vel_x"]
    sats["sub_rel_vel_y"] = sats["sub_com_vel_y"] - sats["BCG_vel_y"]
    sats["sub_rel_vel_z"] = sats["sub_com_vel_z"] - sats["BCG_vel_z"]

    # Define the number of satellite galaxies for each halo.
    N_sat = sats.groupby(by = "halo_ID").apply('size').rename("N_sat_500c").reset_index()
    # Define the gapper velocity dispersion of satellite subhalos for each component.
    print(f"\nCalculating gapper velocity dispersion with {num_bootstrap} bootstrap realizations.")
    sigma_G_x = sats.groupby(by = "halo_ID")["sub_rel_vel_x"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_x = sigma_G_x.rename("sat_vel_disp_x_500c").reset_index()
    sigma_G_y = sats.groupby(by = "halo_ID")["sub_rel_vel_y"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_y = sigma_G_y.rename("sat_vel_disp_y_500c").reset_index()
    sigma_G_z = sats.groupby(by = "halo_ID")["sub_rel_vel_z"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_z = sigma_G_z.rename("sat_vel_disp_z_500c").reset_index()

    # Create a data frame containing velocity dispersion of all three components in order to join with halo dataframe.
    sigma_G_comps = sigma_G_x.merge(sigma_G_y, on = "halo_ID").merge(sigma_G_z, on = "halo_ID")

    # Add number of satellite galaxies and velocity dispersion of each component to halo data frame.
    halos = halos.merge(N_sat, on = "halo_ID", how = "left").merge(sigma_G_comps, on = "halo_ID", how = "left")
    # Calculate the 3D velocity dispersion.
    halos["sat_vel_disp_500c"] = np.sqrt(halos["sat_vel_disp_x_500c"]**2 + 
                                         halos["sat_vel_disp_y_500c"]**2 + 
                                         halos["sat_vel_disp_z_500c"]**2)
    # Take the logarithm of new satellite values.                                         
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        halos["N_sat_500c"] = np.log10(halos["N_sat_500c"])
        halos["sat_vel_disp_x_500c"] = np.log10(halos["sat_vel_disp_x_500c"])
        halos["sat_vel_disp_y_500c"] = np.log10(halos["sat_vel_disp_y_500c"])
        halos["sat_vel_disp_z_500c"] = np.log10(halos["sat_vel_disp_z_500c"])
        halos["sat_vel_disp_500c"] = np.log10(halos["sat_vel_disp_500c"])

    # Return the augmented halo data frame.
    halos["halo_ID"] = halos["halo_ID"].astype(np.int64)
    halos_final_data = halos[result_columns].sort_values("M_500c", ascending = False).reset_index().drop("index", axis = 1)

    return halos_final_data


def FLAMINGO_load_and_add_properties(halo_file, subhalo_file, min_sub_M_star,
                                     result_columns = None, num_bootstrap = 1000,
                                     random_seed = 0):
    '''
    Load FLAMINGO halo properties and add new properties to them.

    '''
    
    # Load halo and subhalo catalogs. Note in FLAMINGO, the halo catalog is the SO properties
    # of the central subhalos.
    halos = pd.read_csv(halo_file)
    subs = pd.read_csv(subhalo_file)

    # If result_columns is not provided, use all columns.
    if result_columns is None:
        result_columns = halos.columns

    # Add new properties to halo data frame.
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_wo_recent_AGN_500c"]
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_wo_recent_AGN_500c"]
    halos["sSFR_500c"] = halos["SFR_500c"] - halos["M_star_500c"]

    # Rename columns for interpretability.
    centrals_rename_cols = {"R_500c_physical": "R_500c", "sub_pos_physical_x": "BCG_pos_x",
                            "sub_pos_physical_y": "BCG_pos_y", "sub_pos_physical_z": "BCG_pos_z",
                            "sub_com_vel_x": "BCG_vel_x", "sub_com_vel_y": "BCG_vel_y",
                            "sub_com_vel_z": "BCG_vel_z"}
    centrals = halos.copy().rename(centrals_rename_cols, axis = 1)

    # Merge the subhalo and central data frame so that we have the BCG positions and velocities.
    subs = subs.merge(centrals[["host_ID"] + list(centrals_rename_cols.values())], on = "host_ID", how = "left")
    subs["sub_dist_from_halo"] = np.linalg.norm([subs["BCG_pos_x"] - subs["sub_pos_physical_x"],
                                                 subs["BCG_pos_y"] - subs["sub_pos_physical_y"],
                                                 subs["BCG_pos_z"] - subs["sub_pos_physical_z"]], ord = 2, axis = 0)
    # Only consider subhalos within R_500c and with stellar mass greater than the minimum suhalo stellar mass given.
    subs = subs[(subs["sub_M_star"] >= min_sub_M_star) & (subs["sub_dist_from_halo"] <= subs["R_500c"])]

    # Define two new properties: the log stellar mass gap between the first and second highest stellar mass subhalos
    # and the first and fourth highest stellar mass subhalos.
    M_star_12 = subs.groupby(by = "host_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], 1, 2), 
                                                   include_groups=False
                                                  ).rename("M_star_12").reset_index()
    M_star_14 = subs.groupby(by = "host_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], 1, 4), 
                                                   include_groups=False
                                                  ).rename("M_star_14").reset_index()
    # Add the two new properties to original halos data frame.
    halos = halos.merge(M_star_12, on = "host_ID", how = "left").merge(M_star_14, on = "host_ID", how = "left")

    # Define satellite (non-central) subhalos.
    sats = subs[subs["is_central"] == 0].copy()
    # Define the subhalo component-wise relative velocities with respect to the BCG velocity for each satellite subhalo.     
    sats["sub_rel_vel_x"] = sats["sub_com_vel_x"] - sats["BCG_vel_x"]
    sats["sub_rel_vel_y"] = sats["sub_com_vel_y"] - sats["BCG_vel_y"]
    sats["sub_rel_vel_z"] = sats["sub_com_vel_z"] - sats["BCG_vel_z"]
    
    # Define the number of satellite galaxies for each halo.
    N_sat = sats.groupby(by = "host_ID").apply('size').rename("N_sat_500c").reset_index()
    # Define the gapper velocity dispersion of satellite subhalos for each component.
    print(f"\nCalculating gapper velocity dispersion with {num_bootstrap} bootstrap realizations.")
    sigma_G_x = sats.groupby(by = "host_ID")["sub_rel_vel_x"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_x = sigma_G_x.rename("sat_vel_disp_x_500c").reset_index()
    sigma_G_y = sats.groupby(by = "host_ID")["sub_rel_vel_y"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_y = sigma_G_y.rename("sat_vel_disp_y_500c").reset_index()
    sigma_G_z = sats.groupby(by = "host_ID")["sub_rel_vel_z"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_z = sigma_G_z.rename("sat_vel_disp_z_500c").reset_index()

    # Create a data frame containing velocity dispersion of all three components in order to join with halo dataframe.
    sigma_G_comps = sigma_G_x.merge(sigma_G_y, on = "host_ID").merge(sigma_G_z, on = "host_ID")

    # Add number of satellite galaxies and velocity dispersion of each component to halo data frame.
    halos = halos.merge(N_sat, on = "host_ID", how = "left").merge(sigma_G_comps, on = "host_ID", how = "left")
    # Calculate the 3D velocity dispersion.
    halos["sat_vel_disp_500c"] = np.sqrt(halos["sat_vel_disp_x_500c"]**2 + 
                                         halos["sat_vel_disp_y_500c"]**2 + 
                                         halos["sat_vel_disp_z_500c"]**2)
    # Take the logarithm of new satellite values.                                          
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        halos["N_sat_500c"] = np.log10(halos["N_sat_500c"])
        halos["sat_vel_disp_x_500c"] = np.log10(halos["sat_vel_disp_x_500c"])
        halos["sat_vel_disp_y_500c"] = np.log10(halos["sat_vel_disp_y_500c"])
        halos["sat_vel_disp_z_500c"] = np.log10(halos["sat_vel_disp_z_500c"])
        halos["sat_vel_disp_500c"] = np.log10(halos["sat_vel_disp_500c"])

    # Return the augmented halo data frame.
    halos.rename({"R_200c_physical": "R_200c", "R_500c_physical": "R_500c"}, axis = 1, inplace = True)
    halos["halo_cat_ID"] = halos["halo_cat_ID"].astype(np.int64)
    halos["host_ID"] = halos["host_ID"].astype(np.int64)
    halos_final_data = halos[result_columns].sort_values("M_500c", ascending = False).reset_index().drop("index", axis = 1)

    return halos_final_data


def save_h5_halo_catalog(df, sim_name, snap, min_sub_M_star, num_bootstrap):
    '''
    Save new halo catalogs in h5 files with descriptions information.

    '''

    if "flam" in sim_name.lower():
        snap = f"00{snap}"
    with h5py.File(f"{sim_name}_halo_catalog_snap{snap}.h5", "w") as f:
        data = df.to_numpy()
        properties = df.columns
    
        dset = f.create_dataset('data', data=data, dtype = np.float64)
        dset.attrs['properties'] = np.array(properties, dtype='S')

        description = f"This data includes halo data from {sim_name} at snapshot {snap}. Hot gas is of temperature " + \
                      "greater than or equal to 10^5 K and cold gas is of temperature less than or equal to 10^4 K." + \
                      " The minimum log stellar mass for a subhalo to be considered a satellite galaxy is " \
                      f"{min_sub_M_star} Msol. The velocity dispersion represents the log gapper median " + \
                      f"using {num_bootstrap} bootstrap samples."
        scale_info = "All fields are on a logarithmic scale except for R_500c, R_200c. "
        units_info = {"masses": "Msol", "Radii": "Mpc", "Temperatures": "K", "luminosities": "erg/s", "Y_X": "Msol K",
                      "Y_SZ": "Msol L", "SFR": "Msol/yr", "sSFR": "1/yr", "velocity dispersion": "km/s"}

        metadata = {"description": description,
                    "author": "Eddie Aljamal",
                    "scales": scale_info,
                    "units": units_info}

        json_metadata = json.dumps(metadata)
        dset.attrs['metadata'] = json_metadata


def main():
    '''
    Save halo catalogs as h5 files for TNG300-1, TNG-Cluster, FLAMINGO-L1_m8 for different snapshots.

    '''

    data_path = "<path/to/halo/and/subhalo/catalogs/"
    min_sub_M_star = float(sys.argv[1])
    num_bootstrap = int(sys.argv[2])

    tng_snaps = [33, 50, 67, 99]
    tng_result_columns = ["halo_ID", "M_200c", "R_200c", "M_500c", "R_500c", 
                          "M_hot_gas_500c", "M_cold_gas_500c", "T_sl_500c", "T_mw_hot_gas_500c", 
                          "L_X_ROSAT_obs_500c", "L_X_ROSAT_obs_ce_500c", "Y_X_500c", "Y_SZ_500c",
                          "M_star_500c", "M_star_BCG_30kpc", "M_star_BCG_100kpc", "M_bh_500c",
                          "M_star_12", "M_star_14", "SFR_500c", "sSFR_500c",
                          "N_sat_500c", "sat_vel_disp_x_500c", "sat_vel_disp_y_500c", 
                          "sat_vel_disp_z_500c", "sat_vel_disp_500c"]
    flam_snaps = [38, 58, 68, 78]
    flam_result_columns = ["halo_cat_ID", "host_ID", "M_200c", "R_200c", "M_500c", "R_500c", 
                           "M_hot_gas_500c", "M_cold_gas_500c", "T_sl_500c", "T_sl_wo_recent_AGN_500c",
                           "T_mw_hot_gas_500c", "T_mw_hot_gas_wo_recent_AGN_500c",
                           "L_X_ROSAT_obs_500c", "L_X_ROSAT_obs_wo_recent_AGN_500c",
                           "L_X_ROSAT_obs_ce_500c", "L_X_ROSAT_obs_wo_recent_AGN_ce_500c",
                           "Y_X_500c", "Y_SZ_500c", 
                           "M_star_500c", "M_star_BCG_30kpc", "M_star_BCG_100kpc", "M_bh_500c",
                           "M_star_12", "M_star_14", "SFR_500c", "sSFR_500c",
                           "N_sat_500c", "sat_vel_disp_x_500c", "sat_vel_disp_y_500c", 
                           "sat_vel_disp_z_500c", "sat_vel_disp_500c"]
                           
    for tng_snap in tng_snaps:
        tng300_halo_file = data_path + f"TNG300_1_halo_catalog_snap{tng_snap}.csv"
        tng300_subhalo_file = data_path + f"TNG300_1_subhalo_catalog_snap{tng_snap}.csv"
        tng_cluster_halo_file = data_path + f"TNG_Cluster_halo_catalog_snap{tng_snap}.csv"
        tng_cluster_subhalo_file = data_path + f"TNG_Cluster_subhalo_catalog_snap{tng_snap}.csv"

        tng300_halo_catalog = TNG_load_and_add_properties(tng300_halo_file, tng300_subhalo_file,
                                                            min_sub_M_star, result_columns = tng_result_columns,
                                                            num_bootstrap = num_bootstrap)
        save_h5_halo_catalog(tng300_halo_catalog, "TNG300_1", tng_snap, min_sub_M_star, num_bootstrap)

        tng_cluster_halo_catalog = TNG_load_and_add_properties(tng_cluster_halo_file, tng_cluster_subhalo_file,
                                                                 min_sub_M_star, result_columns = tng_result_columns,
                                                                 num_bootstrap = num_bootstrap)
        save_h5_halo_catalog(tng_cluster_halo_catalog, "TNG_Cluster", tng_snap, min_sub_M_star, num_bootstrap)
        
    for flam_snap in flam_snaps:
        flam_halo_file = data_path + f"FLAMINGO_L1000N3600_halo_catalog_snap00{flam_snap}.csv"
        flam_subhalo_file = data_path + f"FLAMINGO_L1000N3600_subhalo_catalog_snap00{flam_snap}.csv"
        flam_halo_catalog = FLAMINGO_load_and_add_properties(flam_halo_file, flam_subhalo_file,
                                                               min_sub_M_star, result_columns = flam_result_columns,
                                                               num_bootstrap = num_bootstrap)

        save_h5_halo_catalog(flam_halo_catalog, "FLAMINGO_L1000N3600", flam_snap, min_sub_M_star, num_bootstrap)


if __name__ == "__main__":
    # Time the duration of the analysis.
    start = time()
    main()
    end = time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\n\nDone! Total time elapsed {hours:02d}:{minutes:02d}:{seconds:02d}\n")