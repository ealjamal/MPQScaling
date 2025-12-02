import pandas as pd
import numpy as np
import h5py
import json
import sys
import warnings
from time import time
import logging


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


def log_stellar_mass_gap(subhalo_log_stellar_masses, sub_num_star_particles, brighter_ind, fainter_ind, min_num_star_particles=1):
    '''
    Calculate the log stellar mass gap between the ith brightest and jth brightest
    subhalos. Brightest refers to the largest subhalo stellar mass.

    '''

    subhalo_log_stellar_masses = np.array(subhalo_log_stellar_masses)
    sub_num_star_particles = np.array(sub_num_star_particles)

    num_star_particles_mask = np.where(sub_num_star_particles >= min_num_star_particles)
    subhalo_log_stellar_masses = subhalo_log_stellar_masses[num_star_particles_mask]
    
    if len(subhalo_log_stellar_masses) < fainter_ind:
        return np.nan

    sorted_sub_log_stellar_masses = np.sort(subhalo_log_stellar_masses)[::-1]
    brighter_sub_log_stellar_mass = sorted_sub_log_stellar_masses[brighter_ind - 1]
    fainter_sub_log_stellar_mass = sorted_sub_log_stellar_masses[fainter_ind - 1]

    log_stellar_mass_gap = np.log10(np.power(10, brighter_sub_log_stellar_mass) - 
                                    np.power(10, fainter_sub_log_stellar_mass))
    
    return log_stellar_mass_gap


def load_h5_halo_catalog(halo_file):
    '''
    Load halo catalog from .h5 file.

    --------
    Params
    --------

    halo_file (string):
        File name with .h5 extension containing halo catalog data.
    
    --------
    Output
    --------

    halo_df (pandas.DataFrame):
        Data frame containing halo catalog properties.
    
    '''

    with h5py.File(halo_file, "r") as f:
        # Load the dataset
        dset = f["data"]

        # Load the data into a DataFrame
        properties = [prop.decode('utf-8') if isinstance(prop, bytes) else prop for prop in dset.attrs['properties']]
        halo_df = pd.DataFrame(dset[:], columns=properties)

        return halo_df

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


def log_stellar_mass_gap(subhalo_log_stellar_masses, sub_num_star_particles, brighter_ind, fainter_ind, min_num_star_particles=1):
    '''
    Calculate the log stellar mass gap between the ith brightest and jth brightest
    subhalos. Brightest refers to the largest subhalo stellar mass.

    '''

    subhalo_log_stellar_masses = np.array(subhalo_log_stellar_masses)
    sub_num_star_particles = np.array(sub_num_star_particles)

    num_star_particles_mask = np.where(sub_num_star_particles >= min_num_star_particles)
    subhalo_log_stellar_masses = subhalo_log_stellar_masses[num_star_particles_mask]
    
    if len(subhalo_log_stellar_masses) < fainter_ind:
        return np.nan

    sorted_sub_log_stellar_masses = np.sort(subhalo_log_stellar_masses)[::-1]
    brighter_sub_log_stellar_mass = sorted_sub_log_stellar_masses[brighter_ind - 1]
    fainter_sub_log_stellar_mass = sorted_sub_log_stellar_masses[fainter_ind - 1]

    log_stellar_mass_gap = np.log10(np.power(10, brighter_sub_log_stellar_mass) - 
                                    np.power(10, fainter_sub_log_stellar_mass))
    
    return log_stellar_mass_gap

def TNG_load_and_add_properties(halo_file, subhalo_file, min_sub_M_star,
                                result_columns = None, exclude_subhalo_flag = True, 
                                exclude_hierarchical_subhalos = True, num_bootstrap = 1000,
                                random_seed = 0):
    '''
    Load TNG halo properties and add new properties to them.

    '''

    # Load halo and subhalo catalogs.
    halos = pd.read_csv(halo_file)
    subs = pd.read_csv(subhalo_file)

    # Add new properties to halo data frame.
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_500c"]
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_500c"]
    halos["sSFR_500c"] = halos["SFR_500c"] - halos["M_star_500c"]

    # Merge the halo and subhalo catalogs.
    halos_subs_merged = halos.merge(subs, left_on = "halo_ID", right_on = "host_ID", how = "inner", suffixes = ("", "_sub"))
    # Define a column that identifies which subhalo is central subhalo, distances from halo center, and whether subhalo is in R_500c of parent halo.
    halos_subs_merged["is_central"] = (halos_subs_merged["sub_ID"] == halos_subs_merged["BCG_ID"])
    halos_subs_merged["sub_dist_from_halo"] = np.linalg.norm([halos_subs_merged["halo_pos_physical_x"] - halos_subs_merged["sub_pos_x"], 
                                                              halos_subs_merged["halo_pos_physical_y"] - halos_subs_merged["sub_pos_y"],
                                                              halos_subs_merged["halo_pos_physical_z"] - halos_subs_merged["sub_pos_z"]], axis=0, ord=2)
    halos_subs_merged["in_R_500c"] = (halos_subs_merged["sub_dist_from_halo"] <= halos_subs_merged["R_500c"])

    # Define a central subhalo data frame with new column names in order to calculate relative velocities with respect to the BCG velocity.
    centrals = halos_subs_merged[halos_subs_merged["is_central"] == 1].rename({"sub_flag": "BCG_flag",
                                                                               "sub_parent_sub": "BCG_parent_sub", 
                                                                               "sub_com_vel_x": "BCG_vel_x",
                                                                               "sub_com_vel_y": "BCG_vel_y",
                                                                               "sub_com_vel_z": "BCG_vel_z"},
                                                                              axis = 1).copy()
    
    # Exclude subhalos that are not of cosmological origin (sub_flag == 0), and subhalos that are part of hierarchical structure (sub_parent_sub != 0).
    if exclude_subhalo_flag:
        halos_subs_merged = halos_subs_merged[halos_subs_merged["sub_flag"] == 1]
    if exclude_hierarchical_subhalos:
        halos_subs_merged = halos_subs_merged[halos_subs_merged["sub_parent_sub"] == 0]

    # Define a data frame for satellites.
    sats = halos_subs_merged[halos_subs_merged["is_central"] == 0].copy()
    # Find the total stellar mass in satellites inside of R_500c (if exclusions above apply, then this excludes hierarchical subhalos to avoid double counting
    # and subhalos that aren't of cosmological origin since they are not counted as galaxies)
    total_sat_M_star_in_R_500c = sats[["halo_ID", "sub_M_star_tot_500c"]].groupby("halo_ID").apply(lambda x: np.power(10, x).sum(), include_groups=False).rename({"sub_M_star_tot_500c": "M_star_sat_500c"}, axis = 1).reset_index()

    # Define a data frame for all subhalos in R_500c above a minimum stellar mass.
    halos_subs_merged_in_R_500c_above_min_star_mass = halos_subs_merged[(halos_subs_merged["in_R_500c"] == 1) & (halos_subs_merged["sub_M_star"] >= min_sub_M_star)].copy()
    # Define two new properties: the log stellar mass gap between the first and second highest stellar mass subhalos
    # and the first and fourth highest stellar mass subhalos.
    M_star_12 = halos_subs_merged_in_R_500c_above_min_star_mass.groupby(by="halo_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["num_star_particles"], 1, 2, min_num_star_particles=1), 
                                                                                            include_groups=False
                                                                                            ).rename("M_star_12").reset_index()
    M_star_14 = halos_subs_merged_in_R_500c_above_min_star_mass.groupby(by="halo_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["num_star_particles"], 1, 4, min_num_star_particles=1), 
                                                                                            include_groups=False
                                                                                            ).rename("M_star_14").reset_index()

    # Define a data frame for all satellites in R_500c above a minimum stellar mass.
    sats_in_R_500c_above_min_star_mass = sats[(sats["sub_M_star"] >= min_sub_M_star) & (sats["in_R_500c"] == 1)].merge(centrals[["halo_ID",
                                                                                                                                 "BCG_vel_x",
                                                                                                                                 "BCG_vel_y",
                                                                                                                                 "BCG_vel_z"]], on = "halo_ID")
    # Define the satellite component-wise relative velocities with respect to the BCG velocity for each satellite subhalo.                                                                                      
    sats_in_R_500c_above_min_star_mass["sub_rel_vel_x"] = sats_in_R_500c_above_min_star_mass["sub_com_vel_x"] - sats_in_R_500c_above_min_star_mass["BCG_vel_x"]
    sats_in_R_500c_above_min_star_mass["sub_rel_vel_y"] = sats_in_R_500c_above_min_star_mass["sub_com_vel_y"] - sats_in_R_500c_above_min_star_mass["BCG_vel_y"]
    sats_in_R_500c_above_min_star_mass["sub_rel_vel_z"] = sats_in_R_500c_above_min_star_mass["sub_com_vel_z"] - sats_in_R_500c_above_min_star_mass["BCG_vel_z"]
    # Define the number of satellite galaxies for each halo.
    N_sat = sats_in_R_500c_above_min_star_mass.groupby(by = "halo_ID").apply('size').rename("N_sat_500c").reset_index()
    # Define the gapper velocity dispersion of satellite subhalos for each component.
    print(f"\nCalculating gapper velocity dispersion with {num_bootstrap} bootstrap realizations.")
    sigma_G_x = sats_in_R_500c_above_min_star_mass.groupby(by = "halo_ID")["sub_rel_vel_x"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                                                  random_seed = random_seed)
    sigma_G_x = sigma_G_x.rename("sat_vel_disp_x_500c").reset_index()
    sigma_G_y = sats_in_R_500c_above_min_star_mass.groupby(by = "halo_ID")["sub_rel_vel_y"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                                                  random_seed = random_seed)
    sigma_G_y = sigma_G_y.rename("sat_vel_disp_y_500c").reset_index()
    sigma_G_z = sats_in_R_500c_above_min_star_mass.groupby(by = "halo_ID")["sub_rel_vel_z"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                                                  random_seed = random_seed)
    sigma_G_z = sigma_G_z.rename("sat_vel_disp_z_500c").reset_index()

    # Merge BCG stellar mass as well as the BCG_flag (same as sub_flag) and BCG_parent_sub (same as sub_parent_sub) into halo data frame.
    halos = halos.merge(centrals[["halo_ID", "BCG_flag", "BCG_parent_sub", "M_star_BCG_30kpc", "M_star_BCG_100kpc"]], on="halo_ID", how="left")
    # Merge total satellite stellar mass in R_500c as well stellar mass gap measures into halo data frame.
    halos = halos.merge(total_sat_M_star_in_R_500c, on="halo_ID", how="left").merge(M_star_12, on="halo_ID", how="left").merge(M_star_14, on="halo_ID", how="left")
    # Merge number of satellite galaxies into halo data frame.
    halos = halos.merge(N_sat, on="halo_ID", how="left")
    # Merge 1D velocity dispersion measures into halo data frame.
    halos = halos.merge(sigma_G_x, on="halo_ID", how="left").merge(sigma_G_y, on="halo_ID", how="left").merge(sigma_G_z, on="halo_ID", how="left")
    # 3D velocity dispersion calculation.
    halos["sat_vel_disp_500c"] = np.sqrt(halos["sat_vel_disp_x_500c"]**2 + 
                                         halos["sat_vel_disp_y_500c"]**2 + 
                                         halos["sat_vel_disp_z_500c"]**2)
    
    # Take the logarithm of new satellite values.                                         
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        halos["M_star_sat_500c"] = np.log10(halos["M_star_sat_500c"])
        halos["M_star_ICL_100kpc_500c"] = np.log10(np.power(10, halos["M_star_500c"]) - np.power(10, halos["M_star_BCG_100kpc"]) - np.power(10, halos["M_star_sat_500c"]))
        halos["M_star_ICL_30kpc_500c"] = np.log10(np.power(10, halos["M_star_500c"]) - np.power(10, halos["M_star_BCG_30kpc"]) - np.power(10, halos["M_star_sat_500c"]))
        halos["N_sat_500c"] = np.log10(halos["N_sat_500c"])
        halos["sat_vel_disp_x_500c"] = np.log10(halos["sat_vel_disp_x_500c"])
        halos["sat_vel_disp_y_500c"] = np.log10(halos["sat_vel_disp_y_500c"])
        halos["sat_vel_disp_z_500c"] = np.log10(halos["sat_vel_disp_z_500c"])
        halos["sat_vel_disp_500c"] = np.log10(halos["sat_vel_disp_500c"])

    # Rename columns for ease of reading.
    halos.rename(columns={'halo_pos_physical_x': 'halo_pos_x',
                          'halo_pos_physical_y': 'halo_pos_y',
                          'halo_pos_physical_z': 'halo_pos_z'}, inplace=True)

    # Print the number central galaxies that aren't of cosmological origin BCG_flag == 0.
    print(f"Number of centrals with SubhaloFlag = 0: {len(halos[halos["BCG_flag"] == 0])}")
    # Print the number of central galaxies that are part of a hierarchical structure.
    print(f"Number of centrals with SubhaloParent != 0: {len(halos[halos["BCG_parent_sub"] != 0])}")

    # If exclusion applies, get rid of halos with centrals that aren't of cosmological origin BCG_flag == 0 (binary).
    if exclude_subhalo_flag:
        halos = halos[halos["BCG_flag"] == 1].copy()
        print("Dropped all BCGs with SubhaloFlag = 0.")
    # If exclusion applies, get rid of halos with centrals that are part of a hierarchical structure BCG_parent_sub != 0.
    if exclude_hierarchical_subhalos:
        halos = halos[halos["BCG_parent_sub"] == 0].copy()
        print("Dropped all BCGs with SubhaloParent != 0.")

    # If result_columns is not provided, use all columns.
    if result_columns is None:
        result_columns = list(halos.columns)

    # Create the final data frame with only result column with sorted values of M_500c.
    halos_final_data = halos[result_columns].sort_values("M_500c", ascending = False).reset_index(drop=True)
    # Have halo IDs of integer type.
    halos_final_data["halo_ID"] = halos_final_data["halo_ID"].astype(np.int64)

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

    # Add new properties to halo data frame.
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_wo_recent_AGN_500c"]
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_wo_recent_AGN_500c"]
    halos["sSFR_500c"] = halos["SFR_500c"] - halos["M_star_500c"]

    # Rename columns for ease of reading.
    centrals_rename_cols = {"R_500c_physical": "R_500c", "sub_pos_physical_x": "BCG_pos_x",
                            "sub_pos_physical_y": "BCG_pos_y", "sub_pos_physical_z": "BCG_pos_z",
                            "sub_com_vel_x": "BCG_vel_x", "sub_com_vel_y": "BCG_vel_y",
                            "sub_com_vel_z": "BCG_vel_z"}
    # Create data frame for central subhalos since halos data frame includes only centrals with spherical overdensity (SO) properties
    centrals = halos.copy().rename(centrals_rename_cols, axis = 1)

    # Merge the subhalo and central data frame so that we have the BCG positions and velocities.
    subs = subs.merge(centrals[["host_ID"] + list(centrals_rename_cols.values())], on = "host_ID", how = "left")
    # Calculate the distance of the subhalo from the BCG center (halo center).
    subs["sub_dist_from_halo"] = np.linalg.norm([subs["BCG_pos_x"] - subs["sub_pos_physical_x"],
                                                 subs["BCG_pos_y"] - subs["sub_pos_physical_y"],
                                                 subs["BCG_pos_z"] - subs["sub_pos_physical_z"]], ord = 2, axis = 0)

    # Create data frame for all subhalos in R_500c above the minimum stellar mass.
    subs_in_R_500c_above_min_star_mass = subs[(subs["sub_dist_from_halo"] <= subs["R_500c"]) & (subs["sub_M_star"] >= min_sub_M_star)].copy()
    # Define two new properties: the log stellar mass gap between the first and second highest stellar mass subhalos
    # and the first and fourth highest stellar mass subhalos.
    M_star_12 = subs_in_R_500c_above_min_star_mass.groupby(by = "host_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["sub_num_star_particles"], 1, 2, min_num_star_particles=1), 
                                                                                 include_groups=False
                                                                                ).rename("M_star_12").reset_index()

    M_star_14 = subs_in_R_500c_above_min_star_mass.groupby(by = "host_ID").apply(lambda x: log_stellar_mass_gap(x["sub_M_star"], x["sub_num_star_particles"], 1, 4, min_num_star_particles=1), 
                                                                                 include_groups=False
                                                                                ).rename("M_star_14").reset_index()
    
    # Define a data frame for all satellites in R_500c above a minimum stellar mass.
    sats_in_R_500c_above_min_star_mass = subs[(subs["sub_dist_from_halo"] <= subs["R_500c"]) & (subs["is_central"] == 0) & (subs["sub_M_star"] >= min_sub_M_star)].copy()
    # Define the subhalo component-wise relative velocities with respect to the BCG velocity for each satellite subhalo.     
    sats_in_R_500c_above_min_star_mass["sub_rel_vel_x"] = sats_in_R_500c_above_min_star_mass["sub_com_vel_x"] - sats_in_R_500c_above_min_star_mass["BCG_vel_x"]
    sats_in_R_500c_above_min_star_mass["sub_rel_vel_y"] = sats_in_R_500c_above_min_star_mass["sub_com_vel_y"] - sats_in_R_500c_above_min_star_mass["BCG_vel_y"]
    sats_in_R_500c_above_min_star_mass["sub_rel_vel_z"] = sats_in_R_500c_above_min_star_mass["sub_com_vel_z"] - sats_in_R_500c_above_min_star_mass["BCG_vel_z"]
    # Define the number of satellite galaxies for each halo.
    N_sat = sats_in_R_500c_above_min_star_mass.groupby(by = "host_ID").apply('size').rename("N_sat_500c").reset_index()
    # Define the gapper velocity dispersion of satellite subhalos for each component.
    print(f"\nCalculating gapper velocity dispersion with {num_bootstrap} bootstrap realizations.")
    sigma_G_x = sats_in_R_500c_above_min_star_mass.groupby(by = "host_ID")["sub_rel_vel_x"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_x = sigma_G_x.rename("sat_vel_disp_x_500c").reset_index()
    sigma_G_y = sats_in_R_500c_above_min_star_mass.groupby(by = "host_ID")["sub_rel_vel_y"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_y = sigma_G_y.rename("sat_vel_disp_y_500c").reset_index()
    sigma_G_z = sats_in_R_500c_above_min_star_mass.groupby(by = "host_ID")["sub_rel_vel_z"].apply(gapper_vel_disp_bootstrap, num_bootstrap = num_bootstrap,
                                                                    random_seed = random_seed)
    sigma_G_z = sigma_G_z.rename("sat_vel_disp_z_500c").reset_index()
    # Create a data frame containing velocity dispersion of all three components in order to join with halo dataframe.
    sigma_G_comps = sigma_G_x.merge(sigma_G_y, on = "host_ID").merge(sigma_G_z, on = "host_ID")

    # Add the stellar mass gap properties to the halo data frame.
    halos = halos.merge(M_star_12, on = "host_ID", how = "left").merge(M_star_14, on = "host_ID", how = "left")
    # Add number of satellite galaxies and velocity dispersion of each component to halo data frame.
    halos = halos.merge(N_sat, on = "host_ID", how = "left").merge(sigma_G_comps, on = "host_ID", how = "left")
    # Calculate the 3D velocity dispersion.
    halos["sat_vel_disp_500c"] = np.sqrt(halos["sat_vel_disp_x_500c"]**2 + 
                                         halos["sat_vel_disp_y_500c"]**2 + 
                                         halos["sat_vel_disp_z_500c"]**2)
    
    # Take the logarithm of new satellite values.                                          
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        halos["M_star_ICL_100kpc_500c"] = np.log10(np.power(10, halos["M_star_500c"]) - np.power(10, halos["M_star_BCG_100kpc"]) - np.power(10, halos["M_star_sat_500c"]))
        halos["M_star_ICL_30kpc_500c"] = np.log10(np.power(10, halos["M_star_500c"]) - np.power(10, halos["M_star_BCG_30kpc"]) - np.power(10, halos["M_star_sat_500c"]))
        halos["N_sat_500c"] = np.log10(halos["N_sat_500c"])
        halos["sat_vel_disp_x_500c"] = np.log10(halos["sat_vel_disp_x_500c"])
        halos["sat_vel_disp_y_500c"] = np.log10(halos["sat_vel_disp_y_500c"])
        halos["sat_vel_disp_z_500c"] = np.log10(halos["sat_vel_disp_z_500c"])
        halos["sat_vel_disp_500c"] = np.log10(halos["sat_vel_disp_500c"])                          
    
    # Rename columns for ease of reading.
    halos.rename({"R_200c_physical": "R_200c", "R_500c_physical": "R_500c"}, axis = 1, inplace = True)
    # Have halo IDs and host IDs of integer type.
    halos["halo_cat_ID"] = halos["halo_cat_ID"].astype(np.int64)
    halos["host_ID"] = halos["host_ID"].astype(np.int64)

    # If result_columns is not provided, use all columns.
    if result_columns is None:
        result_columns = halos.columns

    # Create the final data frame with only result column with sorted values of M_500c.
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
                      "Y_SZ": "Msol K", "SFR": "Msol/yr", "sSFR": "1/yr", "velocity dispersion": "km/s"}

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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    data_path = "/nfs/turbo/lsa-evrard/MPQ/halo-subhalo-catalogs/"
    min_sub_M_star = float(sys.argv[1])
    num_bootstrap = int(sys.argv[2])

    tng_snaps = [33, 50, 67, 99]
    tng_result_columns = ["halo_ID", "M_200c", "R_200c", "M_500c", "R_500c", 
                          "M_dm_500c", "M_hot_gas_500c", "M_cold_gas_500c", "T_sl_500c", "T_mw_hot_gas_500c", 
                          "L_X_ROSAT_obs_500c", "L_X_ROSAT_obs_ce_500c", "Y_X_500c", "Y_SZ_500c",
                          "M_star_500c", "M_star_BCG_30kpc", "M_star_BCG_100kpc", "M_bh_500c",
                          "M_star_sat_500c", "M_star_ICL_30kpc_500c", "M_star_ICL_100kpc_500c",
                          "M_star_12", "M_star_14", "SFR_500c", "sSFR_500c",
                          "N_sat_500c", "sat_vel_disp_x_500c", "sat_vel_disp_y_500c", 
                          "sat_vel_disp_z_500c", "sat_vel_disp_500c"]
    flam_snaps = [38, 58, 68, 78]
    flam_result_columns = ["halo_cat_ID", "host_ID", "M_200c", "R_200c", "M_500c", "R_500c", 
                           "M_dm_500c", "M_hot_gas_500c", "M_cold_gas_500c", "T_sl_500c", "T_sl_wo_recent_AGN_500c",
                           "T_mw_hot_gas_500c", "T_mw_hot_gas_wo_recent_AGN_500c",
                           "L_X_ROSAT_obs_500c", "L_X_ROSAT_obs_wo_recent_AGN_500c",
                           "L_X_ROSAT_obs_ce_500c", "L_X_ROSAT_obs_wo_recent_AGN_ce_500c",
                           "Y_X_500c", "Y_SZ_500c", 
                           "M_star_500c", "M_star_BCG_30kpc", "M_star_BCG_100kpc", "M_bh_500c",
                           "M_star_sat_500c", "M_star_ICL_30kpc_500c", "M_star_ICL_100kpc_500c",
                           "M_star_12", "M_star_14", "SFR_500c", "sSFR_500c",
                           "N_sat_500c", "sat_vel_disp_x_500c", "sat_vel_disp_y_500c", 
                           "sat_vel_disp_z_500c", "sat_vel_disp_500c"]
                           
    for tng_snap in tng_snaps:
        logging.info("\033[1m" +  f"Saving data for TNG300-1 snapshot {tng_snap}" + "\033[0m")
        tng300_halo_file = data_path + f"TNG300_1_halo_catalog_snap{tng_snap}.csv"
        tng300_subhalo_file = data_path + f"TNG300_1_subhalo_catalog_snap{tng_snap}.csv"
        tng_cluster_halo_file = data_path + f"TNG_Cluster_halo_catalog_snap{tng_snap}.csv"
        tng_cluster_subhalo_file = data_path + f"TNG_Cluster_subhalo_catalog_snap{tng_snap}.csv"

        tng300_halo_catalog = TNG_load_and_add_properties(tng300_halo_file, tng300_subhalo_file,
                                                          min_sub_M_star, result_columns = tng_result_columns,
                                                          exclude_subhalo_flag = True, 
                                                          exclude_hierarchical_subhalos = True,
                                                          num_bootstrap = num_bootstrap)
        save_h5_halo_catalog(tng300_halo_catalog, "TNG300_1", tng_snap, min_sub_M_star, num_bootstrap)

        logging.info("\033[1m" +  f"Saving data for TNG-Cluster snapshot {tng_snap}" + "\033[0m")
        tng_cluster_halo_catalog = TNG_load_and_add_properties(tng_cluster_halo_file, tng_cluster_subhalo_file,
                                                               min_sub_M_star, result_columns = tng_result_columns,
                                                               exclude_subhalo_flag = True, 
                                                               exclude_hierarchical_subhalos = True, 
                                                               num_bootstrap = num_bootstrap)
        save_h5_halo_catalog(tng_cluster_halo_catalog, "TNG_Cluster", tng_snap, min_sub_M_star, num_bootstrap)
        
    for flam_snap in flam_snaps:
        logging.info("\033[1m" +  f"Saving data for FLAMINGO_L1000N3600 snapshot {flam_snap}" + "\033[0m")
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