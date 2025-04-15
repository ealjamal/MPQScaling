# This will analyze the scaling parameters, covariances, correlations, and MPQs 
# of halos in the TNG300-1 + TNG-Cluster combined halo catalogue.

import pandas as pd
import numpy as np
from MPQScaling import MPQScaling
from gas_data_loading_functions import load_and_add_columns, find_max_nth_largest_scale
from time import time
import sys


def main():
    ##### Helpful variables for data loading

    data_path = "/Volumes/external-hard/cosmo-research/gas-properties-paper/data/" # path to halo catalogues
    sim_name = "TNG300_1+TNG_Cluster" # simulation name that will be used to save .csv files
    snap = int(sys.argv[1]) # snapshot for simulations
    halo_file_tng300 = data_path + f"TNG300_1_halo_catalog_snap{snap}.csv" # path for TNG300-1 halos
    halo_file_tng_clus = data_path + f"TNG_Cluster_halo_catalog_snap{snap}.csv" # path for TNG-Cluster halos
    # Columns that will be part of the analysis
    columns = ["M_500c", "M_hot_gas_500c", "T_sl_500c", "L_X_ROSAT_obs_500c", "Y_X_500c", 
               "Y_SZ_500c", "T_mw_hot_gas_500c", "L_X_ROSAT_obs_ce_500c"]
    
    print("\n\033[1mPreprocessing data...\033[0m")

    ##### Load Halo Catalogue

    # Load TNG300-1 halo catalogue of provided snapshot
    data_tng300 = load_and_add_columns(sim = "TNG300_1", snap = snap, halo_file = halo_file_tng300, result_columns = columns)
    # Load TNG-Cluster halo catalogue of provided snapshot
    data_tng_clus = load_and_add_columns(sim = "TNG_Cluster", snap = snap, halo_file = halo_file_tng_clus, result_columns = columns)
    # Concatenate TNG300-1 and TNG-Cluster halo catalogues
    data = pd.concat([data_tng300, data_tng_clus])

    ##### MPQScaling analysis

    # Define the scale variable as M500c and extract mass of the 21st most massive halo
    top_halo_num, scale_var = 20, "M_500c"
    mass_top_nth = find_max_nth_largest_scale(data, top_halo_num, scale_var)

    ##### KLLR settings

    xrange = (13, mass_top_nth) # KLLR analysis mass range, lower mass is 10^{13}Msun and top mass 
                                # is mass of 21st most massive halo
    bins = 20 # number of KLLR bins
    nBootstrap = 1000 # number of bootstrap samples to use to compute percentiles
    percentile = [16., 84.] # Upper and lower percentile ranges
    kernel_type = "gaussian" # gaussian kernel for the KLLR analysis
    kernel_width = 0.2 # width of the gaussian kernel
    scatter_factor = np.log(10) # scatter and covariance will be reported in natural log of properties
    max_iter_cov_search = 10000 # number of iteration when sampling covariance if we can't find symm. positive definite
    # Properties that will be used for analysis, we will produce normalization slope, scatter, pairwise covariances, 
    # and pairwise correlations of these properties
    props = ["M_hot_gas_500c", "T_sl_500c", "L_X_ROSAT_obs_500c", "Y_X_500c",
             "Y_SZ_500c", "T_mw_hot_gas_500c", "L_X_ROSAT_obs_ce_500c"]
    # We will calculate the the MPQ that results from the combined properties with the following indices,
    # thus, we calculate the combined MPQ of M_hot_gas_500c, T_sl_500c, L_X_ROSAT_obs_500c, Y_X_500c, and Y_SZ_500c
    combos_mpq_indices = [0, 1, 2, 3, 4]

    print(f"\n{scale_var} xrange: {xrange}")

    ##### MPQScaling analysis

    # Intialize MPQScaling context for this analysis
    mpq_scaling = MPQScaling(properties = props, scale_var = "M_500c", bins = bins, xrange = xrange,
                             nBootstrap = nBootstrap, percentile = percentile, kernel_type = kernel_type,
                             kernel_width = kernel_width, scatter_factor  = scatter_factor, verbose = True,
                             max_iter_cov_search = max_iter_cov_search)
    
    # Calculate the scaling parameters
    mpq_scaling.calculate_scaling_parameters(data)
    # Calculate the covariance matrix
    mpq_scaling.calculate_covariance_matrix(data)
    # Calculate the correlation matrix
    mpq_scaling.calculate_correlation_matrix(data)

    # Retrieve binned mass and scaling parameters and define a data frame containing them and save
    scaling_params = mpq_scaling.get_scaling_parameters()
    scaling_params_df = pd.DataFrame(scaling_params)
    scaling_params_df.to_csv(f"gas_scaling_parameters_{sim_name}_snap{snap}.csv", index = False)

    # Retrieve binned mass and pairwise property covariance and define a data frame containing them and save
    covs = mpq_scaling.get_covariances()
    covs_df = pd.DataFrame(covs)
    covs_df.to_csv(f"gas_covariances_{sim_name}_snap{snap}.csv", index = False)

    # Retrieve binned mass and pairwise property correlations and define a data frame containing them and save
    corrs = mpq_scaling.get_correlations()
    corrs_df = pd.DataFrame(corrs)
    corrs_df.to_csv(f"gas_correlations_{sim_name}_snap{snap}.csv", index = False)

    # Calculate the MPQs for the individual properties in 'props'
    mpq_individual = mpq_scaling.get_mpq(num_props_in_combination = 1)
    # Calculate the MPQs for the combined properties M_hot_gas_500c, T_sl_500c,
    # L_X_ROSAT_obs_500c, Y_X_500c, and Y_SZ_500c
    mpq_combo = mpq_scaling.get_mpq(combination = combos_mpq_indices)
    # Create data frame containing the MPQs of individual properties
    mpq_ind_df = pd.DataFrame(mpq_individual)
    # Create data frame containing the MPQ of the combined properties
    mpq_combo_df = pd.DataFrame(mpq_combo)
    mpq_combo_df.drop("M_500c", axis = 1, inplace = True)
    # Merge the the individual MPQ data frame with the combined MPQ data frame and save
    mpq_df = mpq_ind_df.merge(mpq_combo_df, on = "bin")
    mpq_df.to_csv(f"gas_MPQ_{sim_name}_snap{snap}.csv", index = False)


if __name__ == "__main__":
    # Time the duration of the analysis
    start = time()
    main()
    end = time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\n\nDone! Total time elapsed {hours:02d}:{minutes:02d}:{seconds:02d}\n")