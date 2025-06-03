# These are helper functions for loading data and analysis of gas properties in TNG and FLAMINGO simulations.

import numpy as np
import pandas as pd


def load_and_add_properties(sim, snap, halo_file, result_columns = None):
    '''
    Load multiple data frames within simulations and add two gas properties for X-ray pressure (Y_X_500c) and tSZ pressure (Y_SZ_500c).

    '''

    # Initialize empty array to hold all data frames that will be loaded.
    halos = pd.read_csv(halo_file) # load halos data frame
    if result_columns is None:
        result_colums = halos.columns

    num_halos = len(halos) # number of halos in this data frame
    halos["Y_X_500c"] = halos["M_hot_gas_500c"] + halos["T_sl_500c"] if "FLAM" not in halo_file else halos["M_hot_gas_500c"] + halos["T_sl_wo_recent_AGN_500c"] # add X-ray pressure, Y_X_500c = M_gas_500c * T_sl_500c
    halos["Y_SZ_500c"] = halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_500c"] if "FLAM" not in halo_file else halos["M_hot_gas_500c"] + halos["T_mw_hot_gas_wo_recent_AGN_500c"] # add X-ray pressure, Y_SZ_500c = M_gas_500c * T_mw_500c
    halos = halos[result_columns] # filter to only include columns relevant for this analysis
    halos = halos[~(halos == float("-inf")).any(axis = 1)].sort_values(by = "M_500c", ascending = False).reset_index().drop("index", axis = 1) # get rid of any halos that do not have information for the properties of interest
    num_halos_after_filter = len(halos) # number of halos after filtering

    # Print the number of halos that were dropped after -inf filter.
    print(f"\nNumber of halos that were dropped after '-inf' filter from simulation {sim} snap {snap} : {num_halos - num_halos_after_filter}")
    

    return halos


def find_max_nth_largest_scale(df, n, scale_var):
    '''
    Find the nth largest value in the scale variable given the halos data frame
    
    '''
    
    max_nth_largest_scale = np.sort(df[scale_var])[-n]


    return max_nth_largest_scale