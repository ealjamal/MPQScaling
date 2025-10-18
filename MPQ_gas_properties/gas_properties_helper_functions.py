# These are helper functions for loading data and analysis of gas properties in TNG and FLAMINGO simulations.

import numpy as np
import pandas as pd
import h5py

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


def find_max_nth_largest_scale(df, n, scale_var):
    '''
    Find the nth largest value in the scale variable given the halos data frame.

    --------
    Params
    --------

    df (pandas.DataFrame):
        Halo catalog data frame.
    
    n (integer):
        The function will return the scale_var with the nth largest value.

    scale_var (string):
        Column in the halo catalog data frame that we want to find the
        nth largest value for.

    --------
    Output
    --------

    max_nth_largest_scale (float):
        The nth largest value of the scale_var.
    
    '''
    
    max_nth_largest_scale = np.sort(df[scale_var])[-n]

    return max_nth_largest_scale