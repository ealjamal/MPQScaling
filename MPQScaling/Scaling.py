import pandas as pd
import numpy as np
from itertools import combinations


def flatten_lower_triangle_of_binned_matrices(binned_matrices):
    '''
    Convenience function that extracts the covariance/correlation between each pair of
    properties and flattens them to rows where each entry is the correlation
    of a pair of properties at a specific bin. It does this for each pair of 
    properties of interest passed into ScalingCalculator.

    --------
    Params
    --------    

    binned_matrices: (numpy array)
        Must be of shape (# properties, # properties, # bins) that comes
        from the KLLR caclulations of covariance/correlation.

    --------
    Output
    -------- 

    binned_matrices_flat: (numpy array)
        Shape (# properties choose 2, # of bins). Flattened covariances/
        correlations where the second dimension traverses the bins
        and the first dimension is for each pair of properties.

    combos: (list of tuples)
        The possible pairs of indices for the properties that were used
        to calculate the covariances/correlations in the first dimension
        of binned_matrices_flat in order.

    '''

    binned_matrices_flat = []
    num_props = binned_matrices.shape[0]
    combos = list(combinations(range(num_props), 2))

    for combo in combos:
        binned_matrices_flat.append(binned_matrices[combo[0], combo[1], :])

    binned_matrices_flat = np.array(binned_matrices_flat)
        
    return binned_matrices_flat, combos


def indices_to_props(indices, properties):
    '''
    Convenicence function to map an index or a combination of indices to their
    associated property. If there is more than one index, there properties will
    be linked to each other with an underscore.

    --------
    Params
    -------- 

    indices: (str of ints)
        A string containing the integer indices for the properties that are 
        going to be mapped.
    
    properties: (list of str)
        All the properties of interest that are passed to ScalingCalculator.

    --------
    Output
    -------- 

    props: (str)
        A string of properties where different properties are separated by
        underscores.

    '''

    prop_list = []
    for ind in indices:
        prop = properties[int(ind)]
        prop_list.append(prop)

    props = "_".join(prop_list)

    return props

def scaling_params(scaling_calc, data):
    '''
    Calculate KLLR parameters of scale, normalization, slope, scatter (natural
    log units) with errors for each property passed to the ScalingCalculator.

    --------
    Params
    -------- 

    scaling_calc: (ScalingCalculator)
        An instance of ScalingCalculator initialized with properties and KLLR
        settings.

    data: (pd.DataFrame)
        Data frame containing columns that are passed to scaling_calc.

    --------
    Output
    -------- 

    kllr_params_dict: (dict)
        Dictionary containing the scale as well as normalization, slope, and
        scatter for each property. For example for the property "M_gas", the 
        normalization, slope, scatter with their lower and upper percentiles
        can be accessed using kllr_params_dict["M_gas_norm"],
        kllr_params_dict["M_gas_norm-"], kllr_params_dict["M_gas_norm+"],
        kllr_params_dict["M_gas_slope"], kllr_params_dict["M_gas_slope-"],
        kllr_params_dict["M_gas_slope+"], kllr_params_dict["M_gas_scatter"],
        kllr_params_dict["M_gas_scatter-"], kllr_params_dict["M_gas_scatter+"],
        respectively. This is designed to be easily convertible to a 
        data frame.
    
    '''

    num_bins = scaling_calc.bins # number of bins set by ScalingCalculator context.
    bins = np.arange(scaling_calc.bins) # bins array
    scale_var = scaling_calc.scale_var # scale variable set by ScalingCalculator context.
    properties = scaling_calc.properties # all properties passed to ScalingCalculator context.

    scaling_calc.calculate_kllr_parameters(data) # calculate kllr parameters in ScalingCalculator
    kllr_params = scaling_calc.get_kllr_parameters() 

    scale = kllr_params['scale']
    norms = kllr_params['norms']
    norms_minus = kllr_params['norms-']
    norms_plus = kllr_params['norms+']
    slopes = kllr_params['slopes']
    slopes_minus = kllr_params['slopes-']
    slopes_plus = kllr_params['slopes+']
    scatters = kllr_params['scatters']
    scatters_minus = kllr_params['scatters-']
    scatters_plus = kllr_params['scatters+']
    
    # Create data frame with kllr parameters matched to bin and scale and
    # turn it into dictionary.
    kllr_params_data = np.stack([norms, norms_minus, norms_plus,
                                 slopes, slopes_minus, slopes_plus,
                                 scatters, scatters_minus, scatters_plus],
                                axis = 1).reshape(-1, num_bins)
    
    # Define columns for the kllr parameters.
    kllr_params_columns = []
    for prop in properties:
        kllr_params_columns = kllr_params_columns + [prop + "_norm"] + [prop + "_norm-"] + [prop + "_norm+"] + \
                                                    [prop + "_slope"] + [prop + "_slope-"] + [prop + "_slope+"] + \
                                                    [prop + "_scatter"] + [prop + "_scatter-"] + [prop + "_scatter+"]
        
    kllr_params_df = pd.DataFrame(data = kllr_params_data.T, columns = kllr_params_columns)
    kllr_params_df.insert(0, "bin", bins)
    kllr_params_df.insert(1, f"{scale_var}", scale)

    kllr_params_dict = kllr_params_df.to_dict()


    return kllr_params_dict


def covariances(scaling_calc, data):
    '''
    Calculate binned pairwise property covariance.

    --------
    Params
    -------- 

    scaling_calc: (ScalingCalculator)
        An instance of ScalingCalculator initialized with properties and KLLR
        settings.

    data: (pd.DataFrame)
        Data frame containing columns that are passed to scaling_calc.

    --------
    Output
    -------- 

    kllr_params_dict: (dict)
        Dictionary containing the pairwise property covariances. For example,
        if we want to get the covariance between "M_gas" and "M_star" with the
        lower and upper percentiles, we would use cov_dict["M_gas_M_star_cov"],
        cov_dict["M_gas_M_star_cov-"], cov_dict["M_gas_M_star_cov+"], 
        respectively. This is designed to be easily convertible to a 
        data frame.
       
    '''
    
    num_bins = scaling_calc.bins # number of bins set by ScalingCalculator context.
    bins = np.arange(scaling_calc.bins) # bins array
    scale_var = scaling_calc.scale_var # scale variable set by ScalingCalculator context.
    properties = scaling_calc.properties # all properties passed to ScalingCalculator context.

    scaling_calc.calculate_covariance_matrix(data) # calculate binned covariance matrices.
    covariances = scaling_calc.get_covariance()
    scale = covariances["scale"]
    cov = covariances['cov']
    cov_minus = covariances['cov-']
    cov_plus = covariances['cov+']
    
    # Turn covariance matrices to rows to be stored in data frame.
    cov_flat, combos = flatten_lower_triangle_of_binned_matrices(cov)
    cov_minus_flat, _ = flatten_lower_triangle_of_binned_matrices(cov_minus)
    cov_plus_flat, _ = flatten_lower_triangle_of_binned_matrices(cov_plus)

    # Create data frame with pairwise property covariances matched to bin and
    # scale and turn it into dictionary.
    cov_data = np.stack([cov_flat, cov_minus_flat, cov_plus_flat], axis = 1).reshape(-1, num_bins)
    # Create columns for covariance data frame.
    cov_columns = []
    for combo in combos:
        cov_columns = cov_columns + [f"{properties[combo[0]]}_" f"{properties[combo[1]]}_cov"] + \
                                      [f"{properties[combo[0]]}_" f"{properties[combo[1]]}_cov-"] + \
                                      [f"{properties[combo[0]]}_" f"{properties[combo[1]]}_cov+"]

    cov_df = pd.DataFrame(data = cov_data.T, columns = cov_columns)
    cov_df.insert(0, "bin", bins)
    cov_df.insert(1, f"{scale_var}", scale)    

    cov_dict = cov_df.to_dict()

    
    return cov_dict


def correlations(scaling_calc, data):
    '''
    Calculate binned pairwise property correlations.

    --------
    Params
    -------- 

    scaling_calc: (ScalingCalculator)
        An instance of ScalingCalculator initialized with properties and KLLR
        settings.

    data: (pd.DataFrame)
        Data frame containing columns that are passed to scaling_calc.

    --------
    Output
    -------- 

    kllr_params_dict: (dict)
        Dictionary containing the pairwise property correlations. For example,
        if we want to get the correlation between "M_gas" and "M_star" with the
        lower and upper percentiles, we would use cov_dict["M_gas_M_star_corr"],
        cov_dict["M_gas_M_star_corr-"], cov_dict["M_gas_M_star_corr+"], 
        respectively. This is designed to be easily convertible to a 
        data frame.
       
    '''
    
    num_bins = scaling_calc.bins # number of bins set by ScalingCalculator context.
    bins = np.arange(scaling_calc.bins) # bins array
    scale_var = scaling_calc.scale_var # scale variable set by ScalingCalculator context.
    properties = scaling_calc.properties # all properties passed to ScalingCalculator context.

    scaling_calc.calculate_correlation_matrix(data) # calculate binned correlation matrices.
    correlations = scaling_calc.get_correlation()
    scale = correlations["scale"]
    corr = correlations['corr']
    corr_minus = correlations['corr-']
    corr_plus = correlations['corr+']
    
    # Turn correlation matrices to rows to be stored in data frame.
    corr_flat, combos = flatten_lower_triangle_of_binned_matrices(corr)
    corr_minus_flat, _ = flatten_lower_triangle_of_binned_matrices(corr_minus)
    corr_plus_flat, _ = flatten_lower_triangle_of_binned_matrices(corr_plus)

    # Create data frame with pairwise property correlations matched to bin and
    # scale and turn it into dictionary.
    corr_data = np.stack([corr_flat, corr_minus_flat, corr_plus_flat], axis = 1).reshape(-1, num_bins)
    # Define columns for correlations data frame.
    corr_columns = []
    for combo in combos:
        corr_columns = corr_columns + [f"{properties[combo[0]]}_" f"{properties[combo[1]]}_corr"] + \
                                      [f"{properties[combo[0]]}_" f"{properties[combo[1]]}_corr-"] + \
                                      [f"{properties[combo[0]]}_" f"{properties[combo[1]]}_corr+"]

    corr_df = pd.DataFrame(data = corr_data.T, columns = corr_columns)
    corr_df.insert(0, "bin", bins)
    corr_df.insert(1, f"{scale_var}", scale)

    corr_dict = corr_df.to_dict()        

    
    return corr_dict


def mpq_scaling(scaling_calc, combos_mpq_indices):
    '''
    Calculate the individual property MPQ along with the MPQ of combinations
    of properties of the user's choice.
    
    Note, the scaling_params, and covariances function must be applied first 
    before applying this function because the kllr_params and covariances
    must be calculated within scaling_calc in order to calculate MPQ.


    --------
    Params
    -------- 

    scaling_calc: (ScalingCalculator)
        An instance of ScalingCalculator initialized with properties and KLLR
        settings.

    combos_mpq_indices: (list of strings containing ints)
        A string of indices of the properties that will be used to calculate
        the MPQ implied by the combination of these properties. For example,
        if we have properties = ["M_gas", "M_star", "M_BH"], and we want MPQs
        of "M_gas" + "M_star", "M_gas" + "M_BH", and "M_gas" + "M_star" + "M_BH",
        we would pass combos_mpq_indices = ["01", "02", "012]

    --------
    Output
    -------- 

    mpq_dict: (dict)
        Dictionary containing the individual mpqs as well as the mpqs of requested
        combination of properties. The individual mpqs with lower and upper
        percentiles can be accessed by mpq_dict["M_gas_mpq"], 
        mpq_dict["M_gas_mpq-"], mpq_dict["M_gas_mpq+"]. The MPQ of the
        combined properties can be accessed with
        mpq_dict["M_gas_M_star_mpq"], mpq_dict["M_gas_M_star_mpq-"],
        mpq_dict["M_gas_M_star_mpq+"]. This is designed to be easily convertible
        to a data frame.
       
    '''

    num_bins = scaling_calc.bins # number of bins set by ScalingCalculator context.
    bins = np.arange(scaling_calc.bins) # bins array
    scale_var = scaling_calc.scale_var # scale variable set by ScalingCalculator context.
    scale = scaling_calc.scale # binned scale
    properties = scaling_calc.properties # all properties passed to ScalingCalculator context.

    # Calculate MPQs of individual properties.
    mpq_individual = scaling_calc.mpq_combo(num_props_in_combo = 1)
    # Calculate MPQ for the combination of properties requested.
    mpq_combo_dict = dict()
    for combo_mpq_inds in combos_mpq_indices:
        num_prop_in_combo = len(combo_mpq_inds)
        prop = indices_to_props(combo_mpq_inds, properties)
        mpq_combo = scaling_calc.mpq_combo(num_props_in_combo = num_prop_in_combo)
        mpq_combo_dict[f"{prop}_mpq"] = mpq_combo[f"mpq_{combo_mpq_inds}"].reshape(num_bins,)
        mpq_combo_dict[f"{prop}_mpq-"] = mpq_combo[f"mpq-_{combo_mpq_inds}"].reshape(num_bins,)
        mpq_combo_dict[f"{prop}_mpq+"] = mpq_combo[f"mpq+_{combo_mpq_inds}"].reshape(num_bins,)

    # Create data frame with MPQs matched to bin and scale and turn it into dictionary.
    mpq_individual_data = np.stack([[mpq_individual[f"mpq_{i}"] for i in range(len(properties))],
                                    [mpq_individual[f"mpq-_{i}"] for i in range(len(properties))], 
                                    [mpq_individual[f"mpq+_{i}"] for i in range(len(properties))]],
                                    axis = 1).reshape(-1, num_bins)
    
    # Define MPQ columns for individual MPQ data frame.
    mpq_individual_columns = []
    for prop in properties:
        mpq_individual_columns = mpq_individual_columns + [prop + "_mpq"] + [prop + "_mpq-"] + [prop + "_mpq+"]

    mpq_individual_df = pd.DataFrame(data = mpq_individual_data.T, columns = mpq_individual_columns)
    mpq_individual_df.insert(0, "bin", np.arange(num_bins))
    mpq_individual_df.insert(1, f"{scale_var}", scale)

    # Create the combination MPQ data frame and merge with individual MPQ
    # data frame to have all MPQs in a single data frame and dictionary.
    mpq_combo_df = pd.DataFrame(mpq_combo_dict)
    mpq_combo_df.insert(0, "bin", bins)
    mpq_df = mpq_individual_df.merge(mpq_combo_df, on = "bin")

    mpq_dict = mpq_df.to_dict()

    
    return mpq_dict