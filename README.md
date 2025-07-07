# Localized Scaling Relations and Mass Proxy Quality (MPQ)

Produce localized scaling relation parameters, covariances, correlations and Mass Proxy Qualities (MPQs) for a set of halo properties.

This repository hosts both data from TNG300-1, TNG-Cluster and FLAMINGO_L1-m8 simulations and scripts to produce the normalization, slope, scatter, pairwise correlation, pairwise covariance, and Mass Proxy Quality for any data set.

## Dependencies

`numpy`, `pandas`, `scipy`, `matplotlib`, `kllr`.

## QUICKSTART

```
from MPQScaling import MPQScaling
import pandas as pd

data_file = "<path/to/data/data.csv>"

data = pd.read_csv(data_file)
properties = ["M_hot_gas_500c", "T_sl_500c", "L_X_500c", "Y_X_500c", "Y_SZ_500c"]

scaling_calc = MPQScaling(properties = properties, scale_var = "M_500c", bins = 20,
                          xrange = None, nBootstrap = 100, percentile = [16., 84.],
                          kernel_type = "gaussian", kernel_width = 0.2, verbose = True)

# Calculate binned scaling parameters
scaling_calc.calculate_scaling_parameters(data)
# Store scaling parameters
scaling_params = scaling_calc.get_scaling_parameters()

# Calculate binned covariance matrix
scaling_calc.calculate_covariance_matrix(data)
# Store covariance
covariance = scaling_calc.get_covariances()

# Find the MPQ of all individual properties.
mpqs_individual = scaling_calc.get_mpq(num_props_in_combination = 1)
# Find the combined MPQ of the properties in 'properties' with index 0 and 1.
mpqs_gas_temp = scaling_calc.get_mpq(combination = [0, 1])

# Calculate binned correlation matrix
scaling_calc.calculate_correlation_matrix(data)
# Store correlation
correlations = scaling_calc.get_correlations()

```

### Caveats

Since the MPQ calculation, equation (4) of [Evrard et. al 2014](https://academic.oup.com/mnras/article/441/4/3562/1217975), involves the covariance and slopes, `MPQScaling.calculate_scaling_parameters(...)` and `MPQScaling.calculate_covariance_matrix(...)` need to be run before running `MPQScaling.get_mpq(...)`.

Scripts that were used two MPQ properties papers are in `MPQ-gas-properties` and `MPQ-stellar-properties` (in prep.) with the analysis data in `.csv` files.

Halo catalogs from IllustrisTNG (TNG-300), TNG-Cluster, and FLAMINGO-L1_m8 are in `halo-catalogs` as `.h5` files.

If you find any errors/bugs in the code, please reach out to ealjamal@umich.edu.