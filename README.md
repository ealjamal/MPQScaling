# Localized Scaling Relations and Mass Proxy Quality (MPQ)

Produce localized scaling relation parameters, covariances, correlations and Mass Proxy Qualities (MPQs) for a set of halo properties.

This repository hosts both data from TNG300-1, TNG-Cluster and FLAMINGO_L1-m8 simulations and scripts to produce the normalization, slope, scatter, pairwise correlation, pairwise covariance, and Mass Proxy Quality for any data set.

## QUICKSTART

```
import Scaling
import ScalingCalculator
import pandas as pd

data_path = "<path/to/data/>"

data = pd.read_csv(data_path)
properties = ["M_hot_gas_500c", "T_sl_500c", "L_X_500c", "Y_X_500c", "Y_SZ_500c"]
combos_mpq_indices = ["01", 01234"]
sim_name = "TNG"

scaling_calc = ScalingCalculator(properties = properties, scale_var = "M_500c", bins = 20,
                                 xrange = None, nBootstrap = 100, percentile = [16., 84.],
                                 kernel_type = "gaussian", kernel_width = 0.2, verbose = True)

kllr_params = Scaling.scaling_params(scaling_calc, data)
covs = Scaling.covariances(scaling_calc, data)
corrs = Scaling.correlations(scaling_calc, data)
mpqs = Scaling.mpq_scaling(scaling_calc, combos_mpq_indices)

```

### Caveats

Since the MPQ calculation, equation (4) of [Evrard et. al 2014](https://academic.oup.com/mnras/article/441/4/3562/1217975), involves the covariance and slopes, `Scaling.scaling_params(...)` and `Scaling.covariances(...)` need to be run before running `Scaling.mpq_scaling(...)`.


If you find any errors/bugs in the code, please reach out to ealjamal@umich.edu.