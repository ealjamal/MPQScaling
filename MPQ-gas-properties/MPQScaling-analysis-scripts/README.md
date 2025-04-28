# Scaling and MPQ analysis scripts for Illustris-TNG300-1, Illustris-TNG-Cluster, FLAMINGO-L1_m8 simulations

These are the scripts used to load the Scaling and analysis data of halo gas properties in presented in `../Data` directory. 

These scripts take in a snapshot argument and save the binned scaling parameters (normalization, slope, scatter), pairwise property covariances, pairwise property correlations, and mass proxy qualities of the individual properties as well as the combination of the core five halo properties of that snapshot. These core gas properties are: hot gas mass, spectroscopic-like temperature, X-ray luminosity, X-ray pressure, and thermal pressure.

The halo properties used in this analysis as well as which MPQ combination is saved can be altered for specific science goals by changing `props` and `combos_mpq_indices`, respectively.

The description of the data loaded by these scripts is described in `../Data/READ.md`.