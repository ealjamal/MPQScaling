# Scaling and MPQ analysis scripts for IllustrisTNG (TNG300-1), TNG-Cluster, FLAMINGO-L1_m8 simulations

These are the scripts used to analyze the IllusrisTNG (TNG-300-1), TNG-Cluster, FLAMINGO-L1_m8 halo catalogues in `halo-catalogs` to produce scaling parameters (normalization, slope, intrinsic scatter), covariances, correlations, and mass-proxy qualities (MPQs) that are saved (after a bit of cleaning) in `Data` directory. 

These scripts take in a snapshot argument and save the binned scaling parameters (normalization, slope, scatter), pairwise property covariances, pairwise property correlations, and MPQs of the individual properties as well as the combination of the core five halo properties of that snapshot. These core gas properties are: hot gas mass, spectroscopic-like temperature, soft-band X-ray luminosity, X-ray thermal energy, and tSZ thermal energy. Other properties such as mass-weighted average temperature, core-excised soft-band X-ray luminosity are also present. For FLAMINGO, properties without the exclusion of cells that are recently-heated AGN feedback are present.

The halo properties used in this analysis as well as which MPQ combination is saved can be altered for specific science goals by changing `props` and `combos_mpq_indices`, respectively.

The description of the data loaded by these scripts is described in `/Data/README.md`.