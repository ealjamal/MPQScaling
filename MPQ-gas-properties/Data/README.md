# Scaling and MPQ analysis data for Illustris-TNG300-1, Illustris-TNG-Cluster, FLAMINGO-L1_m8 simulations

## Discription of simulations

We use the highest resolution run of the Illustris-TNG ~300 Mpc box simulation (TNG300-1). We also use the ~1 Gpc zoom-in simulation of Illustris-TNG (TNG-Cluster). We combine TNG300-1 and TNG-Cluster halo catalogues and analyze it under the name TNG300_1+TNG_Cluster.  Lastly, we use the FLAMINGO 1 Gpc box simulation with the highest resolution (FLAMINGO-L1_m8).

The halo catalogues used for this analysis are available by request through email: ealjamal@umich.edu. For both the TNG300-1 + TNG_Cluster and FLAMINGO-L1_m8 halo catalogues, we run The analysis for 4 snapshots/redshifts. For TNG300-1 + TNG-Cluster, these are: 33/2, 50/1, 67/0.5, 99/0. For FLAMINGO-L1_m8, these are: 38/2, 58/1, 68/0.5, 78/0 and include only halos with $log(M_{500c}/M_{\odot}) \geq 12.5$.

## Discription of analysis data

Naming convention for simulations: TNG300-1 and TNG-Cluster are named as such, but  FLAMINGO-L1_m8 is named as "FLAM_L1000N3600" where L1000 denotes the side length of 1000 Mpc for the simulation box and N3600 indicates 3600^3 total baryonic particles.

For the TNG300-1 halo catalogue (without combining TNG-Cluster), we only have data for z = 0 (snapshot 99).

### Halo gas properties included in analysis

We have scaling parameters, covariances, correlations and MPQs for seven gas properties with the scale variable being $M_{500c}$, the total mass within $R_{500c}$.		

1. Hot gas mass ($T \geq 10^6$ K) in $R_{500c}$ ("M_hot_gas_mass_500c"),

2. Spectroscopic-like temperature of gas particles with $k_{\rm B}T > 0.1$ keV in $R_{500c}$ ("T_sl_500c" in TNG, "T_sl_wo_recent_AGN_500c" in FLAMINGO as we get rid of recently heated AGN cells),

3. Observer name X-ray luminosity ROSAT band $[0.5-2.0]$ keV in $R_{500c}$ ("L_X_ROSAT_obs_500c" in TNG, "L_X_ROSAT_obs_wo_recent_AGN_500c" in FLAMINGO as we get rid of recently heated AGN cells),

4. X-ray pressure in $R_{500c}$ ("Y_X_500c") defined as the product of the hot gas mass and the spectroscopic-like temperature,

5. Thermal pressure in $R_{500c}$ ("Y_SZ_500c") defined as the product of the hot gas mass and the mass-weighted average temperature,

6. Mass-weighted average temperature of hot gas particles ($T \geq 10^6$ K) in $R_{500c}$ ("T_mw_hot_gas_500c" in TNG, "T_mw_hot_gas_wo_recent_AGN_500c" in FLAMINGO),

7. Core-excised X-ray luminosity in ROSAT band $[0.5-2.0]$ keV in $R_{500c}$ ("L_X_ROSAT_obs_ce_500c" in TNG, "L_X_ROSAT_obs_wo_recent_AGN_ce" in FLAMINGO).
        
### Analysis settings

Kernel Localized Linear Regression (KLLR) settings used in the analysis:

	* bins (number of bins)  = 20,

	* nBootstrap (number of bootstrap realizations) = 1000,

	* xrange (range of the scale variable used in the analysis in dex) = (13, mass of 21st most massive halo in catalogue),

	* percentile (the lower and upper bootstrap percentiles) = [16., 84.],

	* kernel_type (type of kernel to calculate weights) = 'gaussian',

	* kernel width (the standard deviation of the kernel) = 0.2 dex,

	* scatter_factor (set to display scatter in natural log) = numpy.log(10).

### Analysis Files

For each simulation and snapshot, there are 4 .csv files:
1. Scaling parameters: These are .csv files named as such gas_scaling_parameters_{simulation}_snap{snapshot}.csv. These contain the binned mass as well as the normalization, slope, and scatter of each property. Note, the scatter is given as the scatter in the natural log of properties.

2. Property covariance: These are .csv files name as such gas_covariances_{simulation}_snap{snapshot}.csv. These contain, the pairwise property covariances.

3. Property covariance: These are .csv files name as such gas_correlations_{simulation}_snap{snapshot}.csv. These contain, the pairwise property correlations.

4. Mass proxy Quality (MPQ): These are .csv files name as such gas_MPQ_{simulation}_snap{snapshot}.csv. These contain, the individual property MPQs as well as the combined MPQs of the first five properties listed above Note, the mass scatter is given as the scatter in the natural log of halo mass (M500c).