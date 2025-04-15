# This a script to load halo (SO) data and subhalo data from the FLAMINGO simulations
# with support for parallelization.

import numpy as np
import pandas as pd
import swiftsimio as sw
import unyt as u
from astropy.cosmology import w0waCDM, z_at_value
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import warnings
import sys
import logging

##### Initializations, file paths and helpful variables

simulation = "L1000N3600" # simulation
num_jobs = int(sys.argv[1]) # number of cpus
snap = f"{int(sys.argv[2]):04d}" # snapshot for which to load data
min_sub_star_mass = float(sys.argv[3]) # minimum subhalo stellar mass
file_path = f"/cosma8/data/dp004/flamingo/Runs/{simulation}/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_{snap}.hdf5"
virtual_path = f"./particle-files-{simulation}-snap-{snap}/flamingo_L1000N3600_{snap}.hdf5" # file path for virtual hdf5 file for particle data (this uses make_virtual_snapshot.py in swiftsimio github)
# Define constants for mask initialization in order to only define the virtual_path mask for each worker not for each halo for efficiency.
mask_initialized = False
mask = None

##### Analysis-specific parameters

# These can be changed in order to produce data of interest for a project.
min_M500c = 12.5 # minimum halo mass in R_500c to have in data. units: M_sun
max_M500c = float("inf") # maximum halo mass in R_500c to have in data. units: M_sun
cold_gas_temperature = 1e4 # upper bound on temperature for gas to be considered cold. units: Kelvin.
inner_shell_inner_rad  = 0.15 # inner radius of inner shell in R_500c units
inner_shell_outer_rad = 0.35 # outer radius of inner shell in R_500c units
mid_shell_inner_rad = 0.5 # inner radius of mid shell in R_500c units
mid_shell_outer_rad = 0.7 # outer radius of mid shell in R_500c units
outer_shell_inner_rad = 0.8 # inner radius of outer shell in R_500c units
outer_shell_outer_rad = 1 # outer radius of outer shell in R_500c units

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

#### Helper function

def recently_heated_a_limit():
    '''Calculate the scale expansion factor that is the limit for recent AGN feedback, a_limit. Each particle has a tag for the
       scale expansion factor denoting when it was last heated by an AGN. Any such tag less than a_limit will be 
       considered as recently heated by an AGN. However, that is not the only case that is necessary for a particle
       to be excluded because it was recently heated by AGN, there are also temperature conditions that are listed in
       the calculate_halo_quantities function.
    '''

    particle_data = sw.load(virtual_path) # load particle data
    # Extract cosmological parameters
    H0 = particle_data.metadata.cosmology.H0
    Omega_b0 = particle_data.metadata.cosmology.Ob0
    Omega_de0 = particle_data.metadata.cosmology.Ode0
    Omega_m0 = particle_data.metadata.cosmology.Om0
    w_0 = particle_data.metadata.cosmology.w0
    w_a = particle_data.metadata.cosmology.wa
    Tcmb0 = particle_data.metadata.cosmology.Tcmb0
    z_now = particle_data.metadata.z
    # Define cosmology using the cosmological parameters
    cosmology = w0waCDM(
            H0 = H0,
            Om0 = Omega_m0,
            Ode0 = Omega_de0,
            w0 = w_0,
            wa = w_a,
            Tcmb0 = Tcmb0,
            Ob0 = Omega_b0,
        ) 
    
    lookback_time_now = particle_data.metadata.cosmology.lookback_time(z_now) # look back time at the redshift of this snapshot
    delta_recent_time = 15 * u.Myr # define our time cutoff for recently heated  
    lookback_time_limit = lookback_time_now + delta_recent_time.to_astropy() # look back time limit for recently heated time cutoff
    z_limit = z_at_value(cosmology.lookback_time, lookback_time_limit) # redshift limit to be considered recently heated
    # Check if z_limit has units
    if hasattr(z_limit, "value"):
            z_limit = z_limit.value

    a_limit = 1.0 / (1.0 + z_limit) # calculate scale expansion factor limit, a_limit, from the redshift limit.


    return a_limit

a_limit = recently_heated_a_limit() # Set the a_limit as a constant for this snapshot

#### Data Loading Functions

def load_halo_subhalo_df():
    '''Load halo and subhalo data frames with properties already defined in the SOAP catalogue.
    '''

    # Start timer for loading data frames
    start_time = time.time() 
    data = sw.load(file_path) # subhalo data
    redshift = data.metadata.z
    # Input halo quantities
    logging.info(f"\n\nLoading data for simulation {simulation} snapshot {snap} and redshift = {redshift:0.2f}\n")
    # Halo Properties
    halo_cat_index = data.input_halos.halo_catalogue_index.value # subhalo ID in the original catalogue
    host_ID = data.input_halos_hbtplus.host_fofid.value # ID of the host of this subhalo.
    is_central = data.input_halos.is_central.value # 1 if this is a central subhalo of the FOF and 0 if not
    sub_pos_comoving = data.input_halos.halo_centre.in_units(u.Mpc) # comoving position of the most bound particle of this subhalo
    sub_pos_physical = sub_pos_comoving.to_physical().value # physical position of the most bound particle of this subhalo

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        # Spherical overdensity quantities
        M_500c = np.log10(data.spherical_overdensity_500_crit.total_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun
        R_500c_comoving = data.spherical_overdensity_500_crit.soradius.in_units(u.Mpc) # units: Mpc
        R_500c_physical = R_500c_comoving.to_physical().value # units: Mpc
        M_200c = np.log10(data.spherical_overdensity_200_crit.total_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun
        R_200c_comoving = data.spherical_overdensity_200_crit.soradius.in_units(u.Mpc) # units: Mpc
        R_200c_physical = R_200c_comoving.to_physical().value # units: Mpc
        L_X_ROSAT_obs_wo_recent_AGN_500c = np.log10(data.spherical_overdensity_500_crit.xray_luminosity_without_recent_agnheating[:, 2].to_physical().in_units(u.erg / u.second).value) # units: erg/s
        L_X_ROSAT_obs_wo_recent_AGN_500c_ce = np.log10(data.spherical_overdensity_500_crit.xray_luminosity_without_recent_agnheating_core_excision[:, 2].to_physical().in_units(u.erg / u.second).value) # units: erg/gs
        L_X_ROSAT_rest_wo_recent_AGN_500c = np.log10(data.spherical_overdensity_500_crit.xray_luminosity_in_restframe_without_recent_agnheating[:, 2].to_physical().in_units(u.erg / u.second).value) # units: erg/s
        gas_mass_500c = np.log10(data.spherical_overdensity_500_crit.gas_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun
        hot_gas_mass_500c = np.log10(data.spherical_overdensity_500_crit.hot_gas_mass.to_physical().in_units(u.Solar_Mass).value) # hot gas has temperature > 1e5 K units: Msun
        stellar_mass_500c = np.log10(data.spherical_overdensity_500_crit.stellar_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun
        bh_mass_500c = np.log10(data.spherical_overdensity_500_crit.black_holes_subgrid_mass.to_physical().in_units(u.Solar_Mass).value) # Black Hole subgrid mass. units: Msun
        mass_avg_gas_temp_500c = np.log10(data.spherical_overdensity_500_crit.gas_temperature.to_physical().in_units(u.K).value) # units: K
        mass_avg_hot_gas_temp_500c = np.log10(data.spherical_overdensity_500_crit.gas_temperature_without_cool_gas.to_physical().in_units(u.K).value) # units: K
        mass_avg_hot_gas_temp_wo_recent_AGN_500c = np.log10(data.spherical_overdensity_500_crit.gas_temperature_without_cool_gas_and_recent_agnheating.to_physical().in_units(u.K).value) # units: K
        spect_like_temp_gas_500c = np.log10(data.spherical_overdensity_500_crit.spectroscopic_like_temperature_core_excision.to_physical().in_units(u.K).value) # units: K
        spect_like_temp_gas_wo_recent_AGN_500c = np.log10(data.spherical_overdensity_500_crit.spectroscopic_like_temperature_without_recent_agnheating_core_excision.to_physical().in_units(u.K).value) # units: K
        SFR = np.log10(data.spherical_overdensity_500_crit.star_formation_rate.to_physical().in_units(u.Solar_Mass / u.yr).value) # units: Msun / yr


        # Exclusive sphere quantities
        BCG_stellar_mass_100kpc = np.log10(data.exclusive_sphere_100kpc.stellar_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun
        BCG_stellar_mass_30kpc = np.log10(data.exclusive_sphere_30kpc.stellar_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun

        # Bound subhalo quantities
        sub_stellar_mass = np.log10(data.bound_subhalo.stellar_mass.to_physical().in_units(u.Solar_Mass).value) # units: Msun
        sub_stellar_half_mass_rad_comoving = data.bound_subhalo.half_mass_radius_stars.in_units(u.Mpc) # comoving stellar half mass radius. units: Mpc
        sub_stellar_half_mass_rad_physical = sub_stellar_half_mass_rad_comoving.to_physical().value # physical stellar half mass radius. units: Mpc
        sub_com_vel = data.bound_subhalo.centre_of_mass_velocity.to_physical().in_units(u.km / u.second).value # subhalo COM velocity. units: km/s

    sub_pos_comoving_x = sub_pos_comoving[:, 0].value
    sub_pos_comoving_y = sub_pos_comoving[:, 1].value
    sub_pos_comoving_z = sub_pos_comoving[:, 2].value
    sub_pos_physical_x = sub_pos_physical[:, 0]
    sub_pos_physical_y = sub_pos_physical[:, 1]
    sub_pos_physical_z = sub_pos_physical[:, 2]

    sub_com_vel_x = sub_com_vel[:, 0]
    sub_com_vel_y = sub_com_vel[:, 1]
    sub_com_vel_z = sub_com_vel[:, 2]

    # Create subhalo DataFrame using the above subhalo data extracted from the sim.
    df_values = [halo_cat_index, host_ID, is_central, sub_pos_comoving_x, sub_pos_comoving_y, sub_pos_comoving_z,
                 sub_pos_physical_x, sub_pos_physical_y, sub_pos_physical_z,
                 sub_com_vel_x, sub_com_vel_y, sub_com_vel_z,
                 M_200c, R_200c_comoving.value, R_200c_physical,
                 M_500c, R_500c_comoving.value, R_500c_physical,
                 L_X_ROSAT_obs_wo_recent_AGN_500c, L_X_ROSAT_obs_wo_recent_AGN_500c_ce, L_X_ROSAT_rest_wo_recent_AGN_500c,
                 gas_mass_500c, hot_gas_mass_500c, stellar_mass_500c, bh_mass_500c,
                 mass_avg_gas_temp_500c, mass_avg_hot_gas_temp_500c, mass_avg_hot_gas_temp_wo_recent_AGN_500c, spect_like_temp_gas_500c, spect_like_temp_gas_wo_recent_AGN_500c,
                 BCG_stellar_mass_100kpc, BCG_stellar_mass_30kpc,
                 sub_stellar_mass, sub_stellar_half_mass_rad_comoving.value, sub_stellar_half_mass_rad_physical, SFR]
    
    df_keys = ["halo_cat_ID", "host_ID", "is_central", "sub_pos_comoving_x", "sub_pos_comoving_y", "sub_pos_comoving_z",
               "sub_pos_physical_x", "sub_pos_physical_y", "sub_pos_physical_z", 
               "sub_com_vel_x", "sub_com_vel_y", "sub_com_vel_z", 
               "M_200c", "R_200c_comoving", "R_200c_physical",  
               "M_500c", "R_500c_comoving", "R_500c_physical",
               "L_X_ROSAT_obs_wo_recent_AGN_500c", "L_X_ROSAT_obs_wo_recent_AGN_ce_500c", "L_X_ROSAT_rest_wo_recent_AGN_500c",
               "M_gas_500c", "M_hot_gas_500c", "M_star_500c", "M_bh_500c",
               "T_mw_500c", "T_mw_hot_gas_500c", "T_mw_hot_gas_wo_recent_AGN_500c", "T_sl_500c", "T_sl_wo_recent_AGN_500c",
               "M_star_BCG_100kpc", "M_star_BCG_30kpc", 
               "sub_M_star", "sub_stellar_half_mass_rad_comoving", "sub_stellar_half_mass_rad_physical", "SFR_500c"]
    
    df_subs = pd.DataFrame(dict(zip(df_keys, df_values)))

    # In FLAMINGO, they keep track of halos, spherical overdensity (SO) properties and exclusive sphere properties (ES) are calculated 
    # for all subhalos and can be trasferred to halos by considering only the central subhalos. The SO and ES properties then describe
    # the parent halo of the cetnral subhalo and the subhalo properties describe the central subhalo.
    df_halos = df_subs[df_subs["is_central"] == 1].copy() # 
    df_halos = df_halos[(df_halos['M_500c'] >= min_M500c) & (df_halos['M_500c'] <= max_M500c)].copy()
    df_halos.set_index(np.arange(len(df_halos)), inplace = True)
    df_halos.sort_values(by = "M_500c", ascending = False, inplace = True)

    # We get rid of the SO columns for the subhalo data because they only make sense for centrals and we get rid of subhalos with small stellar mass.
    sub_columns = ["halo_cat_ID", "host_ID", "is_central", "sub_pos_physical_x", "sub_pos_physical_y", "sub_pos_physical_z",
                   "sub_com_vel_x", "sub_com_vel_y", "sub_com_vel_z", "sub_M_star", "sub_stellar_half_mass_rad_physical"]
    df_subs = df_subs[df_subs["sub_M_star"] >= min_sub_star_mass]
    df_subs[sub_columns].to_csv(f"FLAM_{simulation}_subhalo_catalog_snap{snap}.csv", index = False)

    # Calculate time taken to load and save data frames.
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logging.info(f"\nLoaded {len(df_halos)} halos and saved subhalo data frame with {len(df_subs)} subhalos\n")
    logging.info(f"\nTime elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\n")


    return df_halos

def calculate_halo_quantities(row, Tmin = -1, Tmax = 0.3, rad_mult = 1.01):
    '''Calculate halo properties using the data of particles around the halo
    '''

    # Define a global variables for the mask of particle data around the halo. This is an unpicklable variable
    # so it cannot be passed into a function that will be used in the parallelization and will have to be 
    # initialized for each worker for efficiency.
    global mask

    pos_x, pos_y, pos_z = sw.cosmo_array([row["sub_pos_comoving_x"], row["sub_pos_comoving_y"], row["sub_pos_comoving_z"]], units = u.Mpc, comoving = True) # position of the center of the halo
    R_500c = row["R_500c_comoving"] * u.Mpc
    stellar_half_mass_rad = row["sub_stellar_half_mass_rad_comoving"]

    # Load only particles that are around the halo to save memory and time. This is a very coarse filtering as it loads more particles. We us rad_mult to make sure we get all particles in R_500c.
    load_region = [[pos_x - rad_mult * R_500c, pos_x + rad_mult * R_500c],
                   [pos_y - rad_mult * R_500c, pos_y + rad_mult * R_500c],
                   [pos_z - rad_mult * R_500c, pos_z + rad_mult * R_500c]]
    mask.constrain_spatial(load_region) # We only collect the particles in the range around the target halo defined above to save on memory and time
    particle_data = sw.load(virtual_path, mask = mask) # load particle data in box around the halo
    AGN_delta_T_in_K = float(particle_data.metadata.parameters['EAGLEAGN:AGN_delta_T_K']) # temparture injection of the AGN, used for filtering recently heated AGN particles.

    gas_coords = particle_data.gas.coordinates.to_comoving().in_units(u.Mpc).value # gas particle coordinates. units: comoving Mpc
    gas_masses = particle_data.gas.masses.to_physical().in_units(u.Solar_Mass).value # gas particle masses
    gas_densities = particle_data.gas.densities.to_physical().in_units(u.Solar_Mass / (u.kpc)**3).value # gas particle densities
    gas_temperatures = particle_data.gas.temperatures.in_units(u.K).value # gas particle temperatures
    last_agn_feedback = particle_data.gas.last_agnfeedback_scale_factors.value # scale expansion factor denoting when the particle was last heated by AGN
    star_coords = particle_data.stars.coordinates.in_units(u.Mpc).value # star particle coordinates
    star_masses = particle_data.stars.masses.to_physical().in_units(u.Solar_Mass).value # star particle masses

    halo_pos = np.array([pos_x, pos_y, pos_z]) # position of center of halo. units: comoving Mpc
    gas_dist_from_halo = np.linalg.norm(gas_coords - halo_pos, axis = 1, ord = 2) # gas particle distances away from center of halo.
    star_dist_from_halo = np.linalg.norm(star_coords - halo_pos, axis = 1, ord = 2) # star particle distances away from center of halo.

    # Masks
    cold_gas_in_500c = np.where((gas_dist_from_halo <= R_500c.value) & (gas_temperatures <= cold_gas_temperature)) # mask for cold gas in R_500c
    # Mask for recently heated AGN gas particles inside R_500c as described in the SOAP catalogue.
    not_recently_heated_in_500c = np.where(
        (gas_dist_from_halo <= R_500c.value) &
        (
            (last_agn_feedback < a_limit) | 
            (gas_temperatures < (np.power(10., Tmin) * AGN_delta_T_in_K)) | 
            (gas_temperatures > (np.power(10., Tmax) * AGN_delta_T_in_K))
        )
    )
    # Mask for recently heated AGN gas particles inside inner shell
    not_recently_heated_in_inner = np.where(
        (gas_dist_from_halo >= (inner_shell_inner_rad * R_500c.value)) & (gas_dist_from_halo <= (inner_shell_outer_rad * R_500c.value)) &
        (
            (last_agn_feedback < a_limit) | 
            (gas_temperatures < (np.power(10., Tmin) * AGN_delta_T_in_K)) | 
            (gas_temperatures > (np.power(10., Tmax) * AGN_delta_T_in_K))
        )
    )
    # Mask for recently heated AGN gas particles inside middle shell
    not_recently_heated_in_mid = np.where(
        (gas_dist_from_halo >= (mid_shell_inner_rad * R_500c.value)) & (gas_dist_from_halo <= (mid_shell_outer_rad * R_500c.value)) &
        (
            (last_agn_feedback < a_limit) | 
            (gas_temperatures < (np.power(10., Tmin) * AGN_delta_T_in_K)) | 
            (gas_temperatures > (np.power(10., Tmax) * AGN_delta_T_in_K))
        )
    )
    # Mask for recently heated AGN gas particles inside outer shell
    not_recently_heated_in_outer = np.where(
        (gas_dist_from_halo >= (outer_shell_inner_rad * R_500c.value)) & (gas_dist_from_halo <= (outer_shell_outer_rad * R_500c.value)) &
        (
            (last_agn_feedback < a_limit) | 
            (gas_temperatures < (np.power(10., Tmin) * AGN_delta_T_in_K)) | 
            (gas_temperatures > (np.power(10., Tmax) * AGN_delta_T_in_K))
        )
    )
    # Mask for star particles within twice the stellar half mass radius.
    star_in_2stellar_half_mass_rad = np.where(star_dist_from_halo <= (2 * stellar_half_mass_rad)) # mask For star particles inside of twice the stellar half mass radius

    # Filtering particles
    gas_masses_in_500c = gas_masses[not_recently_heated_in_500c] # masses of gas particles that aren't recently heated by AGN
    cold_gas_masses_in_500c = gas_masses[cold_gas_in_500c]
    star_masses_in_2half_mass_rad = star_masses[star_in_2stellar_half_mass_rad] # masses of star particles inside of twice the stellar half mass radius
    # variables relevant for gas clumping factor
    gas_mass_in_inner_shell = gas_masses[not_recently_heated_in_inner] # masses of gas particles in inner shell
    gas_density_in_inner_shell = gas_densities[not_recently_heated_in_inner] # densities of gas particles in inner shell
    volumes_in_inner_shell = gas_mass_in_inner_shell/gas_density_in_inner_shell # volumes of gas particles in inner shell
    gas_mass_in_mid_shell = gas_masses[not_recently_heated_in_mid] # masses of gas particles in mid shell
    gas_density_in_mid_shell = gas_densities[not_recently_heated_in_mid] # densities of gas particles in mid shell
    volumes_in_mid_shell = gas_mass_in_mid_shell/gas_density_in_mid_shell # volumes of gas particles in mid shell
    gas_mass_in_outer_shell = gas_masses[not_recently_heated_in_outer] # masses of gas particles in outer shell
    gas_density_in_outer_shell = gas_densities[not_recently_heated_in_outer] # densities of gas particles in outer shell
    volumes_in_outer_shell = gas_mass_in_outer_shell/gas_density_in_outer_shell # volumes of gas particles in outer shell
    # Calculate aggregate quantities, avoid RuntimeWarning from the logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        gas_mass_in_500c = np.log10(np.sum(gas_masses_in_500c)) # total gas mass of particles that aren't recently heated by AGN in R_500c. This will be a check on code to compare with M_gas_500c.
        cold_gas_mass_in_500c = np.log10(np.sum(cold_gas_masses_in_500c))
        star_mass_2half_mass_rad = np.log10(np.sum(star_masses_in_2half_mass_rad)) # stellar mass in twice the stellar half mass radius
        clump_inner = 0.5 * np.log10(np.sum(volumes_in_inner_shell) * np.sum(gas_mass_in_inner_shell * gas_density_in_inner_shell)/(np.sum(gas_mass_in_inner_shell)**2)) # clumping factor of gas in inner shell
        clump_mid = 0.5 * np.log10(np.sum(volumes_in_mid_shell) * np.sum(gas_mass_in_mid_shell * gas_density_in_mid_shell)/(np.sum(gas_mass_in_mid_shell)**2)) # clumping factor of gas in mid shell
        clump_outer = 0.5 * np.log10(np.sum(volumes_in_outer_shell) * np.sum(gas_mass_in_outer_shell * gas_density_in_outer_shell)/(np.sum(gas_mass_in_outer_shell)**2)) # clumping factor of gas in outer shell
    

    return gas_mass_in_500c, cold_gas_mass_in_500c, star_mass_2half_mass_rad, clump_inner, clump_mid, clump_outer

def initialize_worker():
    '''Initialize mask unpicklable variable within each worker.
    '''
    
    global mask, mask_initialized
    if not mask_initialized:
        mask = sw.mask(virtual_path)
        mask_initialized = True

def worker(chunk, columns):
    '''Perform calculation of halo quantities within each CPU on a chunk of the halo data frame.
    '''

    # Intialize mask variable
    initialize_worker()

    gas_mass_in_500c_calc, cold_gas_mass_in_500c, star_mass_in_2half_mass_rad, C_inner, C_mid, C_outer = [], [], [], [], [], []

    chunk_df = pd.DataFrame(chunk, columns = columns) # a chunk of the halo data frame that is passed to the CPU
    for _, row in tqdm(chunk_df.iterrows(), total = len(chunk_df)):
        try:
            gas_mass, cold_gas_mass, star_mass, clump_inner, clump_mid, clump_outer = calculate_halo_quantities(row)
            gas_mass_in_500c_calc.append(gas_mass)
            cold_gas_mass_in_500c.append(cold_gas_mass)
            star_mass_in_2half_mass_rad.append(star_mass)
            C_inner.append(clump_inner)
            C_mid.append(clump_mid)
            C_outer.append(clump_outer)
        except Exception as e:
            logging.error(f"Error processing row {row['halo_cat_ID']}: {e}")
            gas_mass_in_500c_calc.append(np.nan)
            cold_gas_mass_in_500c.append(np.nan)
            star_mass_in_2half_mass_rad.append(np.nan)
            C_inner.append(np.nan)
            C_mid.append(np.nan)
            C_outer.append(np.nan)


    return gas_mass_in_500c_calc, cold_gas_mass_in_500c, star_mass_in_2half_mass_rad, C_inner, C_mid, C_outer

def main():
    '''Split halo data frame across CPUs to parallelize halo quanitity. Then gather all the data from each CPU and add it to the
       original halo data frame. Save the resulting data frame.
    '''
    

    df_halos = load_halo_subhalo_df() # load halo data frame
    chunked_data = np.array_split(df_halos.values, num_jobs) # split halo data frame to chunks in order to be passed to CPUs
    columns = df_halos.columns

    logging.info("\nStarting parallel processing")
    results = Parallel(n_jobs = num_jobs)(delayed(worker)(chunk, columns) for chunk in chunked_data) # parallelize halo calculations
    
    # Gather calculated halo properties from each CPU.
    gather_gas_mass, gather_cold_gas_mass, gather_star_mass, gather_C_inner, gather_C_mid, gather_C_outer = [], [], [], [], [], []
    for res in results:
        gather_gas_mass.extend(res[0])
        gather_cold_gas_mass.extend(res[1])
        gather_star_mass.extend(res[2])
        gather_C_inner.extend(res[3])
        gather_C_mid.extend(res[4])
        gather_C_outer.extend(res[5])
     # Add columns to halo data frame
    df_halos["M_gas_500c_calc"] = gather_gas_mass
    df_halos["M_cold_gas_500c"] = gather_cold_gas_mass
    df_halos["M_star_in_2stellar_half_mass_rad"] = gather_star_mass
    df_halos["gas_vw_clumping_inner"] = gather_C_inner
    df_halos["gas_vw_clumping_mid"] = gather_C_mid
    df_halos["gas_vw_clumping_outer"] = gather_C_outer

    # Save updated halo data frame
    output_file = f"FLAM_{simulation}_halo_catalog_snap{snap}.csv"
    df_halos.to_csv(output_file, index = False)


if __name__ == '__main__':
    # Start timer
    start = time.time()
    main()
    # Calculate and display total time elapsed
    end = time.time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    logging.info(f"\n\nDone! Total time elapsed {hours:02d}:{minutes:02d}:{seconds:02d}\n")
