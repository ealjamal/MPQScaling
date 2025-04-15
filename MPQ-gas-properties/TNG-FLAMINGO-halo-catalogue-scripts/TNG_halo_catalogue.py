# This a script to load halo (SO) data and subhalo data from the TNG simulations
# with support for parallelization.

import illustris_python as il
import numpy as np
import pandas as pd
import unyt as u
import h5py
from joblib import Parallel, delayed
from Xray_interpolator import XrayCalculator
from tqdm import tqdm
import time
import warnings
import sys
import logging

##### Initializations, file paths and helpful variables

simulation = "TNG-Cluster" # simulation
num_jobs = int(sys.argv[1]) # number of cpus
snap = int(sys.argv[2]) # snapshot for which to load data
min_sub_star_mass = float(sys.argv[3]) # minimum subhalo stellar mass
snap_str = f"{snap:03d}"
# Path for halos and particles particles
sim_data_path = f"/virgotng/universe/IllustrisTNG/{simulation}/output" if simulation != "TNG-Cluster" else f"/virgotng/mpia/TNG-Cluster/L680n8192TNG/output"
# Path for the snapshot, used only to load header data
snap_file = f"/virgotng/universe/IllustrisTNG/{simulation}/output/snapdir_{snap_str}/snap_{snap_str}.0.hdf5" if simulation != "TNG-Cluster" else f"/virgotng/mpia/TNG-Cluster/L680n8192TNG/output/snapdir_{snap_str}/snap_{snap_str}.0.hdf5"
xray_table_path = f"./X_Ray_table_metals_full_withconvolved.hdf5" # Xray emissivity table that was used in the FLAMINGO simulations.
# Extract cosmological parameter information and redshift
with h5py.File(snap_file, "r") as f:
    redshift = f["Header"].attrs["Redshift"]
    scale_factor = 1/(1 + redshift)
    little_h = f["Header"].attrs["HubbleParam"]
# Initialize xray calculator to calculate observer frame ROSAT band luminosities.
xray_calc = XrayCalculator(np.array([round(redshift, 1), round(redshift, 1)]), xray_table_path, ['ROSAT'], ['energies_intrinsic'])
# Initialize logging configurations
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

##### Physical Constants

K_B = 1.380649*10**(-16) # Boltzmann constant. units: CGS (erg/K)
K_B_in_keV = 8.6173303 * 10**(-8) # Boltzmann constant. units: keV
gamma = 5/3 # Thermodynamic adiabatic index
X_H = 0.76 # Hydrogen fraction
mass_of_proton = 1.67262192369e-24 # Mass of proton. units: grams

##### Analysis-specific parameters

# These can be changed in order to produce data of interest for a project.
# For example, if one is interested in analyzing the stellar mass of the central 
# galaxy within 50kpc of the center, one can change `star_BCG_dist_cut2` to 0.05.
min_M500c  = 12.5 # minimum halo mass in R_500c to have in data. units: M_sun
max_M500c = float("inf") # maximum halo mass in R_500c to have in data. units: M_sun
star_BCG_dist_cut1 = .1 # the aperture to consider stellar mass of BCG i.e. the central galaxy. units: Mpc
star_BCG_dist_cut2 = .03 # another aperture to consider stellar mass of BCG i.e. the central galaxy. units: Mpc
hot_gas_temp = 1e5 # lower bound on temperature for gas to be considered hot. units: Kelvin.
cold_gas_temp = 1e4 # upper bound on temperature for gas to be considered cold. units: Kelvin.
spect_hot_gas_temp = 0.1/K_B_in_keV # temperature threshold for calculating spectroscopic-like temperature.
core_outer_radius = 0.15 # in R500c units
# We define three shells to calculate the gas clumping factor in each cell, inner shell, mid shell and outer shell.
inner_shell_inner_rad  = 0.15 # inner radius of inner shell in R_500c units
inner_shell_outer_rad = 0.35 # outer radius of inner shell in R_500c units
mid_shell_inner_rad = 0.5 # inner radius of mid shell in R_500c units
mid_shell_outer_rad = 0.7 # outer radius of mid shell in R_500c units
outer_shell_inner_rad = 0.8 # inner radius of outer shell in R_500c units
outer_shell_outer_rad = 1 # outer radius of outer shell in R_500c units

##### Data Fields

# Other quantities can be added to these fields. With the change of this global
# variables, the loading function have to be updated to include these other
# quantities in the loaded data frame.
halo_fields = ["GroupFirstSub", "GroupPos", "GroupVel",
               "Group_M_Crit500", "Group_R_Crit500",
               "Group_M_Crit200", "Group_R_Crit200"] if simulation != "TNG-Cluster" else ["GroupPrimaryZoomTarget", "GroupFirstSub", "GroupPos", "GroupVel",
                                                                                          "Group_M_Crit500", "Group_R_Crit500",
                                                                                          "Group_M_Crit200", "Group_R_Crit200"] # TNG-cluster is the only one that has GroupPrimaryZoomTarget         
subhalo_fields = ["SubhaloGrNr", "SubhaloMassType",
                  "SubhaloPos", "SubhaloVel", "SubhaloHalfmassRadType"]
gas_fields = ["Coordinates", "Density", "ElectronAbundance",
              "InternalEnergy", "Masses", "GFM_Metals", "StarFormationRate"] # fields for gas particles
star_fields = ["Coordinates", "Masses"] # fields for gas particles
bh_fields = ["Coordinates", "Masses"] # fields for bh (Black Hole) particles

##### Data loading functions.

def load_halo_df():
    '''Load data frame for halos.
    '''

    # Time loading data frame.
    start_time = time.time()
    # Load the halo catalog from the simulation with specific data fields
    # provided in the global variable `halo_fields`.
    logging.info(f"\n\nLoading halo data for simulation {simulation} snapshot {snap} and redshift = {redshift:0.2f}\n")
    halos = il.groupcat.loadHalos(basePath = sim_data_path, snapNum = snap, fields = halo_fields)

    # Create separate data arrays for each data field, change units, use log scale.
    cent_galaxy_ID = halos['GroupFirstSub'] # ID for the central galaxy.
    halo_pos = halos['GroupPos'] * scale_factor/(1000 * little_h) # position vector of the group/halo, ckpc/h -> Mpc
    halo_vel = halos['GroupVel'] * 1/scale_factor # velocity vector of the group/halo, km/s/a -> km/s
    halo_R500 = halos['Group_R_Crit500'] * scale_factor/(1000 * little_h) # R500c of the group, ckpc/h -> Mpc
    halo_R200 = halos['Group_R_Crit200'] * scale_factor/(1000 * little_h) # R200c of the group, ckpc/h -> Mpc
    # Check for RuntimeWarning that result from taking the log of zero.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        halo_M500 = np.log10(halos['Group_M_Crit500'] * 1e10/little_h) # 1e10Msun/h -> Msun
        halo_M200 = np.log10(halos['Group_M_Crit200'] * 1e10/little_h) # 1e10Msun/h -> Msun

    # Split different dimensions of halo position and velocity.
    (halo_pos_x, halo_pos_y, halo_pos_z)  = (halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2])
    (halo_vel_x, halo_vel_y, halo_vel_z) = (halo_vel[:, 0], halo_vel[:, 1], halo_vel[:, 2])

    # Create halo DataFrame using the above halo data extracted from the sim.
    halo_keys = ["M_200c", "R_200c", "M_500c", "R_500c", "halo_pos_physical_x", "halo_pos_physical_y", "halo_pos_physical_z",
                 "halo_vel_x", "halo_vel_y", "halo_vel_z", "BCG_ID"]
    halo_vals = [halo_M200, halo_R200, halo_M500, halo_R500, halo_pos_x, halo_pos_y, halo_pos_z,
                 halo_vel_x, halo_vel_y, halo_vel_z, cent_galaxy_ID]
    
    # For TNG-Cluster we add the column to denote which group was the target of the zoom.
    if simulation == "TNG-Cluster":
        zoom_target = halos['GroupPrimaryZoomTarget']
        halo_keys = ["zoom_target"] + halo_keys
        halo_vals.insert(0, zoom_target)

    # Create halo data frame.
    df_halos = pd.DataFrame(dict(zip(halo_keys, halo_vals)))
    # Add a column to the halo DataFrame for the integer halo IDs which
    # corresponds to the index of the row that each halo occupies.
    df_halos.reset_index(inplace = True)
    df_halos.rename(columns = {"index": "halo_ID"}, inplace = True) # the index in the original halo catalog is the halo ID
    df_halos = df_halos[(df_halos["M_500c"] >= min_M500c) & (df_halos["M_500c"] <= max_M500c)] # filter halos to include only halos with masses in range of interest
    if simulation == "TNG-Cluster":
        df_halos = df_halos[df_halos["zoom_target"] == 1] # only include zoom targets in TNG-Cluster

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logging.info(f"\nLoaded {len(df_halos)} halos\n")
    logging.info(f"\nTime elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\n")


    return df_halos

def load_sub_df():
    '''Load data frame for subhalos.
    '''

    # Time loading data frame.
    start_time = time.time()
    logging.info(f"\n\nLoading subhalo data for simulation {simulation} snapshot {snap} and redshift = {redshift:0.2f}\n")
    # Load the subhalo catalog from the simulation with specific data fields
    # provided in the global variable `subhalo fields`.
    subhalos = il.groupcat.loadSubhalos(basePath = sim_data_path, snapNum = snap, fields = subhalo_fields)

    # Create separate data arrays for subhalos and also change units.
    host_ID = subhalos['SubhaloGrNr'] # the halo_ID of the host of this subhalo
    sub_pos = subhalos['SubhaloPos'] * scale_factor/(1000 * little_h) # ckpc/h -> Mpc
    sub_com_vel = subhalos['SubhaloVel'] # units: km/s
    # Subhalo stellar half mass radius.
    sub_stellar_half_mass_rad = subhalos["SubhaloHalfmassRadType"][:, 4]  * scale_factor/(1000 * little_h) # ckpc/h -> Mpc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        sub_stellar_mass = np.log10(subhalos['SubhaloMassType'][:, 4] * 1e10/little_h) # 10^10M_sun/h -> M_sun

    # Split different dimensions of subhalo position.
    (sub_pos_x, sub_pos_y, sub_pos_z) = (sub_pos[:, 0], sub_pos[:, 1], sub_pos[:, 2])
    (sub_com_vel_x, sub_com_vel_y, sub_com_vel_z) = (sub_com_vel[:, 0], sub_com_vel[:, 1], sub_com_vel[:, 2])

    # Create subhalo DataFrame using the above subhalo data extracted from the simulation.
    sub_keys = ["host_ID", "sub_pos_x", "sub_pos_y", "sub_pos_z",
                "sub_com_vel_x", "sub_com_vel_y", "sub_com_vel_z",
                "sub_M_star", "sub_stellar_half_mass_rad_physical"]
    sub_vals = [host_ID, sub_pos_x, sub_pos_y, sub_pos_z, 
                sub_com_vel_x, sub_com_vel_y, sub_com_vel_z,
                sub_stellar_mass, sub_stellar_half_mass_rad]
    df_subs = pd.DataFrame(dict(zip(sub_keys, sub_vals)))
     # Add a column to the subhalo DataFrame for the in integer subhalo IDs.
    df_subs.reset_index(inplace = True)
    df_subs.rename(columns = {'index': 'sub_ID'}, inplace = True)
    df_subs = df_subs[df_subs["sub_M_star"] >= min_sub_star_mass] # Filter subhalos based on bound stellar mass in subhalo.

    # Calculate and display total elapsed time.
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logging.info(f"\nLoaded {len(df_subs)} subhalos\n")
    logging.info(f"\nTime elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\n")


    return df_subs

def load_halo_gas_data(halo_ID):
    '''Load gas particle data for the halo with ID given by halo_ID
    '''

    # Load gas particles for the halo.
    gas_parts = il.snapshot.loadHalo(basePath = sim_data_path, snapNum = snap, id = int(halo_ID), 
                                     partType = 'gas', fields = gas_fields)
    
    gas_count = gas_parts["count"] # Number of gas particles in this halo (not just bound particles)
    if gas_count > 0:
        gas_coords = gas_parts["Coordinates"] * scale_factor/(1000 * little_h) # coordinates of gas particles. units: ckpc/h -> Mpc
        gas_masses = gas_parts["Masses"] * 1e10/little_h # gas mass of each particle, units: 1e10Msun/h -> Msun
        gas_density = gas_parts["Density"]/(scale_factor/little_h) ** 3 * (1e10/little_h) # gas density of each particle/cell, units: 1e10Msun/h / (ckpc/h)^3 -> Msun/kpc^3
        gas_electron_abundance = gas_parts["ElectronAbundance"] # electron abundance for each gas cell i.e. fractional electron number density with respect to total H number density
        gas_energy = gas_parts["InternalEnergy"] # Internal Energy per unit mass, units: (km/s)^2
        gas_metals = gas_parts["GFM_Metals"] # metal fraction in gas of nine species (last entry is total) shape: (1, N, 10). NOT in solar units.
        gas_SFR = gas_parts["StarFormationRate"] # Star formation rate
        gas_mean_molecular_weight = 4./(1. + 3. * X_H + 4. * X_H * gas_electron_abundance) * mass_of_proton # gas molecular weight, used to calculate temperature.
        gas_temp = (gamma - 1) * gas_energy/K_B * 10**10 * gas_mean_molecular_weight # gas temperature, units: K.
    else:
        # if there are no particles, fill the particle data with None.
        gas_coords = None
        gas_masses = None
        gas_density = None
        gas_electron_abundance = None
        gas_energy = None
        gas_metals = None
        gas_mean_molecular_weight = None
        gas_temp = None

    # Result dictonary for gas particles of this halo.
    gas_props = {"Count": gas_count, "Coordinates": gas_coords, "Masses": gas_masses,
                 "Density": gas_density, "Temperature": gas_temp,
                 "Metals": gas_metals, "SFR": gas_SFR}
    
    
    return gas_props

def load_halo_star_data(halo_ID):
    '''Load star particle data for the halo with halo_ID
    '''
    
    # Load star particles for the halo.
    star_parts = il.snapshot.loadHalo(basePath = sim_data_path, snapNum = snap, id = int(halo_ID), 
                                      partType = 'star', fields = star_fields)
    
    star_count = star_parts["count"] # Number of star particles in this halo (not just bound particles)
    if star_count > 0:
        star_coords = star_parts["Coordinates"] * scale_factor/(1000 * little_h) # coordinates of star particles. units: ckpc/h -> Mpc
        star_masses = star_parts["Masses"] * 1e10/little_h # star mass of each particle, units: 1e10Msun/h -> Msun
    else:
        # if there are not particles, fill the particle data with None.
        star_coords = None
        star_masses = None

    # Result dictonary for stellar particles of this halo.
    star_props = {"Count": star_count, "Coordinates": star_coords, "Masses": star_masses}


    return star_props

def load_halo_bh_data(halo_ID):
    '''Load bh particle data for the halo with halo_ID
    '''

    # Load bh particles for the halo.
    bh_parts = il.snapshot.loadHalo(basePath = sim_data_path, snapNum = snap, id = int(halo_ID), 
                                    partType = 'bh', fields = bh_fields)
    
    bh_count = bh_parts["count"] # Number of bh particles in this halo (not just bound particles)
    if bh_count > 0:
        bh_coords = bh_parts["Coordinates"] * scale_factor/(1000 * little_h) # coordinates of bh particles. units: ckpc/h -> Mpc
        bh_masses = bh_parts["Masses"] * 1e10/little_h # bh mass of each particle, units: 1e10Msun/h -> Msun
    else:
        # if there are not particles, fill the particle data with None.
        bh_coords = None
        bh_masses = None

    # Result dictonary for bh particles of this halo.
    bh_props = {"Count": bh_count, "Coordinates": bh_coords, "Masses": bh_masses}


    return bh_props

def load_sub_star_data(sub_ID):
    '''Load star particle data for the subhalo with sub_ID
    '''

    # Load star particles for the subhalo.
    star_parts = il.snapshot.loadSubhalo(basePath = sim_data_path, snapNum = snap, id = int(sub_ID), 
                                      partType = 'star', fields = star_fields)
    
    star_count = star_parts["count"] # Number of star particles in this subhalo (not just bound particles)
    if star_count > 0:
        star_coords = star_parts["Coordinates"] * scale_factor/(1000 * little_h) # coordinates of star particles. units: ckpc/h -> Mpc
        star_masses = star_parts["Masses"] * 1e10/little_h # gas mass of each particle, units: 1e10Msun/h -> Msun
    else:
         # if there are not particles, fill the particle data with None.
        star_coords = None
        star_masses = None

    # Result dictonary for stellar particles of this subhalo.
    star_props = {"Count": star_count, "Coordinates": star_coords, "Masses": star_masses}


    return star_props

def calculate_luminosity(densities, temperatures, metals, masses, redshifts):
    '''Calculate total observer frame ROSAT band [0.5-2.0keV] luminosity for a single halo based on the emissivity tables in FLAMINGO.
    '''    

    densities = densities * scale_factor**3 * (u.Solar_Mass/u.kpc**3) # attach Msun/kpc^3 units to densities and convert to comoving frame.
    temperatures = temperatures * u.K # attach kelvin units to temperature
    # metals comes loaded as (1, N, num_metals) array, we make it (N, num_metals)
    metals = metals[0, :, :] * u.dimensionless 
    masses = masses * u.Solar_Mass # attach Msun units to mass
    redshifts = redshifts * u.dimensionless 

    idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n = xray_calc.find_indices(densities, temperatures, metals[:, :9], masses, redshifts, fill_value = 0)
    # calculate luminosity that results from each gas particle.
    xray_luminosity = xray_calc.interpolate_X_Ray(idx_z, idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n, bands = ['ROSAT'], observing_types = ['energies_intrinsic'], fill_value = 0)
    # calculate total luminsity for all particles in this halo.
    xray_luminosity_tot = np.sum(xray_luminosity.value.flatten())


    return xray_luminosity_tot

def calculate_halo_quantities(row):
    '''Calculate aggregate particle properties for a each row in original halo data frame loaded using load_halo_df().
    '''

    halo_ID = row["halo_ID"]
    halo_coords = row["halo_pos_physical_x"], row["halo_pos_physical_y"], row["halo_pos_physical_z"]
    R_500c = row["R_500c"]

    gas_props = load_halo_gas_data(halo_ID) # load gas particle data
    gas_count = gas_props["Count"]
    gas_coords = gas_props["Coordinates"]
    gas_masses = gas_props["Masses"]
    gas_densities = gas_props["Density"]
    gas_temps = gas_props["Temperature"]
    gas_metals = gas_props["Metals"]
    gas_SFR = gas_props["SFR"]

    star_props = load_halo_star_data(halo_ID) # load star particle data
    star_count = star_props["Count"]
    star_coords = star_props["Coordinates"]
    star_masses = star_props["Masses"]

    bh_props = load_halo_bh_data(halo_ID) # load bh particle data
    bh_count = bh_props["Count"]
    bh_coords = bh_props["Coordinates"]
    bh_masses = bh_props["Masses"]

    # If there are no gas particles or no stellar particles in this halo, aggregate properties will all be -inf and this halo will be excluded from subsequent analyses.
    if gas_count <= 0 or star_count <= 0:
        return float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")
    # Calculate the distance of each gas particle, star, bh particle from the center of halo (location of most bound particle)
    gas_dist_from_halo = np.linalg.norm(halo_coords - gas_coords, axis = 1, ord = 2)
    star_dist_from_halo = np.linalg.norm(halo_coords - star_coords, axis = 1, ord = 2)
    bh_dist_from_halo = np.linalg.norm(halo_coords - bh_coords, axis = 1, ord = 2)

    # Masks
    gas_in_500c = np.where(gas_dist_from_halo <= R_500c) # mask for gas particles in R_500c
    ce_gas_in_500c = np.where((gas_dist_from_halo >= (core_outer_radius * R_500c)) &
                                  (gas_dist_from_halo <= R_500c)
                                  ) # mask gas for particles in R_500c and core excised i.e. 0.15R_500c <= gas_dist_from_halo <= R_500c
    star_in_500c = np.where(star_dist_from_halo <= R_500c) # mask for star particles in R_500c
    bh_in_500c = np.where(bh_dist_from_halo <= R_500c) # mask for bh particles in R_500c
    hot_gas_in_500c = np.where((gas_dist_from_halo <= R_500c) &
                               (gas_temps >= hot_gas_temp)) # mask for hot gas particles in R_500c
    cold_gas_in_500c = np.where((gas_dist_from_halo <= R_500c) &
                                (gas_temps <= cold_gas_temp)) # mask for cold gas particles in R_500c
    ce_hot_gas_in_500c = np.where((gas_dist_from_halo >= (core_outer_radius * R_500c)) &
                                  (gas_dist_from_halo <= R_500c) 
                                  & (gas_temps > spect_hot_gas_temp)) # mask for hot gas particles in 0.15R_500c <= gas_dist_from_halot <= R_500c
    # masks relevant for gas clumping factor
    gas_in_inner_shell = np.where((gas_dist_from_halo >= (inner_shell_inner_rad * R_500c)) & (gas_dist_from_halo <= (inner_shell_outer_rad * R_500c)))
    gas_in_mid_shell = np.where((gas_dist_from_halo >= (mid_shell_inner_rad * R_500c)) & (gas_dist_from_halo <= (mid_shell_outer_rad * R_500c)))
    gas_in_outer_shell = np.where((gas_dist_from_halo >= (outer_shell_inner_rad * R_500c)) & (gas_dist_from_halo <= (outer_shell_outer_rad * R_500c)))

    # Particle filtering
    gas_mass_in_500c = gas_masses[gas_in_500c] # masses of gas particles in R_500c
    ce_gas_mass_in_500c = gas_masses[ce_gas_in_500c] # masses of gas particles in >= 0.15R_500c and <= R_500c
    star_mass_in_500c = star_masses[star_in_500c] # masses of star particles in R_500c 
    bh_mass_in_500c = bh_masses[bh_in_500c] # masses of bh particles in R_500c 
    hot_gas_mass_in_500c = gas_masses[hot_gas_in_500c] # masses of hot gas particles in R_500c
    cold_gas_mass_in_500c = gas_masses[cold_gas_in_500c] # masses of cold gas particles in R_500c
    gas_density_in_500c = gas_densities[gas_in_500c] # densities of gas particles in R_500c
    ce_gas_density_in_500c = gas_densities[ce_gas_in_500c] # densities of gas particles in >= 0.15R_500c and <= R_500c
    gas_temp_in_500c = gas_temps[gas_in_500c] # temperatures of gas particles in R_500c
    hot_gas_temp_in_500c = gas_temps[hot_gas_in_500c] # temperatures of hot gas particles in R_500c
    ce_gas_temp_in_500c = gas_temps[ce_gas_in_500c] # temperaturse of hot gas particles in >= 0.15R_500c and <= R_500c
    spect_like_weight_hot_gas_500c = gas_densities[ce_hot_gas_in_500c] * gas_masses[ce_hot_gas_in_500c] * gas_temps[ce_hot_gas_in_500c]**(-0.75) # weights to calculate sl temperature in weighted average
    gas_SFR_in_500c = gas_SFR[gas_in_500c] # SFR of gas particles in R_500c
    # variables relevant for gas clumping factor
    gas_mass_in_inner_shell = gas_masses[gas_in_inner_shell] # masses of gas particles in inner shell
    gas_density_in_inner_shell = gas_densities[gas_in_inner_shell] # densities of gas particles in inner shell
    volumes_in_inner_shell = gas_mass_in_inner_shell/gas_density_in_inner_shell # volumes of gas particles in inner shell
    gas_mass_in_mid_shell = gas_masses[gas_in_mid_shell] # masses of gas particles in mid shell
    gas_density_in_mid_shell = gas_densities[gas_in_mid_shell] # densities of gas particles in mid shell
    volumes_in_mid_shell = gas_mass_in_mid_shell/gas_density_in_mid_shell # volumes of gas particles in mid shell
    gas_mass_in_outer_shell = gas_masses[gas_in_outer_shell] # masses of gas particles in outer shell
    gas_density_in_outer_shell = gas_densities[gas_in_outer_shell] # densities of gas particles in outer shell
    volumes_in_outer_shell = gas_mass_in_outer_shell/gas_density_in_outer_shell # volumes of gas particles in outer shell
    # Calculate aggregate quantities (supress RuntimeWarning from log)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        clump_inner = 0.5 * np.log10(np.sum(volumes_in_inner_shell) * np.sum(gas_mass_in_inner_shell * gas_density_in_inner_shell)/(np.sum(gas_mass_in_inner_shell)**2)) # clumping factor of gas in inner shell
        clump_mid = 0.5 * np.log10(np.sum(volumes_in_mid_shell) * np.sum(gas_mass_in_mid_shell * gas_density_in_mid_shell)/(np.sum(gas_mass_in_mid_shell)**2)) # clumping factor of gas in mid shell
        clump_outer = 0.5 * np.log10(np.sum(volumes_in_outer_shell) * np.sum(gas_mass_in_outer_shell * gas_density_in_outer_shell)/(np.sum(gas_mass_in_outer_shell)**2)) # clumping factor of gas in outer shell
        M_gas_500c = np.log10(np.sum(gas_mass_in_500c)) # total gas mass in R_500c
        M_star_500c = np.log10(np.sum(star_mass_in_500c)) # total star mass in R_500c
        M_bh_500c = np.log10(np.sum(bh_mass_in_500c)) if bh_count > 0 else float("-inf")
        M_hot_gas_500c =  np.log10(np.sum(hot_gas_mass_in_500c)) # total hot gas mass in R_500c
        M_cold_gas_500c = np.log10(np.sum(cold_gas_mass_in_500c)) # total cold gas mass in R_500c
        L_X_ROSAT_obs = np.log10(calculate_luminosity(gas_density_in_500c, gas_temp_in_500c, gas_metals[gas_in_500c, :], 
                                                      gas_mass_in_500c, np.zeros_like(gas_mass_in_500c) + round(redshift, 1))) # total x-ray luminosity in R_500c
        L_X_ROSAT_obs_ce = np.log10(calculate_luminosity(ce_gas_density_in_500c, ce_gas_temp_in_500c, gas_metals[ce_gas_in_500c, :], 
                                                         ce_gas_mass_in_500c, np.zeros_like(ce_gas_mass_in_500c) + round(redshift, 1))) # total x-ray ce luminosity in R_500c
        SFR_500c = np.log10(np.sum(gas_SFR_in_500c)) # total SFR in R_500c
    
    try:
        T_mw_500c = np.log10(np.average(gas_temp_in_500c, weights = gas_mass_in_500c)) # mass-weighted temperature of gas in R_500c
    except ZeroDivisionError: # ZeroDivision can happen if there are not particles so that the weights are zero.
        T_mw_500c = float("-inf")

    try:
        T_mw_hot_gas_500c = np.log10(np.average(hot_gas_temp_in_500c, weights = hot_gas_mass_in_500c)) # mass-weighted temperature of hot gas in R_500c
    except ZeroDivisionError:
        T_mw_hot_gas_500c = float("-inf")

    try:
        T_sl_500c = np.log10(np.average(gas_temps[ce_hot_gas_in_500c],
                                                            weights = spect_like_weight_hot_gas_500c)) # spectroscopic-like temperature in R_500c
    except ZeroDivisionError:
        T_sl_500c = float("-inf")


    return M_gas_500c, M_hot_gas_500c, M_cold_gas_500c, M_star_500c, M_bh_500c, T_mw_500c, T_mw_hot_gas_500c, T_sl_500c, L_X_ROSAT_obs, L_X_ROSAT_obs_ce, SFR_500c, clump_inner, clump_mid, clump_outer

def calculate_subhalo_quantities(row):
    '''Calculate aggregate particle properties for a each row in original subhalo data frame loaded using load_sub_df().
    '''

    sub_ID = row["sub_ID"]
    sub_coords = row["sub_pos_x"], row["sub_pos_y"], row["sub_pos_z"]
    stellar_half_mass_rad = row["sub_stellar_half_mass_rad_physical"]

    star_props = load_sub_star_data(sub_ID) # load subhalo star particles
    star_count = star_props["Count"]
    star_coords = star_props["Coordinates"]
    star_masses = star_props["Masses"]

    # If there are no stellar particles in this halo, aggregate properties will all be -inf and this subhalo will be excluded from subsequent analyses.
    if star_count <= 0:
        return float("-inf"), float("-inf"), float("-inf")

    # Calculate the distance of each gas particle and star particle from the center of subhalo (location of most bound particle)
    star_dist_from_sub = np.linalg.norm(sub_coords - star_coords, axis = 1, ord = 2)
    # Masks
    star_in_BCG_cut1 = np.where(star_dist_from_sub <= star_BCG_dist_cut1) # mask for first aperture
    star_in_BCG_cut2 = np.where(star_dist_from_sub <= star_BCG_dist_cut2) # mask for second aperture
    star_in_2stellar_half_mass_rad = np.where(star_dist_from_sub <= (2 * stellar_half_mass_rad)) # mask for stellar particles within 2 * stellar half mass radius

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        M_star_BCG1 = np.log10(np.sum(star_masses[star_in_BCG_cut1])) # total stellar mass of subhalo within first aperture
        M_star_BCG2 = np.log10(np.sum(star_masses[star_in_BCG_cut2])) # total stellar mass of subhalo within first aperture
        M_star_in_2half_mass_rad =  np.log10(np.sum(star_masses[star_in_2stellar_half_mass_rad])) # total stellar mass within 2 * stellar half mass radius


    return M_star_BCG1, M_star_BCG2, M_star_in_2half_mass_rad
    
def halo_worker(chunk, columns):
    '''Perform calculation of halo quantities within each CPU on a chunk of the halo data frame.
    '''

    # Define empty arrays for each halo quantities that will be populated for each worker and then joined in the end. 
    gas_mass_500c, hot_gas_mass_500c, cold_gas_mass_500c, star_mass_500c, bh_mass_500c, T_mw, T_mw_hot_gas, T_sl, xray_luminosity_500c, xray_luminosity_ce_500c, star_formation_rate_500c, C_inner, C_mid, C_outer = [], [], [], [], [], [], [], [], [], [], [], [], [], []

    chunk_df = pd.DataFrame(chunk, columns = columns) # a chunk of the halo data frame that is passed to the CPU
    for _, row in tqdm(chunk_df.iterrows(), total = len(chunk_df)):
        try:
            (M_gas_500c, M_hot_gas_500c, M_cold_gas_500c, M_star_500c, M_bh_500c, T_mw_500c, T_mw_hot_gas_500c,
             T_sl_500c, L_X_ROSAT_obs, L_X_ROSAT_obs_ce, SFR_500c, clump_inner, clump_mid, clump_outer) = calculate_halo_quantities(row)
            gas_mass_500c.append(M_gas_500c)
            hot_gas_mass_500c.append(M_hot_gas_500c)
            cold_gas_mass_500c.append(M_cold_gas_500c)
            star_mass_500c.append(M_star_500c)
            bh_mass_500c.append(M_bh_500c)
            T_mw.append(T_mw_500c)
            T_mw_hot_gas.append(T_mw_hot_gas_500c)  
            T_sl.append(T_sl_500c)
            C_inner.append(clump_inner)
            C_mid.append(clump_mid)
            C_outer.append(clump_outer)
            xray_luminosity_500c.append(L_X_ROSAT_obs)
            xray_luminosity_ce_500c.append(L_X_ROSAT_obs_ce)
            star_formation_rate_500c.append(SFR_500c)
        except Exception as e:
            logging.error(f"Error processing row " + str(row["halo_ID"]) + " : " + f"{e}") # defined f-strng as such because using a single string throws an error on vera cluster.
            gas_mass_500c.append(np.nan)
            hot_gas_mass_500c.append(np.nan)
            cold_gas_mass_500c.append(np.nan)
            star_mass_500c.append(np.nan)
            bh_mass_500c.append(np.nan)
            T_mw.append(np.nan)
            T_mw_hot_gas.append(np.nan)  
            T_sl.append(np.nan)
            C_inner.append(np.nan)
            C_mid.append(np.nan)
            C_outer.append(np.nan)
            xray_luminosity_500c.append(np.nan)
            xray_luminosity_ce_500c.append(np.nan)
            star_formation_rate_500c.append(np.nan)


    return gas_mass_500c, hot_gas_mass_500c, cold_gas_mass_500c, star_mass_500c, bh_mass_500c, T_mw, T_mw_hot_gas, T_sl, xray_luminosity_500c, xray_luminosity_ce_500c, star_formation_rate_500c, C_inner, C_mid, C_outer

def subhalo_worker(chunk, columns):
    '''Perform calculation of subhalo quantities within each CPU on a chunk of the subhalo data frame.
    '''

    # Define empty arrays for each subhalo quantities that will be populated for each worker and then joined in the end. 
    star_BCG1, star_BCG2, star_in_2half_mass_rad = [], [], []

    chunk_df = pd.DataFrame(chunk, columns = columns) # a chunk of the subhalo data frame that is passed to the CPU
    for _, row in tqdm(chunk_df.iterrows(), total = len(chunk_df)):
        try:
            M_star_BCG1, M_star_BCG2, M_star_in_2half_mass_rad = calculate_subhalo_quantities(row)
            star_BCG1.append(M_star_BCG1)
            star_BCG2.append(M_star_BCG2)
            star_in_2half_mass_rad.append(M_star_in_2half_mass_rad)
        except Exception as e:
            logging.error(f"Error processing row " + str(row["sub_ID"]) + " : " + f"{e}") # defined f-strng as such because using a single string throws an error on vera cluster.
            star_BCG1.append(np.nan)
            star_BCG2.append(np.nan)
            star_in_2half_mass_rad.append(np.nan)

    
    return star_BCG1, star_BCG2, star_in_2half_mass_rad

def main():
    '''Split halo data frame and subhalo data frame across CPUs to parallelize halo quanitity and subhalo quantity calculations
       Then gather all the data from each CPU and add it to the original halo and subhalo data frames. Save the resulting data frames.
    '''
    
    df_halos = load_halo_df() # load halo data frame
    halo_chunked_data = np.array_split(df_halos.values, num_jobs) # split halo data frame to chunks in order to be passed to CPUs
    halo_columns = df_halos.columns

    logging.info("\nStarting halo parallel processing")
    halo_results = Parallel(n_jobs = num_jobs)(delayed(halo_worker)(chunk, halo_columns) for chunk in halo_chunked_data) # parallelize halo calculations

    # Gather calculated halo properties from each CPU.
    gather_gas_mass, gather_hot_gas_mass, gather_cold_gas_mass, gather_star_mass, gather_bh_mass, gather_T_mw, gather_T_mw_hot_gas, gather_T_sl, gather_L_X, gather_L_X_ce, gather_SFR, gather_C_inner, gather_C_mid, gather_C_outer  = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for res in halo_results:
        gather_gas_mass.extend(res[0])
        gather_hot_gas_mass.extend(res[1])
        gather_cold_gas_mass.extend(res[2])
        gather_star_mass.extend(res[3])
        gather_bh_mass.extend(res[4])
        gather_T_mw.extend(res[5])
        gather_T_mw_hot_gas.extend(res[6])
        gather_T_sl.extend(res[7])
        gather_L_X.extend(res[8])
        gather_L_X_ce.extend(res[9])
        gather_SFR.extend(res[10])
        gather_C_inner.extend(res[11])
        gather_C_mid.extend(res[12])
        gather_C_outer.extend(res[13])

    # Add columns to halo data frame
    df_halos["M_gas_500c"] = gather_gas_mass
    df_halos["M_hot_gas_500c"] = gather_hot_gas_mass
    df_halos["M_cold_gas_500c"] = gather_cold_gas_mass
    df_halos["M_star_500c"] = gather_star_mass
    df_halos["M_bh_500c"] = gather_bh_mass
    df_halos["T_mw_500c"] = gather_T_mw
    df_halos["T_mw_hot_gas_500c"] = gather_T_mw_hot_gas
    df_halos["T_sl_500c"] = gather_T_sl
    df_halos["L_X_ROSAT_obs_500c"] = gather_L_X
    df_halos["L_X_ROSAT_obs_ce_500c"] = gather_L_X_ce
    df_halos["SFR_500c"] = gather_SFR
    df_halos["gas_vw_clumping_inner"] = gather_C_inner
    df_halos["gas_vw_clumping_mid"] = gather_C_mid
    df_halos["gas_vw_clumping_outer"] = gather_C_outer

    # Save updated halo data frame
    halo_output_file = simulation.replace(r"-", r"_") + f"_halo_catalog_snap{snap:02d}.csv"
    df_halos.to_csv(halo_output_file, index = False)

    df_subs = load_sub_df() # load subhalo data frame
    sub_chunked_data = np.array_split(df_subs.values, num_jobs) # split halo data frame to chunks in order to be passed to CPUs
    sub_columns = df_subs.columns

    logging.info("\nStarting subhalo parallel processing")
    sub_results = Parallel(n_jobs = num_jobs)(delayed(subhalo_worker)(chunk, sub_columns) for chunk in sub_chunked_data) # parallelize subhalo calculations

    # Gather calculated subhalo properties from each CPU.
    gather_star_BCG1, gather_star_BCG2, gather_star_in_2half_mass_rad = [], [], []
    for res in sub_results:
        gather_star_BCG1.extend(res[0])
        gather_star_BCG2.extend(res[1])
        gather_star_in_2half_mass_rad.extend(res[2])

    # Add columns to subhalo data frame
    df_subs[f"M_star_BCG_{int(star_BCG_dist_cut1 * 1000)}kpc"] = gather_star_BCG1
    df_subs[f"M_star_BCG_{int(star_BCG_dist_cut2 * 1000)}kpc"] = gather_star_BCG2
    df_subs["M_star_in_2stellar_half_mass_rad"] = gather_star_in_2half_mass_rad

    # Save updated subhalo data frame
    sub_output_file = simulation.replace(r"-", r"_") + f"_subhalo_catalog_snap{snap}.csv"
    df_subs.to_csv(sub_output_file, index = False)


if __name__ == "__main__":
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
