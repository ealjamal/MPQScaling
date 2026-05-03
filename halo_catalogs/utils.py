import json

import h5py
import numpy as np


def gapper_vel_disp(velocities):
    """
    Compute the gapper velocity dispersion estimator for a 1D velocity array.

    The gapper estimator is a robust, order-statistic-based measure of scale:
        sigma_G = (sqrt(pi) / (n*(n-1))) * sum_i [ w_i * g_i ]
    where g_i = v_{i+1} - v_i are the ordered gaps and w_i = i*(n-i) are
    their weights.

    Parameters
    ----------
    velocities : array-like
        1D array of velocities (km/s).

    Returns
    -------
    float
        Gapper velocity dispersion. Returns -inf if fewer than 2 velocities
        are provided.
    """
    
    sorted_v = np.sort(velocities)
    n = len(sorted_v)
    if n <= 1:
        return float("-inf")
    gaps = sorted_v[1:] - sorted_v[:-1]
    weights = np.arange(1, n) * (n - np.arange(1, n))
    weighted_gaps = gaps * weights
    sigma_G = np.sqrt(np.pi) / (n * (n - 1)) * np.sum(weighted_gaps)

    return sigma_G


def gapper_vel_disp_bootstrap(velocities, num_bootstrap=1000, random_seed=0):
    """
    Estimate the median gapper velocity dispersion via bootstrap resampling.

    Draws `num_bootstrap` samples with replacement from `velocities`, computes
    the gapper dispersion for each, and returns the median of the distribution.

    Parameters
    ----------
    velocities : array-like
        1D array of velocities (km/s).
    num_bootstrap : int
        Number of bootstrap realizations.
    random_seed : int
        Seed for the random number generator (reproducibility).

    Returns
    -------
    float
        Median gapper velocity dispersion across bootstrap samples.
        Returns nan if no valid samples are produced.
    """
    rng = np.random.default_rng(seed=random_seed)
    sigma_G_samples = []
    for _ in range(num_bootstrap):
        sample = rng.choice(velocities, size=len(velocities), replace=True)
        sigma_G = gapper_vel_disp(sample)
        if not np.isnan(sigma_G):
            sigma_G_samples.append(sigma_G)

    sigma_G_median = np.percentile(sigma_G_samples, 50) if sigma_G_samples else np.nan

    return sigma_G_median


def log_stellar_mass_gap(log_masses, num_star_particles, brighter_ind, fainter_ind, min_num_star_particles=1):
    """
    Compute the log stellar mass gap between the i-th and j-th brightest subhalos.

    Subhalos with fewer than `min_num_star_particles` star particles are excluded
    before ranking. The gap is defined as:
        gap = log10( M_brighter - M_fainter )
    where masses are in linear units before subtraction.

    Parameters
    ----------
    log_masses : array-like
        Log10 stellar masses of subhalos (log10 Msol).
    num_star_particles : array-like
        Number of star particles per subhalo; used to filter resolved subhalos.
    brighter_ind : int
        Rank of the brighter subhalo (1 = most massive).
    fainter_ind : int
        Rank of the fainter subhalo (e.g. 2 or 4).
    min_num_star_particles : int
        Minimum star particle count for a subhalo to be included.

    Returns
    -------
    float
        Log10 of the stellar mass difference. Returns nan if fewer than
        `fainter_ind` subhalos pass the particle cut.
    """

    log_masses = np.asarray(log_masses)
    num_star_particles = np.asarray(num_star_particles)
    log_masses = log_masses[num_star_particles >= min_num_star_particles]
    if len(log_masses) < fainter_ind:
        return np.nan
    sorted_masses = np.sort(log_masses)[::-1]
    mass_gap = 10 ** sorted_masses[brighter_ind - 1] - 10 ** sorted_masses[fainter_ind - 1]

    return np.log10(mass_gap)


def save_h5(df, out_path, label, snap, min_sub_M_star, num_bootstrap):
    """
    Save an augmented halo catalog DataFrame to an HDF5 file with metadata.

    The dataset is stored under the key 'data' as float64. Column names are
    saved in the 'properties' attribute and a JSON metadata string is stored
    in the 'metadata' attribute.

    Parameters
    ----------
    df : pandas.DataFrame
        Augmented halo catalog to save.
    out_path : str or Path
        Output file path (will be created or overwritten).
    label : str
        Simulation label used in the description string (e.g. 'TNG300-1').
    snap : int or str
        Snapshot identifier used in the description string.
    min_sub_M_star : float
        Minimum log10 stellar mass threshold applied to satellite subhalos.
    num_bootstrap : int
        Number of bootstrap samples used for the velocity dispersion estimate.
    """
    
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset("data", data=df.to_numpy(), dtype=np.float64)
        dset.attrs["properties"] = np.array(df.columns, dtype="S")
        description = (
            f"Halo catalog from {label} at snapshot {snap}. "
            "Hot gas: T >= 1e5 K; cold gas: T <= 10^4.5 K. "
            f"Min subhalo log stellar mass: {min_sub_M_star} Msol. "
            f"Velocity dispersion: log gapper median ({num_bootstrap} bootstrap samples)."
        )
        dset.attrs["metadata"] = json.dumps({
            "description": description,
            "author": "Eddie Aljamal",
            "scales": "All fields log-scale except R_500c, R_200c.",
            "units": {
                "masses": "Msol", "Radii": "Mpc", "Temperatures": "K",
                "luminosities": "erg/s", "Y_X": "Msol K", "Y_SZ": "Msol K",
                "SFR": "Msol/yr", "sSFR": "1/yr", "velocity dispersion": "km/s",
            },
        })
