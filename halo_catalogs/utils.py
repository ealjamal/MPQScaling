import json

import h5py
import numpy as np

# Comoving box side lengths in Mpc
BOX_SIZE_COMOVING = {
    "TNG300-1":    302.627694124594,
    "TNG300-2":    302.627694124594,
    "TNG-Cluster": 1003.8382049010925,
    "L1000N3600":  1000.00000048,
    "L1000N1800":  1000.00000048,
}

# Snapshot number → redshift for each simulation family
_TNG_SNAP_TO_Z = {
    2: 12, 3: 11, 4: 10, 6: 9, 8: 8, 11: 7, 13: 6, 17: 5, 21: 4, 25: 3,
    33: 2, 40: 1.5, 50: 1, 59: 0.7, 67: 0.5, 72: 0.4, 78: 0.3, 84: 0.2,
    91: 0.1, 99: 0,
}
SNAP_TO_Z = {
    "TNG300-1":    _TNG_SNAP_TO_Z,
    "TNG300-2":    _TNG_SNAP_TO_Z,
    "TNG-Cluster": _TNG_SNAP_TO_Z,
    "L1000N3600":  {10: 5, 14: 4, 18: 3, 38: 2, 48: 1.5, 58: 1, 63: 0.75,
                    68: 0.5, 70: 0.4, 72: 0.3, 74: 0.2, 76: 0.1, 78: 0},
    "L1000N1800":  {9: 5, 13: 4, 17: 3, 37: 2, 47: 1.5, 57: 1, 62: 0.75,
                    67: 0.5, 69: 0.4, 71: 0.3, 73: 0.2, 75: 0.1, 77: 0},
}


def periodic_dist(dx, dy, dz, boxsize):
    """
    Euclidean distance with periodic boundary wrapping.

    Applies the minimum-image convention along each axis independently
    before computing the 3D Euclidean norm. Operates element-wise on
    array inputs, so each row can have a different displacement.

    Parameters
    ----------
    dx, dy, dz : array-like
        Pre-computed displacement components (same units as boxsize).
    boxsize : float
        Side length of the periodic box (same units as displacements).

    Returns
    -------
    numpy.ndarray
        Periodic Euclidean distances.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dz = np.asarray(dz, dtype=float)
    dx -= boxsize * np.round(dx / boxsize)
    dy -= boxsize * np.round(dy / boxsize)
    dz -= boxsize * np.round(dz / boxsize)

    return np.sqrt(dx**2 + dy**2 + dz**2)


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
        Gapper velocity dispersion. Returns nan if fewer than 2 velocities
        are provided.
    """
    
    sorted_v = np.sort(velocities)
    n = len(sorted_v)
    if n <= 1:
        return np.nan
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
