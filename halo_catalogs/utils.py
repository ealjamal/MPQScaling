import json

import h5py
import numpy as np


def gapper_vel_disp(velocities):
    sorted_v = np.sort(velocities)
    n = len(sorted_v)
    if n <= 1:
        return float("-inf")
    gaps = sorted_v[1:] - sorted_v[:-1]
    weights = np.arange(1, n) * (n - np.arange(1, n))
    return np.sqrt(np.pi) / (n * (n - 1)) * np.sum(gaps * weights)


def gapper_vel_disp_bootstrap(velocities, num_bootstrap=1000, random_seed=0):
    rng = np.random.default_rng(seed=random_seed)
    bootstrap_vals = []
    for _ in range(num_bootstrap):
        sample = rng.choice(velocities, size=len(velocities), replace=True)
        val = gapper_vel_disp(sample)
        if not np.isnan(val):
            bootstrap_vals.append(val)
    return np.percentile(bootstrap_vals, 50) if bootstrap_vals else np.nan


def log_stellar_mass_gap(log_masses, num_star_particles, brighter_ind, fainter_ind, min_num_star_particles=1):
    log_masses = np.asarray(log_masses)
    num_star_particles = np.asarray(num_star_particles)
    log_masses = log_masses[num_star_particles >= min_num_star_particles]
    if len(log_masses) < fainter_ind:
        return np.nan
    sorted_masses = np.sort(log_masses)[::-1]
    return np.log10(10 ** sorted_masses[brighter_ind - 1] - 10 ** sorted_masses[fainter_ind - 1])


def save_h5(df, out_path, label, snap, min_sub_M_star, num_bootstrap):
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
