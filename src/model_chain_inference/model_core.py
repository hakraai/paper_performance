"""Core physical and statistical kernels used by the inference models."""

import math
import numpy as np


# ETAS functions
def etas_temporal(t, c, p):
    """ETAS temporal kernel.

    This function returns the ETAS temporal kernel, which is a time-dependent
    decay function. The ETAS kernel is a modification of the Omori-Utsu
    decay function, which is used to describe aftershocks. The ETAS kernel
    is described by the following equation:

    k(t) = (p-1)/c * (c/(c+t))**p

    where t is time, c is a decay constant, and p is a power-law exponent.

    Parameters
    ----------
    t : array_like
        Time
    c : float
        Decay constant
    p : float
        Power-law exponent

    Returns
    -------
    array_like
        ETAS temporal kernel

    """
    # tclip = np.maximum(t, 0.0)
    # filt = np.heaviside(t, 0.0)
    return ((p - 1) / c) * ((c / (c + t)) ** p)


def etas_temporal_cumulative(t, c, p):
    """
    Returns the cumulative ETAS decay function for the given parameters.

    Parameters
    ----------
    t : float or array
        The time(s) for which to compute the ETAS decay function.
    c : float
        The ETAS decay parameter.
    p : float
        The ETAS power law exponent.

    Returns
    -------
    float or array
        The value(s) of the ETAS decay function for the given time(s).
    """
    # tclip = np.maximum(t, 0.0)
    # filt = np.heaviside(t, 0.0)
    return 1 - (c / (c + t)) ** (p - 1)


def etas_spatial(r, d, q):
    """Return the normalized ETAS spatial kernel for radial distance values."""
    r2 = r * r
    return ((q - 1) / (math.pi * d)) * ((d / (d + r2)) ** q)


# def etas_spatial_decay_cumulative(r, d, q):
#     r2 = r * r
#     return 1 - (d / (d + r2)) ** (q - 1)


def etas_productivity(M, K, a, M_ref):
    """Return ETAS productivity as an exponential function of parent magnitude."""
    return K * np.exp(a * (M - M_ref))


# extreme threshold failure rate functions
def extreme_threshold_failure(c, theta_0, theta_1):
    """Return the cumulative ETF response for a covariate field."""
    return np.exp(theta_0 + theta_1 * c)


# b-value functions
def hyperbolic_tangent(c, b0, b1, loc, scale):
    """Evaluate a bounded hyperbolic-tangent transition between two levels."""
    return 0.5 * ((b1 + b0) + (b1 - b0) * np.tanh((c - loc) / scale))


# subsurface functions
def poroelastic_modulus(bulk_modulus, solid_modulus):
    """Compute the combined poroelastic modulus from bulk and solid moduli."""
    # bulk_modulus = H_r
    # solid_modulus = H_s
    return 1 / (1 / bulk_modulus + 1 / solid_modulus)


def biot_coefficient(compressibility, bulk_modulus, solid_modulus):
    """Compute the Biot coefficient from compressibility and elastic moduli."""
    # note that in elastic media and in general: bulk_modulus = 1 / compressibility
    # however we may want to make a distintion between the two
    biot = compressibility * poroelastic_modulus(bulk_modulus, solid_modulus)
    return biot
def stress_susceptibility(gradient, poisson_ratio, bulk_modulus, solid_modulus):
    """Compute the proportionality between pressure drop and stress change."""
    compressibility = 1.0 / bulk_modulus
    biot = biot_coefficient(compressibility, bulk_modulus, solid_modulus)
    gamma = (1 - 2 * poisson_ratio) / (2 - 2 * poisson_ratio)
    return gradient * gamma * biot


def incremental_stress(
    pressure_drop, gradient, poisson_ratio, bulk_modulus, solid_modulus
):
    """Compute incremental stress from pressure drop and elastic properties."""
    ssusc = stress_susceptibility(gradient, poisson_ratio, bulk_modulus, solid_modulus)
    return ssusc * pressure_drop


def radial_normal_weight(r, sigma, dim=2):
    """Return Gaussian radial weights normalized for the requested dimension."""
    return np.exp(-0.5 * (r / sigma) ** 2) / np.sqrt(2 * np.pi * sigma**2) ** dim


__all__ = [
    "biot_coefficient",
    "etas_productivity",
    "etas_spatial",
    "etas_temporal",
    "etas_temporal_cumulative",
    "extreme_threshold_failure",
    "hyperbolic_tangent",
    "incremental_stress",
    "poroelastic_modulus",
    "radial_normal_weight",
    "stress_susceptibility",
]
