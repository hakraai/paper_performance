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
    r2 = r * r
    return ((q - 1) / (math.pi * d)) * ((d / (d + r2)) ** q)


# def etas_spatial_decay_cumulative(r, d, q):
#     r2 = r * r
#     return 1 - (d / (d + r2)) ** (q - 1)


def etas_productivity(M, K, a, M_ref):
    return K * np.exp(a * (M - M_ref))


# def etas_temporal(t, M, K, a, c, p, cumulative = False):
#     # note that M should be relative to a reference (default: M=0)
#     prod = etas_productivity(M, K, a)
#     if cumulative:
#         g = etas_temporal_cumulative(t, c, p)
#     else:
#         g = etas_temporal(t, c, p)
#     return prod * g


def etas_full(t, r, M, K, a, c, d, p, q, M_ref):
    pr = etas_productivity(M, K, a, M_ref)
    sp = etas_spatial(r, d, q)
    tm = etas_temporal(t, c, p)
    return pr * sp * tm


def etas_full_cumulative(t, r, M, K, a, c, d, p, q, M_ref):
    # note that M should be relative to a reference (default: M=0)
    pr = etas_productivity(M, K, a, M_ref)
    sp = etas_spatial(r, d, q)
    tm = etas_temporal_cumulative(t, c, p)
    return pr * sp * tm


# gaussian smoother
def r_smooth(r, sigma):
    return np.exp(-(r**2) / (2.0 * sigma**2))


# extreme threshold failure rate functions
def extreme_threshold_failure(c, theta_0, theta_1):
    return np.exp(theta_0 + theta_1 * c)


def extreme_threshold_failure_rate(c, c_dot, theta_0, theta_1):
    return theta_1 * c_dot * np.exp(theta_0 + theta_1 * c)


def rs_activator(c, c_loc, c_scale):
    return hyperbolic_tangent(c, 0.0, 1.0, c_loc, c_scale)


def rs_exponential(c, c_loc, c_scale_trend, c_scale_activator):
    # this is the instantaneous rate of the RS model, Heimisson (8)
    # but with a hyperbolic tangent activator instead of a heaviside
    # c_loc = DeltaS_c - threshold
    # c_scale_trend = 1 / (Asigma_0),
    # c_scale_activator -> new parameter, controls the width of the activator,`
    # original is retrieved when c_scale_activator -> 0
    # however this should be avoided
    # ~ exp(theta_0 + theta_1 * c) form:
    # ~ theta_0 = - (DeltaS_c/(Asigma_0))
    # ~ theta_1 = (1/Asigma_0)
    activator = rs_activator(c, c_loc, c_scale_activator)
    trend = np.exp((c - c_loc) / c_scale_trend)
    return activator * trend


def rs_exponential_dieterich(c, c_scale_trend):
    trend = np.exp(c / c_scale_trend)
    return trend


def rs_rate(rs_exp, rs_exp_int, r, t_a):
    # Heimisson equation (1)
    return r * t_a * rs_exp / (rs_exp_int + t_a)


def rs_rate_instantaneous(rs_exp, r):
    # Heimisson equation (8)
    return r * rs_exp


# b-value functions
def hyperbolic_tangent(c, b0, b1, loc, scale):
    return 0.5 * ((b1 + b0) + (b1 - b0) * np.tanh((c - loc) / scale))


def inverse_power_law(c, b0, loc, scale, pw):
    return b0 + np.power((c - loc) / scale, -pw)


def linear(c, b0, loc, scale):
    return b0 + (c - loc) / scale


# exponential distribution functions
def ll_exponential(beta, dm):
    # beta == ln(10)b
    # dm == m - m0
    return np.log(beta) - beta * dm  # log-likelihood


# subsurface functions
def poroelastic_modulus(bulk_modulus, solid_modulus):
    # bulk_modulus = H_r
    # solid_modulus = H_s
    return 1 / (1 / bulk_modulus + 1 / solid_modulus)


def biot_coefficient(compressibility, bulk_modulus, solid_modulus):
    # note that in elastic media and in general: bulk_modulus = 1 / compressibility
    # however we may want to make a distintion between the two
    biot = compressibility * poroelastic_modulus(bulk_modulus, solid_modulus)
    return biot


def incremental_stress_high_compressibility(pressure_drop, gradient, poisson_ratio):
    # to arrive at the incremental stress, we need to multiply the following
    # by the biot coefficient
    gamma = (1 - 2 * poisson_ratio) / (2 - 2 * poisson_ratio)
    return pressure_drop * gradient * gamma


def stress_susceptibility(gradient, poisson_ratio, bulk_modulus, solid_modulus):
    compressibility = 1.0 / bulk_modulus
    biot = biot_coefficient(compressibility, bulk_modulus, solid_modulus)
    gamma = (1 - 2 * poisson_ratio) / (2 - 2 * poisson_ratio)
    return gradient * gamma * biot


def susceptibility_modulator(bulk_modulus, solid_modulus, bulk_modulus_ref):
    compressibility_ref = 1.0 / bulk_modulus_ref
    compressibility = 1.0 / bulk_modulus
    biot = biot_coefficient(compressibility, bulk_modulus, solid_modulus)
    biot_ref = biot_coefficient(compressibility_ref, bulk_modulus_ref, solid_modulus)
    return biot / biot_ref


def incremental_stress(
    pressure_drop, gradient, poisson_ratio, bulk_modulus, solid_modulus
):
    ssusc = stress_susceptibility(gradient, poisson_ratio, bulk_modulus, solid_modulus)
    return ssusc * pressure_drop


def incremental_stress_from_strain(
    vertical_strain, gradient, poisson_ratio, bulk_modulus, solid_modulus
):
    # this allows a distinction between the bulk_modulus and the compressibility
    # that has cause the vertical strain
    pormod = poroelastic_modulus(bulk_modulus, solid_modulus)
    gamma = (1 - 2 * poisson_ratio) / (2 - 2 * poisson_ratio)
    return vertical_strain * gradient * gamma * pormod


def vertical_strain(pressure_drop, compressibility):
    return pressure_drop * compressibility


def compaction(thickness, pressure_drop, compressibility):
    return thickness * pressure_drop * compressibility


def radial_normal_weight(r, sigma, dim=2):
    return np.exp(-0.5 * (r / sigma) ** 2) / np.sqrt(2 * np.pi * sigma**2) ** dim 
