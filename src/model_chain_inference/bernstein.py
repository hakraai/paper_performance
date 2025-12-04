"""
Module for evaluating Bernstein basis functions for a given degree over a range of values.
This module provides functions to create Bernstein basis functions, either for a 1D array of values
or for an xarray DataArray. The basis functions can be used in various applications, such as
statistical modeling or interpolation.
"""

import numpy as np
import scipy.interpolate as si
import xarray as xr


def bernstein_partition_core(vals, degree):
    """
    Evaluate Bernstein basis functions for a given degree over the range of values.
    The values are assumed to be sorted, and the basis functions are created
    such that they cover the entire range of values.
    Args:
        vals (np.ndarray): 1D array of values to create basis functions for.
        degree (int): Degree of the Bernstein polynomial.
    Returns:
        np.ndarray: 2D array of shape (len(vals), degree + 1)
                    containing the Bernstein basis functions evaluated at the values.
    """
    mn, mx = vals.min(), vals.max()  # put entire range in one section
    nodeweights = np.diag((degree + 1) * [1])[:, :, None]
    basis_functions = [si.BPoly(w, [mn, mx]) for w in nodeweights]
    ret = np.array([f(vals) for f in basis_functions])
    return np.moveaxis(ret, 0, -1)


def bernstein_partition(values, degree, fractiles=True, dim=None):
    """
    Evaluate Bernstein basis functions for a given degree over the range of values.
    The values are assumed to be sorted, and the basis functions are created
    such that they cover the entire range of values.
    Args:
        values (xr.DataArray): 1D array of values to create basis functions for.
        degree (int): Degree of the Bernstein polynomial.
        fractiles (bool): If True, the values are sorted and then transformed to fractiles
                         before creating the Bernstein basis functions.
    Returns:
        xr.DataArray: 2D array of shape (len(values), degree + 1)
                      containing the Bernstein basis functions evaluated at the values.
    """
    if dim is None:
        dim = values.dims[-1]
    if fractiles:
        axis = values.get_axis_num(dim)
        values = values.pipe(np.argsort, axis=axis).pipe(np.argsort, axis=axis)
    b_weights = (
        xr.apply_ufunc(
            bernstein_partition_core,
            values,
            degree,
            input_core_dims=[[dim], []],
            output_core_dims=[[dim, "bernstein_basis"]],
            vectorize=True,
        )
        .assign_coords(
            {
                "bernstein_degree": ("bernstein_basis", np.repeat(degree, degree + 1)),
                "bernstein_index": ("bernstein_basis", np.arange(degree + 1)),
            }
        )
        .set_xindex(["bernstein_degree", "bernstein_index"])
        .rename("bernstein_weights")
    )

    return b_weights
