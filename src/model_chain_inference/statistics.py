import numpy as np
import xarray as xr
import scipy.stats as st
import scipy.special as sc


def get_cumulative_probabilities_poisson(observations, mean):
    sf = xr.apply_ufunc(
        st.poisson.sf,
        observations,  # exclusive sf: probability of exceeding observation value
        mean,
    )

    sf_inclusive = xr.apply_ufunc(
        st.poisson.sf,
        observations - 1,  # inclusive sf: probability of observation value or higher
        mean,
    )

    cdf = xr.apply_ufunc(
        st.poisson.cdf,
        observations,  # inclusive: probability of observation value or lower
        mean,
    )

    cdf_exclusive = xr.apply_ufunc(
        st.poisson.cdf,
        observations - 1,  # inclusive: probability of subceeding observation value
        mean,
    )

    ds = xr.Dataset(
        {
            "survival": sf,
            "survival_inclusive": sf_inclusive,
            "cumulative": cdf,
            "cumulative_exclusive": cdf_exclusive,
        }
    ).to_dataarray("p_metric")

    return ds


def get_cumulative_probabilities_nbinom(observations, mean, variance):
    """
    Compute exceedance probabilities for negative binomial distribution
    Parameters
    ----------
    observations : xr.DataArray
        Observations for which to compute exceedance probabilities
    mean : xr.DataArray
        Mean of the negative binomial distribution
    variance : xr.DataArray
        Variance of the negative binomial distribution
    Returns
    -------
    xr.DataArray
        DataArray containing the exceedance probabilities
    """

    sf = xr.apply_ufunc(
        st.nbinom.sf,
        observations,  # exclusive sf: probability of exceeding observation value
        *nbinom_pars_from_mean_var(mean, mean + variance),
    )

    sf_inclusive = xr.apply_ufunc(
        st.nbinom.sf,
        observations - 1,  # inclusive sf: probability of observation value or higher
        *nbinom_pars_from_mean_var(mean, mean + variance),
    )

    cdf = xr.apply_ufunc(
        st.nbinom.cdf,
        observations,  # inclusive: probability of observation value or lower
        *nbinom_pars_from_mean_var(mean, mean + variance),
    )

    cdf_exclusive = xr.apply_ufunc(
        st.nbinom.cdf,
        observations - 1,  # exclusive: probability of subceeding observation value
        *nbinom_pars_from_mean_var(mean, mean + variance),
    )

    ds = xr.Dataset(
        {
            "survival": sf,
            "survival_inclusive": sf_inclusive,
            "cumulative": cdf,
            "cumulative_exclusive": cdf_exclusive,
        }
    ).to_dataarray("p_metric")

    return ds


def get_cumulative_probabilities(observations, mean, variance=None):
    ds = xr.Dataset()
    ds["poisson"] = get_cumulative_probabilities_poisson(observations, mean)
    if variance is not None:
        ds["nbinom"] = get_cumulative_probabilities_nbinom(observations, mean, variance)

    return ds.to_dataarray("distribution")


def get_count_fractiles(mean, variance=None, q=None):
    if q is None:
        q = get_default_fractiles()

    ds = xr.Dataset()
    ds["poisson"] = xr.apply_ufunc(
        st.poisson.ppf,
        q,
        mean,
    )
    if variance is not None:
        ds["norm"] = xr.apply_ufunc(
            st.norm.ppf,
            q,
            mean,
            np.sqrt(variance),
        )
        ds["nbinom"] = xr.apply_ufunc(
            st.nbinom.ppf,
            q,
            *nbinom_pars_from_mean_var(mean, mean + variance),
        )

    return ds.to_dataarray("distribution")


def nbinom_pars_from_mean_var(mean, var):
    p = mean / var
    n = mean**2 / (var - mean)
    return n, p


def get_default_fractiles():
    q = xr.DataArray(
        [0.025, 0.975],
        dims="fractile",
        coords={
            "fractile": ["lower", "upper"],
        },
    )

    return q


def compute_log_likelihood(
    expected_counts,
    realized_counts,
    realized_entropy=None,
    bin_dims=None,
):
    """
    Compute log likelihoods for observed counts given expected counts, according to different
    models:
    - Binned Poisson: assumes counts in each bin are independent Poisson variables. There is no distinction
      between individual events in a bin.
    - Rate model: assumes counts are drawn from a single inhomogeneous Poisson process
      with a rate proportional to the expected counts in each bin, but the total count is not constrained.
      There is no distinction between individual events in a bin.
    - Single point process: assumes all counts are drawn from a single inhomogeneous Poisson point process
      with a rate proportional to the expected counts in each bin, and an evaluation of total event count.
      There is a distinction between individual events in a bin.
    - Multi point process: assumes counts are drawn from multiple independent inhomogeneous
      Poisson point processes, one for each bin. There is a distinction between individual events in a bin.
      The sum of the logs of individual event rates passed through observed_sum_log_expected_counts. This
      allows information of the finest level of detail to be used, even at coarser levels of binning.
      This basically means you have access to the full distribution of event rates, rather than just the
      binned counts.

    Parameters
    ----------
    expected_counts : xr.DataArray
        Expected counts (mean of the distribution) per bin
    realized_counts : xr.DataArray
        Observed counts per bin
    realized_entropy : xr.DataArray, optional
        Precomputed sum of observed counts times log of expected counts,
        by default None. If not provided, it will be computed internally.
    bin_dims : list of str, optional
        Dimensions representing the bins, by default None.
        If None, all dimensions will be used.
    Returns
    -------
    xr.DataArray
        DataArray containing the log likelihoods for different models.
    """

    multi_homogeneous_log_likelihood = xr.apply_ufunc(
        st.poisson.logpmf,
        realized_counts,
        expected_counts,
    ).sum(bin_dims)

    expected_count_total = expected_counts.sum(bin_dims)
    realized_count_total = realized_counts.sum(bin_dims)
    total_poisson_log_likelihood = xr.apply_ufunc(
        st.poisson.logpmf,
        realized_count_total,
        expected_count_total,
    )

    log_pmf = np.log(expected_counts / expected_count_total)
    realized_sum_log_pmf = xr.dot(  # normalized negative entropy
        realized_counts,
        log_pmf,
        dims=bin_dims,
    )
    single_heterogeneous_log_likelihood = (
        total_poisson_log_likelihood + realized_sum_log_pmf
    )

    if realized_entropy is None:
        # when this is the only or finest, most detailed level of rates
        # here, single and multi process likelihoods should be identical,
        # as the individual rates are replaced by the bin rates
        realized_entropy = xr.dot(
            realized_counts,
            -np.log(expected_counts),
            dims=bin_dims,
        )

    factorial_term = sc.gammaln(realized_counts + 1).sum(bin_dims)
    multi_heterogeneous_log_likelihood = (
        -expected_count_total - factorial_term - realized_entropy
    )

    ds = xr.Dataset(
        {
            "multi_hom_poisson": multi_homogeneous_log_likelihood,
            "single_het_poisson": single_heterogeneous_log_likelihood,
            "multi_het_poisson": multi_heterogeneous_log_likelihood,
        }
    )
    return ds.to_array("likelihood_function")
