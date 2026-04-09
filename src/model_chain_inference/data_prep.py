import xarray as xr
import chaintools.tools_grid as tgrid
from .model_core import etas_spatial
from .catalogue import filter_catalogues


def _serialize_attr_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value

def generate_inference_data_local(
    event_data,
    grid_data,
    covariate_id,
    measure_id,
    support_id,
    filterset,
    filterset_etas=None,
    event_index="event_id",
    grid_index="loc",
    target_step=None,
    model_dims=None,
    attributes=None,
    measure_xy_id="measure_xy",
    etas_d=None,
    etas_q=None,
):
    """
    Generate a dataset for inference from event data and grid data.

    Two parts:
    - Background model
    - ETAS model
    """
    # PART 0 - PREPROCESSING
    if target_step is None:
        target_step = [0.01]
    elif isinstance(target_step, (int, float)):
        target_step = [target_step]

    if filterset_etas is None:
        filterset_etas = filterset
    if attributes is None:
        attribute_ids = []
        attribute_steps = []
    else:
        attribute_ids = list(attributes.keys())
        attribute_steps = list(attributes.values())

    # Select support data according to filterset
    if "polygon" in grid_data.dims:
        grid_data = grid_data.sel({"polygon": filterset["polygon"]})

    # subsurface data - coming from either a regular grid or irregular fault data
    measure_grid = grid_data[measure_id].load().fillna(0)
    support_grid = grid_data[support_id].load().fillna(0)
    support_measure_grid = support_grid * measure_grid
    area_grid = grid_data[measure_xy_id].load().fillna(0)
    support_area_grid = support_grid * area_grid

    # select grid cells where support is nonzero - for etas support coverage
    support_area_nodes = support_area_grid.reset_index(grid_index).rename(
        {grid_index: "__loc__"}
    )
    support_area_nodes = support_area_nodes.where(support_area_nodes > 0.0, drop=True)

    # select grid cells where measure is nonzero
    any_dims = [d for d in measure_grid.dims if d not in [grid_index]]
    available = (support_measure_grid > 0).any(any_dims)
    grid_samples = grid_data.where(available, drop=True)

    covariate_samples = grid_samples[covariate_id].load().fillna(0)
    attribute_samples = grid_samples[attribute_ids].load().fillna(0)
    measure_samples = grid_samples[measure_id].load().fillna(0)
    support_samples = grid_samples[support_id].load().fillna(0)
    measure_samples = measure_samples * support_samples

    # normalize (stress) covariate at maximum time
    norm_epoch = filterset["timeframe"].max().item()
    covariate_scale = get_scale_factor(
        covariate_samples, grid_index, norm_epoch, model_dims
    )
    covariate_samples = covariate_samples / covariate_scale

    # combine target samples - covariate + attributes
    target_samples = xr.merge([covariate_samples, attribute_samples])

    # interpolate at interval boundaries
    epochs = filterset["timeframe"].astype("datetime64[ns]")
    target_samples = target_samples.interp({"datetime": epochs})
    tg_step = target_step + attribute_steps
    measure_exposed = tgrid.aggregate_to_grid(
        samples=target_samples,
        target_step=tg_step,
        weights=measure_samples,
        marginalize_dims=grid_index,
        order=1,
    ).rename({covariate_id: "covariate"})

    # get catalogues
    eqcat, eqcat_parent = filter_catalogues(
        event_data,
        filterset,
        filterset_etas,
        event_index,
    )

    # relative parent-child attributes
    event_delays = tgrid.xr_delay(eqcat_parent, eqcat)
    event_distances = tgrid.xr_distance(eqcat_parent, eqcat)
    epoch_delays = tgrid.xr_delay(eqcat_parent, measure_exposed)

    # prepare spatial rates
    etas_spatial_rates = etas_spatial(event_distances, etas_d, etas_q)

    # get radial support coverage for all parent events
    radii_parents = tgrid.xr_distance(support_area_nodes, eqcat_parent)
    radial_weights_parents = etas_spatial(radii_parents, etas_d, etas_q)
    etas_support_coverage = xr.dot(
        support_area_nodes, radial_weights_parents, dim="__loc__"
    )

    # extract the covariates and measures at the event times and locations
    prep_grid_selection = tgrid.prepare_grid_selection(eqcat, covariate_samples)
    covariate_observed = (
        covariate_samples.unstack("loc")
        .sel(
            {
                "datetime": prep_grid_selection["datetime_grid"],
                "x": prep_grid_selection["x_grid"],
                "y": prep_grid_selection["y_grid"],
            }
        )
        .rename({"datetime": "datetime_event"})
    )
    measure_observed = measure_samples.unstack("loc").sel(
        {
            "x": prep_grid_selection["x_grid"],
            "y": prep_grid_selection["y_grid"],
        }
    )
    measure_xyt_observed = (
        grid_data["measure_xyt"]
        .load()
        .unstack("loc")
        .sel(
            {
                "datetime": prep_grid_selection["datetime_grid"].sel(
                    {"datetime_bound": "lower"}
                ),
                "x": prep_grid_selection["x_grid"],
                "y": prep_grid_selection["y_grid"],
            }
        )
        .drop_vars("datetime")
    )

    # make sure the measure is expressed in terms of a time and space units compatible with
    # the later etas contribution : we choose square meters and days
    measure_observed = measure_observed / measure_xyt_observed

    # create epoch histogram
    cumul_histogram = (
        (eqcat["datetime"] <= measure_exposed["datetime"])
        .sum(event_index)
        .drop_vars("datetime")
    )
    time_histogram = cumul_histogram.isel(epoch=-1) - cumul_histogram.isel(epoch=0)

    # space-time-magnitude coordinates
    spacetime = xr.Dataset(
        {
            "x": eqcat["x"],
            "y": eqcat["y"],
            "t": eqcat["datetime"].dt.second.astype(float),
        }
    ).to_array("xyt")
    spacetime_magnitude = xr.Dataset(
        {
            "x": eqcat["x"],
            "y": eqcat["y"],
            "t": eqcat["datetime"].dt.second.astype(float),
            "magnitude": eqcat["magnitude"],
        }
    ).to_array("xytm")

    # print("create dataset")
    ds = xr.Dataset(
        {
            # Catalogue data
            "event_count": time_histogram,
            #
            # Event data
            "spacetime": spacetime,
            "spacetime_magnitude": spacetime_magnitude,
            "magnitude": eqcat["magnitude"],
            "covariate_observed": covariate_observed,
            "measure_observed": measure_observed,
            #
            # Relative (etas) data
            "parent_magnitude": eqcat_parent["magnitude"],
            "event_delays": event_delays,
            "epoch_delays": epoch_delays,
            "event_distances": event_distances,
            "etas_support_coverage": etas_support_coverage,
            "etas_spatial_rates": etas_spatial_rates,
            #
            # Epoch data
            "covariate_scale": covariate_scale,
            "covariate_exposed": measure_exposed["covariate"],
            "measure_exposed": measure_exposed,
            #
            "m_min": filterset["mmin"],
        }
    ).reset_coords(drop=True)

    if attribute_ids:
        attributes_observed = (
            attribute_samples.unstack("loc")
            .sel(
                {
                    "x": prep_grid_selection["x_grid"],
                    "y": prep_grid_selection["y_grid"],
                }
            )
            .rename({id: id + "_observed" for id in attribute_ids})
        )
        ds = ds.merge(attributes_observed)
        for attr_id in attribute_ids:
            ds[attr_id + "_exposed"] = ds[attr_id]

    ds.attrs.update(
        {key: _serialize_attr_value(value) for key, value in measure_exposed.attrs.items()}
    )
    ds.attrs["covariate_id"] = covariate_id
    ds.attrs["measure_id"] = measure_id
    ds.attrs["support_id"] = support_id
    ds.attrs["polygon"] = _serialize_attr_value(filterset["polygon"].values[()])
    ds.attrs["timeframe"] = _serialize_attr_value(filterset["timeframe"].values)
    ds.attrs["mmin"] = _serialize_attr_value(filterset["mmin"].values[()])
    ds.attrs["polygon_etas"] = _serialize_attr_value(filterset_etas["polygon"].values[()])
    ds.attrs["timeframe_etas"] = _serialize_attr_value(filterset_etas["timeframe"].values)
    ds.attrs["mmin_etas"] = _serialize_attr_value(filterset_etas["mmin"].values[()])
    ds.attrs["dsm_mode"] = "local"

    return ds


def get_scale_factor(samples, aggregate_dims, epoch, model_dims):
    """
    Get the scale factor for the given samples at a specific epoch.
    This function calculates the maximum value of the samples at the specified epoch,
    excluding the dimensions specified in aggregate_dims and model_dims.
    Parameters
    ----------
    samples : xr.Dataset
        The dataset containing the samples with temporal coordinates.
    aggregate_dims : list or str
        The dimensions to be aggregated over when calculating the scale factor.
    epoch : datetime
        The epoch at which the scale factor is calculated.
    model_dims : list or str, optional
        The dimensions that should not be included in the scale factor calculation.
    Returns
    -------
    xr.DataArray
        The scale factor calculated as the maximum value of the samples at the specified epoch,
        excluding the specified dimensions.
    """
    if isinstance(aggregate_dims, str):
        aggregate_dims = [aggregate_dims]

    time_slice = samples.interp({"datetime": epoch}).drop_vars("datetime")

    if model_dims is None:
        model_dims = []
    elif isinstance(model_dims, str):
        model_dims = [model_dims]

    no_scale_dims = list(set(model_dims).union(aggregate_dims))
    scale = time_slice.max(no_scale_dims)

    return scale


def calculate_radial_support_coverage(
    radial_step, radial_stop, support_index, support_nodes, measure_samples
):
    """
    Calculate the radial support coverage for the given support nodes and measure samples.
    This function computes the radial distances between support nodes and measure samples,
    and aggregates these distances into a grid with specified radial steps and stops.
    Parameters
    ----------
    radial_step : float
        The step size for the radial distances.
    radial_stop : float
        The maximum radial distance to consider.
    support_index : str
        The index name for the support nodes.
    support_nodes : xr.DataArray
        The support nodes for which the radial distances will be calculated.
    measure_samples : xr.DataArray
        The measure samples to which the radial distances will be calculated.
    Returns
    -------
    xr.DataArray
        A dataset containing the radial support coverage, aggregated to the specified radial steps.
    """
    radii = tgrid.xr_distance(
        support_nodes,
        measure_samples,
    ).rename("radial_distance")

    # if we will smooth rates, not stresses, then we can simply map out where
    # smoothed values will end up, and to what extent these will end up within
    # the support area
    # for each distance (interval) we can calculate the area of the annulus overlapping
    # with the support area
    # the weighting will be done during the inference, when the smoothing kernel is determined
    radial_support_coverage = tgrid.aggregate_to_grid(
        samples=radii,
        weights=support_nodes,
        target_step=radial_step,
        target_stop=radial_stop,
        marginalize_dims=support_index,
        order=1,
    )

    return radial_support_coverage


def extract_observed_attributes(event_data, attribute_grid, grid_id):
    """
    Extract observed attributes from the event data based on the spatial grid selection.
    This function prepares the spatial grid selection for the event data and extracts the
    observed attributes from the attribute grid. It returns a dataset containing the observed
    attributes at the event locations.
    Parameters
    ----------
    event_data : xr.Dataset
        The dataset containing the event data with spatial coordinates.
    attribute_grid : xr.Dataset
        The dataset containing the attribute grid data with spatial coordinates.
    grid_id : str
        The identifier for the grid dimension in the attribute grid.
    Returns
    -------
    xr.Dataset
        A dataset containing the observed attributes at the event locations.
    """
    event_grid_prep = tgrid.prepare_spatial_grid_selection(event_data, attribute_grid)
    attrs_observed = (
        attribute_grid.unstack(grid_id)
        .sel(
            {
                "x": event_grid_prep["x_grid"],
                "y": event_grid_prep["y_grid"],
            },
        )
        .drop_vars(["x", "y"])
    )

    return attrs_observed
