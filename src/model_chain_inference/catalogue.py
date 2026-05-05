"""Catalogue filtering and histogram helpers for forecast evaluation."""


def catalogue_filter(cat, filterset, datetime_dim="datetime", size_dim="magnitude"):
    """Return a boolean mask for catalogue entries that satisfy a filter set."""
    cat = cat.sel(polygon=filterset["polygon"])
    filt = (
        # spatial
        (cat["polygon_distance"] <= 0.0)
        # temporal
        * (
            cat[datetime_dim]
            >= filterset["timeframe"].sel(epoch="start").astype("datetime64[ns]")
        )
        * (
            cat[datetime_dim]
            <= filterset["timeframe"].sel(epoch="finish").astype("datetime64[ns]")
        )
        # magnitude
        * (cat[size_dim] >= filterset["mmin"])
    ).astype(bool)
    return filt


def filter_catalogues(event_data, filterset, filterset_etas, event_id):
    """Return filtered target and parent-event catalogues for background and ETAS use."""
    filt = catalogue_filter(event_data, filterset)
    filt_etas = catalogue_filter(event_data, filterset_etas)
    id_sel = event_data[event_id].where(filt, drop=True).values
    id_sel_etas = event_data[event_id].where(filt_etas, drop=True).values

    # event attributes
    eqcat = event_data.sel(
        {event_id: id_sel, "polygon": filterset["polygon"]},
        drop=True,
    )
    eqcat_parent = event_data.sel(
        {event_id: id_sel_etas, "polygon": filterset_etas["polygon"]},
        drop=True,
    ).rename({event_id: "parent_id"})

    return eqcat, eqcat_parent


def get_realisation(
    grid,
    eqcat,
    filterset,
    event_id="event_id",
    datetime_dim="datetime",
    size_dim="magnitude",
):
    """Count filtered events on the cumulative bins defined by a grid."""
    # apply filters to mark catalogue entries as true or false
    realisation = catalogue_filter(eqcat, filterset)

    # assign events to bins in space
    if "x" in grid.coords and "y" in grid.coords:
        realisation = (
            realisation
            & (eqcat["x_grid"] == grid["x"])
            & (eqcat["y_grid"] == grid["y"])
        )

    # assign events to cumulative bins in time
    if datetime_dim in grid.dims:
        realisation = realisation & (eqcat[datetime_dim] <= grid[datetime_dim])

    # assign events to complementary cumulative bins in size
    if size_dim in grid.dims:
        realisation = realisation & (eqcat[size_dim] >= grid[size_dim])

    # sum over catalogue dimensions to count the boolean true values
    realisation = realisation.sum(event_id)

    # apply diff to get non-cumulative counts
    if datetime_dim in realisation.dims:
        realisation = realisation.diff(datetime_dim, label="lower")
    if size_dim in realisation.dims:
        realisation = -1 * realisation.diff(size_dim, label="lower")

    return realisation


__all__ = [
    "catalogue_filter",
    "filter_catalogues",
    "get_realisation",
]
