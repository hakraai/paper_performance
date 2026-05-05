from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

import chaintools.tools_geometry as tgeo
import chaintools.tools_grid as tgrid
import model_chain_inference.bernstein as bernstein
import model_chain_inference.generate_data as generate_data
import model_chain_inference.model_core as model
from workflow_support.logging import get_logger, progress


LOGGER = get_logger(__name__)


def expected_source_data_outputs(source_data_root: Path) -> list[Path]:
    return [
        source_data_root / "event_data.h5",
        source_data_root / "fault_data.h5",
        source_data_root / "grid_data.h5",
        source_data_root / "groningen_polygons.cpg",
        source_data_root / "groningen_polygons.dbf",
        source_data_root / "groningen_polygons.prj",
        source_data_root / "groningen_polygons.shp",
        source_data_root / "groningen_polygons.shx",
    ]


def resolve_catalogue_end(config: dict[str, object]) -> pd.Timestamp | None:
    configured = config.get("catalogue_end")
    if configured:
        return pd.Timestamp(configured)
    return None


def coordinate_values(config: dict[str, object], key: str, default: np.ndarray) -> np.ndarray:
    values = np.asarray(config.get(key, default), dtype=float)
    if values.ndim == 1 and values.size >= 3:
        steps = np.diff(values)
        if np.allclose(steps, steps[0], rtol=0.0, atol=1e-12):
            return np.linspace(values[0], values[-1], values.size, dtype=float)
    return values


def build_parameters(grid_data: xr.Dataset, config: dict[str, object]) -> xr.Dataset:
    return xr.Dataset(
        {
            "rmax": ("rmax", coordinate_values(config, "rmax", np.linspace(0.375, 0.525, 7))),
            "sigma": ("sigma", coordinate_values(config, "sigma", np.linspace(2200, 3800, 9))),
            "hs_exp": grid_data["hs_exp"],
            "M_plastic": grid_data["M_plastic"],
        }
    )


def load_raw_inputs(
    resources_root: Path,
    catalogue_end: pd.Timestamp | None,
) -> tuple[gpd.GeoDataFrame, xr.Dataset, xr.Dataset, xr.Dataset]:
    polygon_data = generate_data.get_polygon_gdf(resources_root)
    fault_data = generate_data.get_faults(resources_root)
    catalogue_kwargs: dict[str, object] = {}
    if catalogue_end is not None:
        catalogue_kwargs["endtime"] = catalogue_end.isoformat()
    event_data = generate_data.get_catalogue(**catalogue_kwargs)
    grid_data = generate_data.get_grid(resources_root)

    grid_data["compressibility"] = grid_data["compressibility"] / 10.0
    grid_data["compressibility"].attrs["units"] = "1/bar"
    return polygon_data, fault_data, event_data, grid_data


def prepare_polygons(
    polygon_data: gpd.GeoDataFrame,
    grid_data: xr.Dataset,
    support_buffer: float,
) -> xr.DataArray:
    grid_data["support"] = (
        xr.full_like(grid_data["x"] * grid_data["y"], 1)
        .where(grid_data["compressibility"] > 0.0, 0)
        .where(grid_data["reservoir_thickness"] > 0.0, 0)
        .where(np.isfinite(grid_data["reservoir_thickness"]), 0)
        .where(grid_data["pressure_drop"].isel({"datetime": -1}) > 0.0, 0)
    )
    grid_support_polygon = tgeo.get_grid_support_polygon(grid_data["support"], support_buffer)
    polygons = xr.concat(
        [
            xr.DataArray(polygon_data["geometry"], dims="polygon").astype("object"),
            xr.DataArray(grid_support_polygon, coords={"polygon": "grid_support"}),
        ],
        dim="polygon",
    )
    return polygons


def prepare_fault_data(fault_data: xr.Dataset, grid_data: xr.Dataset) -> xr.Dataset:
    fault_grid_data = grid_data[["support", "reservoir_thickness"]]
    fault_data = fault_data.merge(
        fault_grid_data.interp(
            {
                "x": fault_data["x"],
                "y": fault_data["y"],
            },
            method="nearest",
        ).drop_vars(["x", "y"]),
        compat="override",
    )
    fault_filter = fault_data["support"] > 0.0
    fault_data = fault_data.sel(ID=fault_filter)

    extended_reservoir_thickness = xr.Dataset(
        {
            "grid": fault_data["reservoir_thickness"],
            "faults": fault_data["ThicknessTotal"],
        },
    ).to_array("thickness_origin")
    fault_data.update(
        xr.Dataset(
            {
                "reservoir_thickness": extended_reservoir_thickness,
            }
        )
    )

    fault_attributes = {
        "azimuth": np.fmod(fault_data["Azimuth"] + 90.0, 180.0),
        "throw_clipped": np.minimum(fault_data["Throw"], fault_data["reservoir_thickness"]),
        "throw_thickness_ratio": fault_data["Throw"] / fault_data["reservoir_thickness"],
        "fault_area": fault_data["reservoir_thickness"] * fault_data["StrikeLength"],
    }
    fault_data.update(xr.Dataset(fault_attributes))
    return fault_data


def map_faults_to_grid(
    grid_data: xr.Dataset,
    fault_data: xr.Dataset,
    parameters: xr.Dataset,
    bernstein_degrees: list[int],
) -> tuple[xr.Dataset, xr.Dataset]:
    step = xr.DataArray(
        [grid_data[d].diff(d).mean().values for d in ["x", "y"]],
        dims=["loc"],
    )

    fault_attributes: dict[str, xr.DataArray] = {}
    fault_attributes["weights_rmax"] = (
        fault_data["throw_thickness_ratio"] < (parameters["rmax"] / 2)
    ).astype(float)
    fault_attributes["fault_area_rmax"] = fault_data["fault_area"] * fault_attributes["weights_rmax"]
    fault_data.update(xr.Dataset(fault_attributes))

    throw_max_rmax = tgrid.aggregate_to_grid(
        fault_data[["x", "y"]],
        step,
        fault_data["weights_rmax"] * fault_data["Throw"],
        marginalize_dims=["ID"],
        operator=np.fmax,
        order=0,
    )
    throw_sum_rmax = tgrid.aggregate_to_grid(
        fault_data[["x", "y"]],
        step,
        fault_data["weights_rmax"] * fault_data["Throw"] * fault_data["StrikeLength"],
        marginalize_dims=["ID"],
        operator=np.add,
        order=0,
    )
    fault_length_rmax = tgrid.aggregate_to_grid(
        fault_data[["x", "y"]],
        step,
        fault_data["weights_rmax"] * fault_data["StrikeLength"],
        marginalize_dims=["ID"],
        operator=np.add,
        order=0,
    )
    fault_area_rmax = tgrid.aggregate_to_grid(
        fault_data[["x", "y"]],
        step,
        fault_data["weights_rmax"] * fault_data["fault_area"],
        marginalize_dims=["ID"],
        operator=np.add,
        order=0,
    )
    throw_average_rmax = throw_sum_rmax / fault_length_rmax
    grid_data["throw_max_rmax"] = throw_max_rmax
    grid_data["throw_average_rmax"] = throw_average_rmax
    grid_data["fault_length_rmax"] = fault_length_rmax
    grid_data["fault_area_rmax"] = fault_area_rmax

    w_bernstein = [
        bernstein.bernstein_partition(
            fault_data["throw_thickness_ratio"],
            degree=degree,
            fractiles=True,
            dim="ID",
        )
        for degree in bernstein_degrees
    ]
    fault_attributes = {
        "weights_bs": xr.concat(w_bernstein, dim="bernstein_basis"),
    }
    fault_attributes["fault_area_bs"] = fault_data["fault_area"] * fault_attributes["weights_bs"]
    fault_data.update(xr.Dataset(fault_attributes))

    throw_sum_bs = tgrid.aggregate_to_grid(
        samples=fault_data[["x", "y"]],
        target_step=step,
        weights=fault_attributes["weights_bs"] * fault_data["Throw"] * fault_data["StrikeLength"],
        marginalize_dims=["ID"],
        order=0,
    )
    fault_length_bs = tgrid.aggregate_to_grid(
        samples=fault_data[["x", "y"]],
        target_step=step,
        weights=fault_attributes["weights_bs"] * fault_data["StrikeLength"],
        marginalize_dims=["ID"],
        order=0,
    )
    fault_area_bs = tgrid.aggregate_to_grid(
        samples=fault_data[["x", "y"]],
        target_step=step,
        weights=fault_attributes["weights_bs"] * fault_data["fault_area"],
        marginalize_dims=["ID"],
        order=0,
    )
    throw_average_bs = throw_sum_bs / fault_length_bs
    grid_data.update(
        xr.Dataset(
            {
                "throw_average_bs": throw_average_bs,
                "fault_length_bs": fault_length_bs,
                "fault_area_bs": fault_area_bs,
            }
        )
    )

    fault_total_length = (
        fault_length_bs.groupby("bernstein_degree")
        .sum()
        .rename({"bernstein_degree": "bd"})
        .sel({"bd": fault_length_bs["bernstein_degree"]})
        .drop_vars("bd")
    )
    bernstein_fraction = fault_length_bs / fault_total_length
    grid_data.update(xr.Dataset({"bernstein_fraction": bernstein_fraction}))
    return grid_data, fault_data


def calculate_stress(
    grid_data: xr.Dataset,
    fault_data: xr.Dataset,
    parameters: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
    grid_data["pressure_drop_nondecreasing"] = np.clip(
        tgrid.xr_make_nondecreasing(grid_data["pressure_drop"], dim="datetime"),
        0.0,
        None,
    )
    grid_data["stress_linear"] = model.incremental_stress(
        pressure_drop=grid_data["pressure_drop_nondecreasing"],
        gradient=1.0,
        poisson_ratio=0.2,
        bulk_modulus=1.0 / grid_data["compressibility"],
        solid_modulus=10 ** parameters["hs_exp"],
    )
    grid_data["stress_linear_rmax"] = model.incremental_stress(
        pressure_drop=grid_data["pressure_drop_nondecreasing"],
        gradient=grid_data["throw_average_rmax"],
        poisson_ratio=0.2,
        bulk_modulus=1.0 / grid_data["compressibility"],
        solid_modulus=10 ** parameters["hs_exp"],
    )
    grid_data["stress_RTiCM"] = tgrid.xr_make_nondecreasing(
        grid_data["horizontal_stress_RTiCM"],
        dim="datetime",
    )
    grid_data["stress_RTiCM_rmax"] = grid_data["stress_RTiCM"] * grid_data["throw_average_rmax"]
    grid_data.update(
        xr.Dataset(
            {
                "stress_linear_bs": model.incremental_stress(
                    pressure_drop=grid_data["pressure_drop_nondecreasing"],
                    gradient=grid_data["throw_average_bs"],
                    poisson_ratio=0.2,
                    bulk_modulus=1.0 / grid_data["compressibility"],
                    solid_modulus=10 ** parameters["hs_exp"],
                ),
                "stress_RTiCM_bs": (
                    tgrid.xr_make_nondecreasing(
                        grid_data["horizontal_stress_RTiCM"],
                        dim="datetime",
                    )
                    * grid_data["throw_average_bs"]
                ),
            }
        )
    )
    return grid_data, fault_data


def incorporate_polygon_data(
    grid_data: xr.Dataset,
    event_data: xr.Dataset,
    polygons: xr.DataArray,
) -> tuple[xr.Dataset, xr.Dataset]:
    grid_spacing = np.mean([grid_data[d].diff(d).mean().item() for d in ["x", "y"]])
    grid_attributes = {}
    grid_attributes["overlap_fraction"] = tgeo.xr_cell_polygon_overlap_fraction(
        grid_data,
        polygons,
        0.5 * grid_spacing,
        "square",
    )
    grid_attributes["support_fraction"] = grid_attributes["overlap_fraction"] * grid_data["support"]
    grid_data.update(xr.Dataset(grid_attributes))
    event_data["polygon_distance"] = tgeo.xr_point_polygon_distance(event_data[["x", "y"]], polygons)
    return grid_data, event_data


def write_initial_outputs(
    source_data_root: Path,
    event_data: xr.Dataset,
    fault_data: xr.Dataset,
    grid_data_flat: xr.Dataset,
    polygons: xr.DataArray,
) -> dict[str, Path]:
    source_data_root.mkdir(parents=True, exist_ok=True)
    event_data_path = source_data_root / "event_data.h5"
    fault_data_path = source_data_root / "fault_data.h5"
    grid_data_path = source_data_root / "grid_data.h5"
    shapefile_path = source_data_root / "groningen_polygons.shp"

    for suffix in [".cpg", ".dbf", ".prj", ".shp", ".shx"]:
        path = shapefile_path.with_suffix(suffix)
        if path.exists():
            path.unlink()

    gpd.GeoDataFrame(polygons.to_dataframe(), crs="EPSG:28992").to_file(
        shapefile_path,
        driver="ESRI Shapefile",
    )
    event_data.to_netcdf(event_data_path, mode="w", engine="h5netcdf")
    fault_data.drop_vars("geometry", errors="ignore").reset_index(["ID", "bernstein_basis"]).to_netcdf(
        fault_data_path,
        mode="w",
        engine="h5netcdf",
    )
    grid_data_flat.reset_index(["loc", "bernstein_basis"]).to_netcdf(
        grid_data_path,
        mode="w",
        engine="h5netcdf",
    )
    return {
        "event_data": event_data_path,
        "fault_data": fault_data_path,
        "grid_data": grid_data_path,
        "shapefile": shapefile_path,
    }


def smooth_grid_data(
    grid_data_path: Path,
    flat_nodes: xr.DataArray,
    sigma_values: xr.DataArray,
) -> None:
    grid_data_vars = list(
        xr.load_dataset(
            grid_data_path,
            decode_coords="all",
            decode_timedelta=False,
            engine="h5netcdf",
        ).data_vars
    )

    fault_vars = [name for name in grid_data_vars if name.startswith("fault_")]
    for key in progress(fault_vars, desc="smooth fault vars", total=len(fault_vars)):
        data_flat = xr.load_dataset(
            grid_data_path,
            decode_coords="all",
            decode_timedelta=False,
            engine="h5netcdf",
        )[key]

        if "bernstein_degree" in data_flat.coords:
            data_flat = data_flat.set_xindex(["bernstein_degree", "bernstein_index"])

        data = data_flat.set_xindex(["x", "y"]).unstack("loc")
        smooth_data = (
            tgrid.xr_smooth(data, sigma_values).stack({"loc": ["x", "y"]}).sel({"loc": flat_nodes})
        )
        smooth_data.name = key + "_smooth"
        smooth_data.reset_index(
            [value for value in ["loc", "bernstein_basis"] if value in smooth_data.dims]
        ).to_netcdf(
            grid_data_path,
            mode="a",
            engine="h5netcdf",
        )

    stress_vars = [name for name in grid_data_vars if name.startswith("stress")]
    for key in progress(stress_vars, desc="smooth stress vars", total=len(stress_vars)):
        dataset = xr.load_dataset(
            grid_data_path,
            decode_coords="all",
            decode_timedelta=False,
            engine="h5netcdf",
        )
        data_flat = dataset[key]
        if "bs" in key and "stress" in key:
            bernstein_fraction_flat = dataset["bernstein_fraction"]
        else:
            bernstein_fraction_flat = None

        if "bernstein_degree" in data_flat.coords:
            data_flat = data_flat.set_xindex(["bernstein_degree", "bernstein_index"])
        data = data_flat.set_xindex(["x", "y"]).unstack("loc")

        if bernstein_fraction_flat is not None:
            if "bernstein_degree" in bernstein_fraction_flat.coords:
                bernstein_fraction_flat = bernstein_fraction_flat.set_xindex(["bernstein_degree", "bernstein_index"])
            bernstein_fraction = bernstein_fraction_flat.set_xindex(["x", "y"]).unstack("loc")
            data = data * bernstein_fraction

        smooth_data = (
            tgrid.xr_smooth(
                data,
                sigma_values,
                ignore_nans=False,
            )
            .stack({"loc": ["x", "y"]})
            .sel({"loc": flat_nodes})
        )
        smooth_data.name = key + "_smooth"
        smooth_data.reset_index(
            [value for value in ["loc", "bernstein_basis"] if value in smooth_data.dims]
        ).to_netcdf(
            grid_data_path,
            mode="a",
            engine="h5netcdf",
        )


def generate_source_data(
    resources_root: Path,
    source_data_root: Path,
    config: dict[str, object],
) -> dict[str, object]:
    catalogue_end = resolve_catalogue_end(config)
    if catalogue_end is not None:
        LOGGER.info("stage=source-data catalogue-end=%s", catalogue_end.isoformat())

    polygon_data, fault_data, event_data, grid_data = load_raw_inputs(resources_root, catalogue_end)
    parameters = build_parameters(grid_data, config)
    polygons = prepare_polygons(
        polygon_data,
        grid_data,
        float(config.get("support_buffer", 500.0)),
    )
    fault_data = prepare_fault_data(fault_data, grid_data)
    grid_data, fault_data = map_faults_to_grid(
        grid_data,
        fault_data,
        parameters,
        [int(value) for value in config.get("bernstein_degrees", [4])],
    )
    grid_data, fault_data = calculate_stress(grid_data, fault_data, parameters)
    grid_data, event_data = incorporate_polygon_data(grid_data, event_data, polygons)

    grid_data_flat = grid_data.stack({"loc": ["x", "y"]})
    grid_data_flat = grid_data_flat.sel({"loc": grid_data_flat["support"] > 0})
    flat_nodes = grid_data_flat["loc"]
    paths = write_initial_outputs(source_data_root, event_data, fault_data, grid_data_flat, polygons)
    smooth_grid_data(paths["grid_data"], flat_nodes, parameters["sigma"])
    return {"paths": {key: str(value) for key, value in paths.items()}}