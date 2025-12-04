import h5py  # noqa: F401
import rioxarray  # noqa: F401
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import xarray as xr
from pathlib import Path
import sqlite3
import requests
import io
from functools import lru_cache

SECONDS_PER_SIDEREAL_YEAR = 31_558_149.54
SECONDS_PER_DAY = 86_400

outline_file = Path("Groningen_field_outline.csv")
faults_file = Path("Faultdata_NAM_reformatted_cleaned_selected.sqlite3")
comp2021_file = Path("Cm_grids_rotliechend.csv")
pressure_file = Path("XY_PRF_ROTL_GY_V2a_2023.csv")
comp2018_file = Path("ReservoirModel_compressibility_20180122.csv")
thickness_file = Path("ReservoirModel_thickness_20171013.csv")
rticm_stress_file = Path("rticm_stress.h5")


def get_polygon_gdf(path):
    groningen_outline = pd.read_csv(path / outline_file)
    groningen_polygon = Polygon(zip(groningen_outline.x, groningen_outline.y))
    groningen_polygon_data = {
        "polygon": [
            "GroningenFieldGWC",
            "GroningenFieldGWC_b5000",
        ],
        "geometry": [
            groningen_polygon,
            groningen_polygon.buffer(5000),
        ],
    }
    groningen_polygon_gdf = gpd.GeoDataFrame(
        data=groningen_polygon_data,
        geometry="geometry",
        crs=28992,
    ).set_index("polygon")

    return groningen_polygon_gdf


@lru_cache  # do not bother KNMI with repeated calls
def get_catalogue_gdf(**pars):
    def_pars = {
        "minlatitude": 53.00,
        "maxlatitude": 53.55,
        "minlongitude": 6.45,
        "maxlongitude": 7.15,
        "format": "csv",
    }
    params = def_pars | pars
    request = requests.get("http://rdsa.knmi.nl/fdsnws/event/1/query", params=params)
    eq_df = (
        pd.read_csv(
            io.StringIO(request.text),
            header=None,
            names=[
                "knmi_id",
                "datetime",
                "lat",
                "lon",
                "depth",
                "magnitude",
                "community",
            ],
        )
        .rename({"knmi_id": "event_id"}, axis="columns")
        .set_index("event_id")
    )
    eq_df["datetime"] = pd.to_datetime(eq_df["datetime"])

    rd_locs = gpd.points_from_xy(eq_df["lon"], eq_df["lat"], crs=4326).to_crs(28992)
    eq_df["x"] = rd_locs.x
    eq_df["y"] = rd_locs.y

    eq_gdf = gpd.GeoDataFrame(eq_df, geometry=rd_locs)

    return eq_gdf


def get_catalogue(**pars):
    eq_gdf = get_catalogue_gdf(**pars).dropna().drop(columns=["geometry"])
    eq_xr = xr.Dataset(eq_gdf)
    return eq_xr


def get_grid(path):
    pressure_pd = (
        pd.read_csv(path / pressure_file)
        .rename(columns={"X": "x", "Y": "y"})
        .set_index(["x", "y"])
    )
    dates = pd.to_datetime(
        np.char.lstrip(
            np.char.translate(
                pressure_pd.columns.values.astype("str"), str.maketrans("_", "-")
            ),
            "P",
        )
    )
    pressure_pd = pressure_pd.set_axis(dates, axis="columns")
    pressure_xr = xr.DataArray(pressure_pd).rename("pressure")
    pressure_xr = pressure_xr.unstack().rename({"dim_1": "datetime"})
    pressure_xr = pressure_xr / 10.0  # bar to MPa

    comp2018_pd = pd.read_csv(
        path / comp2018_file,
        header=0,
        names=["x", "y", "compressibility"],
    ).set_index(["x", "y"])

    thick_pd = pd.read_csv(
        path / thickness_file,
        header=0,
        names=["x", "y", "reservoir_thickness"],
    ).set_index(["x", "y"])
    thick_pd = thick_pd[thick_pd["reservoir_thickness"] > 1.0]

    comp2021_pd = pd.read_csv(
        path / comp2021_file,
        header=0,
        names=["x", "y", "compressibility"],
    ).set_index(["x", "y"])

    comp2018_xr = xr.Dataset(comp2018_pd).unstack()
    comp2018_xr = comp2018_xr * 10.0  # convert from 1/bar to MPa^-1

    comp2021_xr = xr.Dataset(comp2021_pd).unstack().interp_like(comp2018_xr)

    comp_xr = (
        xr.Dataset(
            {
                "cm_NAM2018": comp2018_xr["compressibility"],
                "cm_NAM2021": comp2021_xr["compressibility"],
            }
        )
        .to_array("cm_grid_version")
        .rename("compressibility")
    )

    thick_xr = xr.Dataset(thick_pd).unstack()

    covdat = xr.merge(
        [pressure_xr, comp_xr, thick_xr],
        combine_attrs="drop",
        join="outer",
    )

    covdat["pressure_drop"] = -(
        covdat["pressure"] - covdat["pressure"].isel(datetime=0)
    )

    rticm_stress_ds = xr.load_dataset(path / rticm_stress_file, engine="h5netcdf")
    rticm_stress_ds = (
        rticm_stress_ds.rename({"time": "datetime"}).squeeze().reset_coords()
    )
    covdat["horizontal_stress_RTiCM"] = rticm_stress_ds["horizontal_stress"]

    # dx, dy on grid points
    dx, dy = [
        (
            covdat[xy].diff(dim=xy, label="upper")
            + covdat[xy].diff(dim=xy, label="lower")
        )
        / 2
        for xy in ["x", "y"]
    ]
    dt_seconds = (
        covdat["datetime"]
        .diff(dim="datetime", label="lower")
        .dt.total_seconds()
        .astype("float64")
    )

    covdat["measure_xy"] = dx * dy
    covdat["measure_t"] = dt_seconds / SECONDS_PER_DAY
    covdat["measure_xyt"] = covdat["measure_xy"] * covdat["measure_t"]
    covdat["measure_xyz"] = covdat["measure_xy"] * covdat["reservoir_thickness"]
    covdat["measure_xyzt"] = covdat["measure_xyt"] * covdat["reservoir_thickness"]

    covdat["pressure"].attrs["units"] = "MPa"
    covdat["horizontal_stress_RTiCM"].attrs["units"] = "MPa"
    covdat["reservoir_thickness"].attrs["units"] = "m"
    covdat["compressibility"].attrs["units"] = "MPa^-1"
    covdat["measure_xy"].attrs["units"] = "m^2"
    covdat["measure_t"].attrs["units"] = "days"
    covdat["measure_xyt"].attrs["units"] = "m^2 days"
    covdat["measure_xyz"].attrs["units"] = "m^3"
    covdat["measure_xyzt"].attrs["units"] = "m^3 days"

    covdat.rio.write_crs(28992, inplace=True)

    return covdat


def get_faults_gdf(path):
    fxr = get_faults(path)

    return gpd.GeoDataFrame(fxr.to_dataframe(), crs=28992).drop(
        ["FAULT_ID", "PILLAR_ID"], axis=1
    )


@lru_cache  # cache the result
def get_faults(path):
    cnx = sqlite3.connect(path / faults_file)
    fault_data = pd.read_sql_query("SELECT * FROM pillar_geom", cnx)
    cnx.commit()
    cnx.close()

    # construct geopandas dataframe from the fault data
    geometry = gpd.points_from_xy(fault_data["X"], fault_data["Y"])
    pillar_gpd = gpd.GeoDataFrame(fault_data, geometry=geometry, crs=28992).set_index(
        ["FAULT_ID", "PILLAR_ID"]
    )

    # to determine strike length per pillar we will conveniently use xarray
    fp_xr = xr.DataArray(pillar_gpd, dims=("ID", "properties"))

    # unstack the ID index to get a 2D array of fault and pillar IDs
    # this will create a ragged array, completed with NaNs
    fp_unstacked = fp_xr.sel({"properties": ["X", "Y"]}).astype("float64").unstack("ID")

    # determine distance to the left and right of each pillar
    # then align the two arrays to the same index
    lo, up = xr.align(
        (fp_unstacked.diff("PILLAR_ID", label="lower") ** 2).sum("properties") ** (0.5),
        (fp_unstacked.diff("PILLAR_ID", label="upper") ** 2).sum("properties") ** (0.5),
        join="outer",
        fill_value=0.0,
    )

    # determine the strike length of each pillar as the average of the distance to the left and right
    strike_length = 0.5 * (lo + up)

    # compute the thickness of each pillar as the average of the thickness of the hanging
    # and the foot wall
    thickness = 0.5 * fp_xr.sel(
        {"properties": ["Thickness1_f", "Thickness_f", "Thickness1_h", "Thickness_h"]}
    ).astype("float64").sum("properties")

    # store geometry as a separate variable
    geometry = fp_xr.sel({"properties": "geometry"}, drop=True)

    # align the strike length with the original but cleaned-up array
    fp, sl = xr.align(
        # select, clean up (drop the geometry column and convert to float),
        # and convert to dataset
        fp_xr.drop_sel({"properties": ["BIN", "Thickness", "Selected", "geometry"]})
        .astype("float64")
        .to_dataset("properties"),
        # stack the pillars to get a 1D array of the strike length of each pillar
        strike_length.stack({"ID": ["FAULT_ID", "PILLAR_ID"]}),
        # distance_xr,
        join="left",
    )

    # add the geometety, strike length and thickness to the dataset
    fp["geometry"] = geometry
    fp["Throw"] = fp["Offset"]
    fp["ThicknessTotal"] = thickness
    fp["StrikeLength"] = sl
    fp["FaultArea"] = fp["StrikeLength"] * fp["ThicknessTotal"]
    fp["ThrowThicknessRatio"] = fp["Throw"] / fp["ThicknessTotal"]

    fp = fp.rename({"X": "x", "Y": "y", "Z": "z"}).rio.write_crs(28992)

    return fp
