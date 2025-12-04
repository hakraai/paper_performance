import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import rioxarray  # noqa: F401
import xarray as xr
import h5py  # noqa: F401
from pathlib import Path
import platform
import sqlite3
import requests
import io
from functools import lru_cache

SECONDS_PER_SIDEREAL_YEAR = 31_558_149.54
SECONDS_PER_DAY = 86_400

if platform.system() == "Windows":
    prepath = Path("C://")
elif platform.node() == "app-mchain01":
    prepath = Path("/data/scratch_dk")
else:
    prepath = Path("/mnt/c")

path = prepath / Path("Cache/Modelketen")

outline_file = path / "Groningen_field_outline.csv"
tenboer_file = path / "Thickness_ten_boer.csv"
vshale_file = path / "Vshale.csv"
salt_file = path / "Gron_hal_thick_RD.csv"
polygon_file = path / "polygons.zip"
postcode_file = path / "CBS-PC4-2020-v1.zip"
zonation_file = path / "Geological_zones_V6.zip"
faults_file = path / "Faultdata_NAM_reformatted_cleaned_selected.sqlite3"
cm_file = path / "Cm_grids_rotliechend.csv"
edb_file = path / "EDBV7.1 PostP+Wierden_Extract.csv"
pressure_file = path / "XY_PRF_ROTL_GY_V2a_2023.csv"
tg_r_file = path / "ReservoirModel_topograds_beta2_3000.0_beta3_0.41.csv"
tg_m_file = path / "ReservoirModel_topograds_beta2_3500.0_beta3_1.1.csv"
comp_file = path / "ReservoirModel_compressibility_20180122.csv"
thickness_file = path / "ReservoirModel_thickness_20171013.csv"
rticm_stress_file = path / "rticm_stress.h5"


def get_polygon_gdf():
    groningen_outline = pd.read_csv(outline_file)
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
        data=groningen_polygon_data, geometry="geometry", crs=28992
    ).set_index("polygon")

    return groningen_polygon_gdf


def get_weits_polygons_gdf():
    return gpd.read_file(polygon_file)


def get_postcode_gdf():
    return gpd.read_file(postcode_file)


def get_zones_gdf():
    return gpd.read_file(zonation_file)


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


def get_grid():
    pressure_pd = (
        pd.read_csv(pressure_file)
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

    tg_m_pd = pd.read_csv(
        tg_m_file, header=0, names=["x", "y", "tg_nam_mag"]
    ).set_index(["x", "y"])
    tg_r_pd = pd.read_csv(
        tg_r_file, header=0, names=["x", "y", "tg_nam_rate"]
    ).set_index(["x", "y"])
    comp_pd = pd.read_csv(
        comp_file, header=0, names=["x", "y", "compressibility"]
    ).set_index(["x", "y"])
    comp_pd = comp_pd[comp_pd["compressibility"] > 0.0]
    thick_pd = pd.read_csv(
        thickness_file, header=0, names=["x", "y", "reservoir_thickness"]
    ).set_index(["x", "y"])
    thick_pd = thick_pd[thick_pd["reservoir_thickness"] > 1.0]

    tg_m_xr = xr.Dataset(tg_m_pd).unstack().rename({"tg_nam_mag": "tg_NAM_mg"})
    tg_r_xr = xr.Dataset(tg_r_pd).unstack().rename({"tg_nam_rate": "tg_NAM_ar"})
    comp_xr = xr.Dataset(comp_pd).unstack()
    thick_xr = xr.Dataset(thick_pd).unstack()

    covdat = xr.merge(
        [pressure_xr, tg_m_xr, tg_r_xr, comp_xr, thick_xr],
        combine_attrs="drop",
        join="outer",
    )

    covdat["pressure_drop"] = -(
        covdat["pressure"] - covdat["pressure"].isel(datetime=0)
    )

    # dx, dy on grid points
    dx, dy = [
        (
            covdat[xy].diff(dim=xy, label="upper")
            + covdat[xy].diff(dim=xy, label="lower")
        )
        / 2
        for xy in ["x", "y"]
    ]
    covdat["measure_xy"] = dx * dy
    dt_seconds = (
        covdat["datetime"]
        .diff(dim="datetime", label="lower")
        .dt.total_seconds()
        .astype("float64")
    )
    covdat["measure_t"] = dt_seconds / SECONDS_PER_DAY
    covdat["measure_xyt"] = covdat["measure_xy"] * covdat["measure_t"]
    covdat["measure_xyz"] = covdat["measure_xy"] * covdat["reservoir_thickness"]
    covdat["measure_xytz"] = covdat["measure_xyt"] * covdat["reservoir_thickness"]

    tenboer_pd = pd.read_csv(
        tenboer_file, header=0, names=["x", "y", "tenboer_thickness"]
    ).set_index(["x", "y"])
    vshale_pd = pd.read_csv(
        vshale_file, header=0, names=["x", "y", "clay_fraction"]
    ).set_index(["x", "y"])
    salt_pd = pd.read_csv(
        salt_file, header=0, usecols=[2, 3, 4], names=["x", "y", "salt_thickness"]
    ).set_index(["x", "y"])

    cm_pd = pd.read_csv(
        cm_file, header=0, names=["x", "y", "compaction_coefficient"]
    ).set_index(["x", "y"])

    tenboer_xr = xr.Dataset(tenboer_pd).unstack().interp_like(covdat)
    vshale_xr = xr.Dataset(vshale_pd).unstack().interp_like(covdat)
    salt_xr = xr.Dataset(salt_pd).unstack().interp_like(covdat)

    cm_xr = xr.Dataset(cm_pd).unstack().interp_like(covdat)

    covdat = xr.merge([covdat, tenboer_xr, vshale_xr, salt_xr, cm_xr])
    covdat["clay_thickness"] = covdat["reservoir_thickness"] * covdat["clay_fraction"]

    rticm_stress_ds = xr.load_dataset(rticm_stress_file)
    rticm_stress_ds = (
        rticm_stress_ds.rename({"time": "datetime"}).squeeze().reset_coords()
    )
    covdat["horizontal_stress_RTiCM"] = rticm_stress_ds["horizontal_stress"]

    covdat["pressure"].attrs["units"] = "MPa"
    covdat["horizontal_stress_RTiCM"].attrs["units"] = "MPa"
    covdat["reservoir_thickness"].attrs["units"] = "m"
    covdat["compressibility"].attrs["units"] = "-"
    covdat["compaction_coefficient"].attrs["units"] = "-"
    covdat["clay_fraction"].attrs["units"] = "-"
    covdat["measure_xy"].attrs["units"] = "m^2"
    covdat["measure_t"].attrs["units"] = "days"
    covdat["measure_xyt"].attrs["units"] = "m^2 days"
    covdat["measure_xyz"].attrs["units"] = "m^3"
    covdat["measure_xytz"].attrs["units"] = "m^3 days"
    covdat["clay_thickness"].attrs["units"] = "m"
    covdat["tenboer_thickness"].attrs["units"] = "m"

    covdat.rio.write_crs(28992, inplace=True)

    return covdat


def get_faults_gdf():
    fxr = get_faults()

    return gpd.GeoDataFrame(fxr.to_dataframe(), crs=28992).drop(
        ["FAULT_ID", "PILLAR_ID"], axis=1
    )


@lru_cache  # cache the result
def get_faults():
    cnx = sqlite3.connect(faults_file)
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
