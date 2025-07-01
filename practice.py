import pandas as pd
import numpy as np
from pathlib import Path
import netCDF4 as nc

DATA_DIR = Path.cwd() / "data/"
STANDARD_DEPTHS = np.array(
    [
        0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
    ]
)


def concat_data_array(param_name: str, sdate: str, edate: str) -> np.ndarray:
    from datetime import datetime
    data_list = []
    sdate_dt = datetime.strptime(sdate, "%Y-%m-%d")
    edate_dt = datetime.strptime(edate, "%Y-%m-%d")
    start_year = sdate_dt.year
    end_year = edate_dt.year
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            f = DATA_DIR / f"sooList_{year}{month:02d}.json"
            try:
                if f.is_file():
                    df = pd.read_json(f)
                    if param_name in df.columns:
                        arr = df[param_name].replace(["", None], np.nan)
                        data_list.append(arr.values)
            except Exception as e:
                print(f"Error reading {f.name}: {e}")
    if data_list:
        return np.concatenate(data_list)
    return np.array([])


def build_dataframe(sdate: str, edate: str) -> pd.DataFrame:
    temperature = concat_data_array("wtr_tmp", sdate, edate)
    depth = concat_data_array("wtr_dep", sdate, edate)
    obs_time = concat_data_array("obs_dtm", sdate, edate)
    longitude = concat_data_array("lon", sdate, edate)
    latitude = concat_data_array("lat", sdate, edate)
    salinity = concat_data_array("sal", sdate, edate)
    dissolved_oxygen = concat_data_array("dox", sdate, edate)

    df = pd.DataFrame(
        {
            "datetime": obs_time,
            "longitude": longitude,
            "latitude": latitude,
            "depth": depth,
            "temperature": temperature,
            "salinity": salinity,
            "dissolved_oxygen": dissolved_oxygen,
        }
    )

    # --- Standard Depth Matching ---
    df = df[~np.isnan(df["depth"])]
    df["std_depth"] = STANDARD_DEPTHS[
        np.abs(df["depth"].values[:, None] - STANDARD_DEPTHS).argmin(axis=1)
    ]

    # --- 3-sigma Quality Control for salinity and dissolved oxygen ---
    for var in ["salinity", "dissolved_oxygen"]:
        vals = df[var]
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        mask = (vals >= mean - 3 * std) & (vals <= mean + 3 * std)
        df.loc[~mask, var] = np.nan

    return df


def save_sparse_to_netcdf_spatiotemporal(
    df: pd.DataFrame, filename: str = "sooList1968010120241231.nc"
) -> None:
    t_vals = np.sort(pd.to_datetime(df["datetime"].unique()))
    x_vals = np.sort(df["longitude"].unique())
    y_vals = np.sort(df["latitude"].unique())
    z_vals = np.sort(df["std_depth"].unique())

    t_index = {v: i for i, v in enumerate(t_vals)}
    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}
    z_index = {v: i for i, v in enumerate(z_vals)}

    t_idx, x_idx, y_idx, z_idx = [], [], [], []
    wtr_tmp_data, sal_data, dox_data = [], [], []

    for _, row in df.iterrows():
        if (
            pd.isna(row["datetime"])
            or np.isnan(row["longitude"])
            or np.isnan(row["latitude"])
            or np.isnan(row["std_depth"])
        ):
            continue
        if not (
            np.isnan(row["temperature"])
            and np.isnan(row["salinity"])
            and np.isnan(row["dissolved_oxygen"])
        ):
            t_idx.append(t_index[pd.to_datetime(row["datetime"])])
            x_idx.append(x_index[row["longitude"]])
            y_idx.append(y_index[row["latitude"]])
            z_idx.append(z_index[row["std_depth"]])
            wtr_tmp_data.append(np.float32(row["temperature"]))
            sal_data.append(np.float32(row["salinity"]))
            dox_data.append(np.float32(row["dissolved_oxygen"]))

    n_obs = len(t_idx)

    with nc.Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension("n_obs", n_obs)
        ds.createDimension("t", len(t_vals))
        ds.createDimension("x", len(x_vals))
        ds.createDimension("y", len(y_vals))
        ds.createDimension("z", len(z_vals))

        # Coordinate variables
        time_var = ds.createVariable("t", "str", ("t",))
        time_var[:] = np.array([str(dt) for dt in t_vals])
        ds.createVariable("x", "f4", ("x",))[:] = x_vals
        ds.createVariable("y", "f4", ("y",))[:] = y_vals
        ds.createVariable("z", "f4", ("z",))[:] = z_vals

        # Index variables
        ds.createVariable("t_idx", "i4", ("n_obs",))[:] = np.array(t_idx, dtype=np.int32)
        ds.createVariable("x_idx", "i4", ("n_obs",))[:] = np.array(x_idx, dtype=np.int32)
        ds.createVariable("y_idx", "i4", ("n_obs",))[:] = np.array(y_idx, dtype=np.int32)
        ds.createVariable("z_idx", "i4", ("n_obs",))[:] = np.array(z_idx, dtype=np.int32)

        # Sparse data variables as float32
        ds.createVariable("wtr_tmp", "f4", ("n_obs",), fill_value=np.nan)[:] = np.array(
            wtr_tmp_data, dtype=np.float32
        )
        ds.createVariable("sal", "f4", ("n_obs",), fill_value=np.nan)[:] = np.array(
            sal_data, dtype=np.float32
        )
        ds.createVariable("dox", "f4", ("n_obs",), fill_value=np.nan)[:] = np.array(
            dox_data, dtype=np.float32
        )

        ds.title = "Sparse Spatiotemporal Ocean Data"
        ds.history = "Created by script (sparse spatiotemporal representation)"


def save_dense_to_netcdf_4d(
    df: pd.DataFrame, filename: str = "sooList_dense_4d.nc"
) -> None:
    """
    Save data as dense 4D (time, depth, lat, lon) NetCDF4 file for use with Iris.
    """
    # Prepare unique sorted axes
    t_vals = np.sort(pd.to_datetime(df["datetime"].unique()))
    z_vals = np.sort(df["std_depth"].unique())
    y_vals = np.sort(df["latitude"].unique())
    x_vals = np.sort(df["longitude"].unique())

    # Build index maps
    t_index = {v: i for i, v in enumerate(t_vals)}
    z_index = {v: i for i, v in enumerate(z_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}
    x_index = {v: i for i, v in enumerate(x_vals)}

    # Initialize arrays with NaN
    shape = (len(t_vals), len(z_vals), len(y_vals), len(x_vals))
    temp_arr = np.full(shape, np.nan, dtype=np.float32)
    sal_arr = np.full(shape, np.nan, dtype=np.float32)
    dox_arr = np.full(shape, np.nan, dtype=np.float32)

    # Fill arrays
    for _, row in df.iterrows():
        if (
            pd.isna(row["datetime"])
            or np.isnan(row["longitude"])
            or np.isnan(row["latitude"])
            or np.isnan(row["std_depth"])
        ):
            continue
        t = t_index[pd.to_datetime(row["datetime"])]
        z = z_index[row["std_depth"]]
        y = y_index[row["latitude"]]
        x = x_index[row["longitude"]]
        if not np.isnan(row["temperature"]):
            temp_arr[t, z, y, x] = row["temperature"]
        if not np.isnan(row["salinity"]):
            sal_arr[t, z, y, x] = row["salinity"]
        if not np.isnan(row["dissolved_oxygen"]):
            dox_arr[t, z, y, x] = row["dissolved_oxygen"]

    # Write to NetCDF4
    with nc.Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(t_vals))
        ds.createDimension("depth", len(z_vals))
        ds.createDimension("lat", len(y_vals))
        ds.createDimension("lon", len(x_vals))

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var[:] = np.array([pd.Timestamp(t).timestamp() for t in t_vals])

        ds.createVariable("depth", "f4", ("depth",))[:] = z_vals
        ds.createVariable("lat", "f4", ("lat",))[:] = y_vals
        ds.createVariable("lon", "f4", ("lon",))[:] = x_vals

        ds.createVariable("temperature", "f4", ("time", "depth", "lat", "lon"), fill_value=np.nan)[:] = temp_arr
        ds.createVariable("salinity", "f4", ("time", "depth", "lat", "lon"), fill_value=np.nan)[:] = sal_arr
        ds.createVariable("dissolved_oxygen", "f4", ("time", "depth", "lat", "lon"), fill_value=np.nan)[:] = dox_arr

        ds.title = "Dense Spatiotemporal Ocean Data"
        ds.history = "Created by script (dense 4D representation)"

if __name__ == "__main__":
    import argparse
    sdate = "1968-01-01"
    edate = "2025-12-31"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true", help="Save as dense 4D NetCDF (Iris compatible)")
    args = parser.parse_args()
    df = build_dataframe(sdate, edate)
    if args.dense:
        save_dense_to_netcdf_4d(df, filename="sooList_dense_4d.nc")
    else:
        save_sparse_to_netcdf_spatiotemporal(df, filename="sooList1968010120241231.nc")
