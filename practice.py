import pandas as pd
import numpy as np
from pathlib import Path
import netCDF4 as nc

DATA_DIR = Path.cwd() / "data/"


def concat_data_array(param_name: str, sdate: str, edate: str):
    """Concatenate a parameter from multiple JSON files into a single NumPy array with NaN for missing values."""
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
                        # Replace empty strings and None with np.nan for missing values
                        arr = df[param_name].replace(["", None], np.nan)
                        data_list.append(arr.values)
            except Exception as e:
                print(f"Error reading {f.name}: {e}")

    if data_list:
        return np.concatenate(data_list)
    else:
        return np.array([])


def datetime_to_julian(dt_array):
    """Convert numpy array of datetime64 or pandas Timestamps to Julian date (float)."""
    dt_index = pd.to_datetime(dt_array, errors="coerce")
    return dt_index.to_julian_date().values


def build_dataframe(sdate, edate):
    """Build the main DataFrame from all arrays."""
    temperature = concat_data_array("wtr_tmp", sdate, edate)
    depth = concat_data_array("wtr_dep", sdate, edate)
    obs_time = concat_data_array("obs_dtm", sdate, edate)
    longitude = concat_data_array("lon", sdate, edate)
    latitude = concat_data_array("lat", sdate, edate)
    salinity = concat_data_array("sal", sdate, edate)
    dissolved_oxygen = concat_data_array("dox", sdate, edate)
    julian_time = datetime_to_julian(obs_time)

    df = pd.DataFrame(
        {
            "julian_time": julian_time,
            "longitude": longitude,
            "latitude": latitude,
            "depth": depth,
            "temperature": temperature,
            "salinity": salinity,
            "dissolved_oxygen": dissolved_oxygen,
        }
    )
    return df


def save_sparse_to_netcdf_spatiotemporal(df, filename="sooList1968010120241231.nc"):
    """
    Save original (non-aggregated) data as sparse arrays to NetCDF4 file.
    All variables are saved as a single-precision (float32) array for memory efficiency.
    Dimensions: spatiotemporal (t, z, y, x) with coordinates.
    Variables: wtr_tmp (temperature), sal (salinity), dox (dissolved oxygen).
    """
    # Get unique sorted values for each dimension
    t_vals = np.sort(df["julian_time"].unique())
    x_vals = np.sort(df["longitude"].unique())
    y_vals = np.sort(df["latitude"].unique())
    z_vals = np.sort(df["depth"].unique())

    # Create indexers for fast lookup
    t_index = {v: i for i, v in enumerate(t_vals)}
    x_index = {v: i for i, v in enumerate(x_vals)}
    y_index = {v: i for i, v in enumerate(y_vals)}
    z_index = {v: i for i, v in enumerate(z_vals)}

    # Prepare lists for sparse representation
    t_idx, x_idx, y_idx, z_idx = [], [], [], []
    wtr_tmp_data, sal_data, dox_data = [], [], []

    for _, row in df.iterrows():
        # Only store rows where at least one variable is not NaN
        if not (
            np.isnan(row["temperature"])
            and np.isnan(row["salinity"])
            and np.isnan(row["dissolved_oxygen"])
        ):
            t_idx.append(t_index[row["julian_time"]])
            x_idx.append(x_index[row["longitude"]])
            y_idx.append(y_index[row["latitude"]])
            z_idx.append(z_index[row["depth"]])
            # Save as float32 for memory efficiency
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
        ds.createVariable("t", "f8", ("t",))[:] = t_vals
        ds.createVariable("x", "f4", ("x",))[:] = x_vals
        ds.createVariable("y", "f4", ("y",))[:] = y_vals
        ds.createVariable("z", "f4", ("z",))[:] = z_vals

        # Index variables
        ds.createVariable("t_idx", "i4", ("n_obs",))[:] = np.array(t_idx, dtype=np.int32)
        ds.createVariable("x_idx", "i4", ("n_obs",))[:] = np.array(x_idx, dtype=np.int32)
        ds.createVariable("y_idx", "i4", ("n_obs",))[:] = np.array(y_idx, dtype=np.int32)
        ds.createVariable("z_idx", "i4", ("n_obs",))[:] = np.array(z_idx, dtype=np.int32)

        # Sparse data variables as float32
        ds.createVariable("wtr_tmp", "f4", ("n_obs",), fill_value=np.nan)[:] = np.array(wtr_tmp_data, dtype=np.float32)
        ds.createVariable("sal", "f4", ("n_obs",), fill_value=np.nan)[:] = np.array(sal_data, dtype=np.float32)
        ds.createVariable("dox", "f4", ("n_obs",), fill_value=np.nan)[:] = np.array(dox_data, dtype=np.float32)

        ds.title = "Sparse Spatiotemporal Ocean Data"
        ds.history = "Created by script (sparse spatiotemporal representation)"


# Example usage:
if __name__ == "__main__":
    sdate = "1968-01-01"
    edate = "2025-12-31"
    df = build_dataframe(sdate, edate)
    save_sparse_to_netcdf_spatiotemporal(df, filename="sooList1968010120241231.nc")
