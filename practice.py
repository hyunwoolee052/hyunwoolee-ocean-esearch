import pandas as pd
import numpy as np
from pathlib import Path
import netCDF4 as nc
import time
import tracemalloc
from scipy import sparse

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


# Use numpy arrays for memory efficiency
sdate = "1968-01-01"
edate = "2025-12-31"
temperature = concat_data_array("wtr_tmp", sdate, edate)
depth = concat_data_array("wtr_dep", sdate, edate)
obs_time = concat_data_array("obs_dtm", sdate, edate)
longitude = concat_data_array("lon", sdate, edate)
latitude = concat_data_array("lat", sdate, edate)
salinity = concat_data_array("sal", sdate, edate)
dissolved_oxygen = concat_data_array("dox", sdate, edate)

julian_time = datetime_to_julian(obs_time)

# Combine all arrays into a DataFrame for easy grouping
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

# Group by julian_time, longitude, latitude, and depth, then aggregate (mean, ignoring NaN)
grouped = (
    df.groupby(["julian_time", "longitude", "latitude", "depth"], dropna=True).reset_index()
)


def save_sparse_to_netcdf(grouped_df, filename="grouped_ocean_data_sparse.nc"):
    """
    Save grouped data as sparse arrays to NetCDF4 file.
    Only non-NaN values are stored with their indices.
    """
    # Get unique sorted values for each dimension
    times = np.sort(grouped_df["julian_time"].unique())
    lons = np.sort(grouped_df["longitude"].unique())
    lats = np.sort(grouped_df["latitude"].unique())
    depths = np.sort(grouped_df["depth"].unique())

    # Create indexers for fast lookup
    time_index = {v: i for i, v in enumerate(times)}
    lon_index = {v: i for i, v in enumerate(lons)}
    lat_index = {v: i for i, v in enumerate(lats)}
    depth_index = {v: i for i, v in enumerate(depths)}

    # Prepare lists for sparse representation
    t_idx, d_idx, la_idx, lo_idx = [], [], [], []
    temp_data, sal_data, dox_data = [], [], []

    for _, row in grouped_df.iterrows():
        t = time_index[row["julian_time"]]
        d = depth_index[row["depth"]]
        la = lat_index[row["latitude"]]
        lo = lon_index[row["longitude"]]
        # Only store non-NaN values
        if not np.isnan(row["temperature"]):
            t_idx.append(t)
            d_idx.append(d)
            la_idx.append(la)
            lo_idx.append(lo)
            temp_data.append(row["temperature"])
        if not np.isnan(row["salinity"]):
            # Use same indices for salinity
            pass if not np.isnan(row["temperature"]) else (t_idx.append(t), d_idx.append(d), la_idx.append(la), lo_idx.append(lo))
            sal_data.append(row["salinity"])
        if not np.isnan(row["dissolved_oxygen"]):
            # Use same indices for dox
            pass if not np.isnan(row["temperature"]) or not np.isnan(row["salinity"]) else (t_idx.append(t), d_idx.append(d), la_idx.append(la), lo_idx.append(lo))
            dox_data.append(row["dissolved_oxygen"])

    # Write to NetCDF (store as 1D arrays with indices)
    with nc.Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension("n_obs", len(t_idx))
        ds.createDimension("time", len(times))
        ds.createDimension("depth", len(depths))
        ds.createDimension("latitude", len(lats))
        ds.createDimension("longitude", len(lons))

        ds.createVariable("time_values", "f8", ("time",))[:] = times
        ds.createVariable("depth_values", "f4", ("depth",))[:] = depths
        ds.createVariable("latitude_values", "f4", ("latitude",))[:] = lats
        ds.createVariable("longitude_values", "f4", ("longitude",))[:] = lons

        ds.createVariable("time_idx", "i4", ("n_obs",))[:] = t_idx
        ds.createVariable("depth_idx", "i4", ("n_obs",))[:] = d_idx
        ds.createVariable("latitude_idx", "i4", ("n_obs",))[:] = la_idx
        ds.createVariable("longitude_idx", "i4", ("n_obs",))[:] = lo_idx

        ds.createVariable("temperature", "f4", ("n_obs",), fill_value=np.nan)[:] = temp_data
        ds.createVariable("salinity", "f4", ("n_obs",), fill_value=np.nan)[:] = sal_data
        ds.createVariable("dissolved_oxygen", "f4", ("n_obs",), fill_value=np.nan)[:] = dox_data

        ds.title = "Sparse Grouped Ocean Data"
        ds.history = "Created by script (sparse representation)"


# Start memory and time tracking
start_time = time.time()
tracemalloc.start()

# Print memory usage and elapsed time
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
tracemalloc.stop()

# Example usage:
save_sparse_to_netcdf(grouped, filename="grouped_ocean_data_sparse.nc")
