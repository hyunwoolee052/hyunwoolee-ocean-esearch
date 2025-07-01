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
    df.groupby(["julian_time", "longitude", "latitude", "depth"], dropna=True)
    .agg({"temperature": "mean", "salinity": "mean", "dissolved_oxygen": "mean"})
    .reset_index()
)


# Save grouped data as a NetCDF file indexed by julian_time, longitude, latitude, and depth
def save_grouped_to_netcdf(grouped_df, filename="grouped_ocean_data.nc"):
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

    # Create empty arrays filled with NaN
    shape = (len(times), len(depths), len(lats), len(lons))
    temp_arr = np.full(shape, np.nan, dtype=np.float32)
    sal_arr = np.full(shape, np.nan, dtype=np.float32)
    dox_arr = np.full(shape, np.nan, dtype=np.float32)

    # Fill arrays
    for _, row in grouped_df.iterrows():
        t = time_index[row["julian_time"]]
        d = depth_index[row["depth"]]
        la = lat_index[row["latitude"]]
        lo = lon_index[row["longitude"]]
        temp_arr[t, d, la, lo] = row["temperature"]
        sal_arr[t, d, la, lo] = row["salinity"]
        dox_arr[t, d, la, lo] = row["dissolved_oxygen"]

    # Write to NetCDF
    with nc.Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("depth", len(depths))
        ds.createDimension("latitude", len(lats))
        ds.createDimension("longitude", len(lons))

        time_var = ds.createVariable("julian_time", "f8", ("time",))
        depth_var = ds.createVariable("depth", "f4", ("depth",))
        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lon_var = ds.createVariable("longitude", "f4", ("longitude",))

        temp_var = ds.createVariable(
            "temperature",
            "f4",
            ("time", "depth", "latitude", "longitude"),
            fill_value=np.nan,
        )
        sal_var = ds.createVariable(
            "salinity",
            "f4",
            ("time", "depth", "latitude", "longitude"),
            fill_value=np.nan,
        )
        dox_var = ds.createVariable(
            "dissolved_oxygen",
            "f4",
            ("time", "depth", "latitude", "longitude"),
            fill_value=np.nan,
        )

        time_var[:] = times
        depth_var[:] = depths
        lat_var[:] = lats
        lon_var[:] = lons

        temp_var[:, :, :, :] = temp_arr
        sal_var[:, :, :, :] = sal_arr
        dox_var[:, :, :, :] = dox_arr

        ds.title = "Grouped Ocean Data"
        ds.history = "Created by script"


def save_grouped_to_sparse_npz(grouped_df, filename="grouped_ocean_data_sparse.npz"):
    """
    Save grouped data as sparse arrays (CSR) for temperature, salinity, and dissolved oxygen.
    Index order: (julian_time, depth, latitude, longitude)
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

    # Prepare lists for COO sparse matrix construction
    coords = []
    temp_data = []
    sal_data = []
    dox_data = []

    for _, row in grouped_df.iterrows():
        t = time_index[row["julian_time"]]
        d = depth_index[row["depth"]]
        la = lat_index[row["latitude"]]
        lo = lon_index[row["longitude"]]
        coords.append((t, d, la, lo))
        temp_data.append(row["temperature"])
        sal_data.append(row["salinity"])
        dox_data.append(row["dissolved_oxygen"])

    coords = np.array(coords)
    shape = (len(times), len(depths), len(lats), len(lons))

    # Create sparse COO matrices
    temp_sparse = sparse.coo_matrix(
        (temp_data, (coords[:, 0], coords[:, 1] * len(lats) * len(lons) + coords[:, 2] * len(lons) + coords[:, 3])),
        shape=(shape[0], shape[1] * shape[2] * shape[3]),
    )
    sal_sparse = sparse.coo_matrix(
        (sal_data, (coords[:, 0], coords[:, 1] * len(lats) * len(lons) + coords[:, 2] * len(lons) + coords[:, 3])),
        shape=(shape[0], shape[1] * shape[2] * shape[3]),
    )
    dox_sparse = sparse.coo_matrix(
        (dox_data, (coords[:, 0], coords[:, 1] * len(lats) * len(lons) + coords[:, 2] * len(lons) + coords[:, 3])),
        shape=(shape[0], shape[1] * shape[2] * shape[3]),
    )

    # Save as .npz file
    np.savez_compressed(
        filename,
        temp_data=temp_sparse.data,
        temp_row=temp_sparse.row,
        temp_col=temp_sparse.col,
        temp_shape=temp_sparse.shape,
        sal_data=sal_sparse.data,
        sal_row=sal_sparse.row,
        sal_col=sal_sparse.col,
        sal_shape=sal_sparse.shape,
        dox_data=dox_sparse.data,
        dox_row=dox_sparse.row,
        dox_col=dox_sparse.col,
        dox_shape=dox_sparse.shape,
        times=times,
        depths=depths,
        lats=lats,
        lons=lons,
    )
    print(f"Sparse arrays saved to {filename}")


# Start memory and time tracking
start_time = time.time()
tracemalloc.start()

# Print memory usage and elapsed time
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

# Save as sparse array
save_grouped_to_sparse_npz(grouped)

# Print memory usage and elapsed time
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.2f} MB; Peak: {peak / 1024**2:.2f} MB")
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
tracemalloc.stop()
