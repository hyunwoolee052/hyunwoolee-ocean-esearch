import pandas as pd
import numpy as np
from pathlib import Path
import netCDF4 as nc

DATA_DIR = Path.cwd() / "data/"
STANDARD_DEPTHS = np.array(
    [
        0,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
    ]
)


def concat_data_array(param_name: str, sdate: str, edate: str) -> np.ndarray:
    """
    Concatenate a parameter from multiple JSON files into a single NumPy array with NaN for missing values.

    Parameters
    ----------
    param_name : str
        Name of the parameter/column to extract from each JSON file.
    sdate : str
        Start date in 'YYYY-MM-DD' format.
    edate : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    np.ndarray
        Concatenated array of the parameter values, with missing values as np.nan.
    """
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
    return np.array([])


def build_dataframe(sdate: str, edate: str) -> pd.DataFrame:
    """
    Build the main DataFrame from all arrays, apply standard depth matching and 3-sigma QC.

    Parameters
    ----------
    sdate : str
        Start date in 'YYYY-MM-DD' format.
    edate : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, longitude, latitude, depth, std_depth,
        temperature, salinity, dissolved_oxygen. Salinity and dissolved oxygen are
        quality controlled (3-sigma).
    """
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
    # Only keep rows with valid depth
    df = df[~np.isnan(df["depth"])]
    # Find closest standard depth for each row
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
    """
    Save original (non-aggregated) data as sparse arrays to NetCDF4 file.
    All variables are saved as a single-precision (float32) array for memory efficiency.

    Dimensions
    ----------
    t : time (datetime as string)
    z : standard depth (float)
    y : latitude (float)
    x : longitude (float)
    n_obs : number of valid observations

    Variables
    ---------
    wtr_tmp : float32
        Water temperature (sparse, indexed by n_obs).
    sal : float32
        Salinity (sparse, indexed by n_obs).
    dox : float32
        Dissolved oxygen (sparse, indexed by n_obs).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: datetime, longitude, latitude, std_depth, temperature, salinity, dissolved_oxygen.
    filename : str, optional
        Output NetCDF4 file name.

    Returns
    -------
    None
    """
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
        # Skip rows where any index dimension is NaN
        if (
            pd.isna(row["datetime"])
            or np.isnan(row["longitude"])
            or np.isnan(row["latitude"])
            or np.isnan(row["std_depth"])
        ):
            continue
        # Only store rows where at least one variable is not NaN
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


if __name__ == "__main__":
    sdate = "1968-01-01"
    edate = "2025-12-31"
    df = build_dataframe(sdate, edate)
    save_sparse_to_netcdf_spatiotemporal(
        df, filename="sooList1968010120241231.nc"
    )
