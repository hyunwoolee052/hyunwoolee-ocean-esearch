import pandas as pd
import netCDF4 as nc
import numpy as np
import dask.dataframe as dd

def write_netcdf(data: pd.DataFrame, filename: str, chunk_size: int = 100000):
    # Use Dask DataFrame for chunked processing
    ddf = dd.from_pandas(data, npartitions=max(1, int(len(data) * data.memory_usage(deep=True).sum() // (1e9 * chunk_size))))

    # Prepare unique sorted dimension values
    times = pd.to_datetime(data['time']).sort_values().unique()
    depths = np.sort(data['depth'].unique())
    lons = np.sort(data['longitude'].unique())
    lats = np.sort(data['latitude'].unique())
    times64 = times.astype('datetime64[ns]')

    with nc.Dataset(filename, 'w', format='NETCDF4') as dataset:
        # Create dimensions
        dataset.createDimension('time', len(times))
        dataset.createDimension('depth', len(depths))
        dataset.createDimension('longitude', len(lons))
        dataset.createDimension('latitude', len(lats))

        # Create variables for dimensions
        time_var = dataset.createVariable('time', 'f8', ('time',))
        depth_var = dataset.createVariable('depth', 'f4', ('depth',))
        longitude_var = dataset.createVariable('longitude', 'f4', ('longitude',))
        latitude_var = dataset.createVariable('latitude', 'f4', ('latitude',))

        # Assign data to dimension variables
        time_var[:] = pd.to_datetime(times).astype('int64') // 10**9  # seconds since epoch
        depth_var[:] = depths
        longitude_var[:] = lons
        latitude_var[:] = lats

        # Create main variables (4D: time, depth, lat, lon)
        temp_var = dataset.createVariable('temperature', 'f4', ('time', 'depth', 'latitude', 'longitude'), fill_value=np.nan)
        sal_var = dataset.createVariable('salinity', 'f4', ('time', 'depth', 'latitude', 'longitude'), fill_value=np.nan)
        dox_var = dataset.createVariable('dissolved_oxygen', 'f4', ('time', 'depth', 'latitude', 'longitude'), fill_value=np.nan)

        # Do NOT pre-fill with np.nan! fill_value handles this.

        # Process in Dask chunks to limit RAM usage
        for partition in ddf.partitions:
            chunk = partition.compute()
            for _, row in chunk.iterrows():
                row_time64 = np.datetime64(pd.to_datetime(row['time']))
                t_idx = np.where(times64 == row_time64)[0][0]
                d_idx = np.where(depths == row['depth'])[0][0]
                lat_idx = np.where(lats == row['latitude'])[0][0]
                lon_idx = np.where(lons == row['longitude'])[0][0]
                temp_var[t_idx, d_idx, lat_idx, lon_idx] = row['temperature']
                sal_var[t_idx, d_idx, lat_idx, lon_idx] = row['salinity']
                dox_var[t_idx, d_idx, lat_idx, lon_idx] = row['dissolved_oxygen']

        # Add attributes
        dataset.title = 'KODC Data'
        dataset.history = 'Created ' + str(pd.Timestamp.now())