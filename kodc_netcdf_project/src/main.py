import pandas as pd
from datetime import datetime
from data_processing import concat_data
from netcdf_writer import write_netcdf

def main():
    sdate = "1968-01-01"
    edate = "2025-12-31"
    
    # Concatenate data for various parameters
    temperature = concat_data("wtr_tmp", sdate, edate)
    depth = concat_data("wtr_dep", sdate, edate)
    time = concat_data("obs_dtm", sdate, edate)
    longitude = concat_data("lon", sdate, edate)
    latitude = concat_data("lat", sdate, edate)
    salinity = concat_data("sal", sdate, edate)
    dissolved_oxygen = concat_data("dox", sdate, edate)

    # Combine all data into a single DataFrame
    combined_data = pd.DataFrame({
        'time': time["obs_dtm"],
        'temperature': temperature["wtr_tmp"],
        'depth': depth["wtr_dep"],
        'longitude': longitude["lon"],
        'latitude': latitude["lat"],
        'salinity': salinity["sal"],
        'dissolved_oxygen': dissolved_oxygen["dox"],
    })

    # Write the combined data to a NetCDF file
    write_netcdf(combined_data, "output_data.nc")

if __name__ == "__main__":
    main()