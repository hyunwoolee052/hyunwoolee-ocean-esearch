# KODC NetCDF Project

This project is designed to process oceanographic data from JSON files and save the concatenated results in a NetCDF4 format for use with the Iris module.

## Project Structure

```
kodc_netcdf_project
├── src
│   ├── main.py            # Entry point of the application
│   ├── data_processing.py  # Functions for concatenating data from JSON files
│   ├── netcdf_writer.py    # Functions for writing data to NetCDF4 files
│   └── utils.py            # Utility functions for data processing
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the project, execute the `main.py` file:

```bash
python src/main.py
```

## Module Descriptions

### `src/data_processing.py`

This module contains the function `concat_data(param_name: str, sdate: str, edate: str)` which collects and concatenates data based on specified parameters and date ranges.

### `src/netcdf_writer.py`

This module provides the function `write_netcdf(data: pd.DataFrame, filename: str)` which takes a DataFrame and a filename as arguments and saves the data in NetCDF format.

### `src/utils.py`

This module includes various utility functions that assist in data processing or file handling, such as data validation and logging.

## Dependencies

The project requires the following Python packages:

- pandas
- netCDF4
- iris

Make sure to install these packages using the `requirements.txt` file.