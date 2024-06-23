"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
from datetime import datetime, timedelta


def modis_target_file(modis_directory, target_date_str):
    """
    Find the MODIS HDF file closest to the target date in the specified directory.

    Args:
        modis_directory (str): The path to the directory containing MODIS HDF files.
        target_date_str (str): The target date in 'YYYY-MM-DD' format.

    Returns:
        str: The path to the closest MODIS HDF file to the target date.
             If no HDF files are found in the directory or the directory doesn't exist, an appropriate message is returned.
    """
    
    # Function to convert filename to date
    def filename_to_date(filename):
        parts = filename.split('.')
        # Extract year from the filename
        year = int(parts[1][1:5])
        # Extract day of year from the filename
        doy = int(parts[1][5:8])
        # Calculate the date from year and day of year
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        return date

    # Check if the provided directory exists
    if not os.path.isdir(modis_directory):
        return "Directory not found."

    # Convert target date string to datetime object
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    except ValueError:
        return "Invalid target date format. Please provide date in 'YYYY-MM-DD' format."

    # Read all filenames in the directory and collect data
    files_data = []
    for filename in os.listdir(modis_directory):
        if filename.endswith('.hdf'):  # Check if the file is an HDF file
            file_date = filename_to_date(filename)
            file_path = os.path.join(modis_directory, filename)
            date_diff = abs((file_date - target_date).days)
            files_data.append((file_path, file_date, date_diff))

    # Check if HDF files were found in the directory
    if not files_data:
        return "No HDF files found in the directory."

    # Convert list to DataFrame
    df = pd.DataFrame(files_data, columns=['FilePath', 'Date', 'DateDiff'])

    # Find the file with the minimum date difference
    min_diff_row = df.loc[df['DateDiff'].idxmin()]
    closest_file_path = min_diff_row['FilePath']
    return closest_file_path

def define_coordinates_modis(lat_bounds, lon_bounds, resolution):
    """
    Define latitude and longitude coordinates for MODIS data.

    Args:
        lat_bounds (tuple): Latitude bounds in the format (min_lat, max_lat).
        lon_bounds (tuple): Longitude bounds in the format (min_lon, max_lon).
        resolution (tuple): Resolution of the grid in degrees for both latitude and longitude.

    Returns:
        tuple: Arrays containing longitude and latitude coordinates.
    """
    # Define longitude array with a given resolution and offset
    lon = np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + 0.05
    # Define latitude array with a given resolution and offset
    lat = np.arange(lat_bounds[1], lat_bounds[0], -resolution[0]) + 0.05  # Invert latitude
    return lon, lat


def hdf_to_xr_modis(hdf_file):
    """
    Convert MODIS HDF file to xarray dataset.

    Args:
        hdf_file (str): Path to the MODIS HDF file.

    Returns:
        xarray.Dataset or None: An xarray dataset containing NDVI data if successful, otherwise None.
    """
    try:
        # Construct the path to the NDVI subset within the HDF file
        ndvi_subset = "HDF4_EOS:EOS_GRID:" + hdf_file + ":MODIS_Grid_16Day_VI_CMG:CMG 0.05 Deg 16 days NDVI"

        # Open the HDF5 file and read NDVI data
        ndvi_ds = gdal.Open(ndvi_subset, gdal.GA_ReadOnly)
        if ndvi_ds is None:
            raise RuntimeError("Failed to open HDF dataset")

        ndvi_data = ndvi_ds.ReadAsArray()
        if ndvi_data is None:
            raise RuntimeError("Failed to read data from HDF dataset")

        # Replace invalid values (-3000) with NaN
        ndvi_data = np.where(ndvi_data == -3000, np.nan, ndvi_data)

        # Define coordinates for the MODIS grid
        longitude, latitude = define_coordinates_modis([-90, 90], [-180, 180], [0.05, 0.05])

        # Create xarray dataset with NDVI data and coordinates
        ds = xr.Dataset(
            {
                "ndvi": (["latitude", "longitude"], ndvi_data),
            },
            coords={
                "longitude": longitude,
                "latitude": latitude,
            },
        )

        return ds
    except Exception as e:
        print("An error occurred:", e)
        return None

def modis_ndvi_read_file(ndvi_dir, overpass_date):
    ndvi_file = modis_target_file(ndvi_dir, overpass_date)
    ndvi_ds = hdf_to_xr_modis(ndvi_file)
    return ndvi_ds


