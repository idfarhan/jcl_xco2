"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import os, glob
import numpy as np
import xarray as xr
from osgeo import gdal

def gfed_target_file(gfed_directory, target_date_str):
    """
    Find the GFED file corresponding to the target date in the specified directory.

    Args:
        gfed_directory (str): The path to the directory containing GFED files.
        target_date_str (str): The target date in 'YYYY-MM-dd' format.

    Returns:
        str or None: The path to the GFED file corresponding to the target date, or None if no matching file is found.
    """
    try:
        # Extract year and month from the date string
        year, month = target_date_str.split('-')[:2]

        # Create a pattern to match files for the specified year and month
        pattern = f"{gfed_directory}/GFED4.1s_{year}*.hdf5"

        # Search for files matching the pattern
        matches = glob.glob(pattern)

        # Return the first match or None if no matches are found
        return matches[0] if matches else None
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred:", e)
        return None

'''
def define_coordinates_gfed(lat_bounds, lon_bounds, resolution):
    """
    Define latitude and longitude coordinates for GFED data.

    Args:
        lat_bounds (tuple): Latitude bounds in the format (min_lat, max_lat).
        lon_bounds (tuple): Longitude bounds in the format (min_lon, max_lon).
        resolution (tuple): Resolution of the grid in degrees for both latitude and longitude.

    Returns:
        tuple: Arrays containing longitude and latitude coordinates.
    """
    # Define longitude array with a given resolution and offset
    lon = np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + 0.25
    # Define latitude array with a given resolution and offset
    lat = np.arange(lat_bounds[0], lat_bounds[1], resolution[0]) + 0.25
    return lon, lat

'''

def define_coordinates_gfed(lat_bounds, lon_bounds, resolution):
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
    lon = np.arange(lon_bounds[0], lon_bounds[1], resolution[1]) + 0.25
    # Define latitude array with a given resolution and offset
    lat = np.arange(lat_bounds[1], lat_bounds[0], -resolution[0]) + 0.25  # Invert latitude
    return lon, lat


def hdf_to_xr_gfed(hdf_file, month):
    """
    Convert GFED HDF file to xarray dataset.

    Args:
        hdf_file (str): Path to the GFED HDF file.
        month (int): Month for which the data is extracted.

    Returns:
        xarray.Dataset or None: An xarray dataset containing GFED data if successful, otherwise None.
    """
    try:
        # Construct the subset path within the HDF file
        gfed_subset = "HDF5:" + hdf_file + "://emissions/" + str(month) + "/C"

        # Open the HDF5 file and read GFED data
        gfed_ds = gdal.Open(gfed_subset, gdal.GA_ReadOnly)
        if gfed_ds is None:
            raise RuntimeError("Failed to open HDF dataset")

        gfed_data = gfed_ds.ReadAsArray()
        if gfed_data is None:
            raise RuntimeError("Failed to read data from HDF dataset")
        
        # Replace invalid values (-3000) with NaN
        gfed_data = np.where(gfed_data == -3000, np.nan, gfed_data)

        # Define coordinates for the GFED grid
        longitude, latitude = define_coordinates_gfed([-90, 90], [-180, 180], [0.25, 0.25])

        # Create xarray dataset with GFED data and coordinates
        ds = xr.Dataset(
            {
                "gfed": (["latitude", "longitude"], gfed_data),
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

def gfed_read_file(gfed_dir, overpass_date):
    gfed_file = gfed_target_file(gfed_dir, overpass_date)
    gfed_ds = hdf_to_xr_gfed(gfed_file, overpass_date[5:7])
    gfed_ds.close()
    return gfed_ds