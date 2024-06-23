"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import os, glob
import xarray as xr
from .era5_processing import adjusting_longitude

def cams_target_file(cams_directory, target_date_str):
    """
    Find the CAMS file corresponding to the target date in the specified directory.

    Args:
        cams_directory (str): The path to the directory containing CAMS files.
        target_date_str (str): The target date in 'YYYY-MM' format.

    Returns:
        str or None: The path to the CAMS file corresponding to the target date, or None if no matching file is found.
    """
    try:
        # Extract year and month from the date string
        year, month = target_date_str.split('-')[:2]

        # Create a pattern to match files for the specified year and month
        pattern = f"{cams_directory}/cams73_latest_co2_col_surface_inst_{year}{month}*.nc"

        # Search for files matching the pattern
        matches = glob.glob(pattern)

        # Return the first match or None if no matches are found
        return matches[0] if matches else None
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred:", e)
        return None
    
def cams_read_file(cams_dir, overpass_date):
    cams_file = cams_target_file(cams_dir, overpass_date)
    cams_ds_monthly = xr.open_dataset(cams_file).rename({'XCO2': 'cams'})
    cams_ds = cams_ds_monthly.sel(time=overpass_date)
    cams_ds_monthly.close()
    return cams_ds
    
def cams_0p75_read_file(cams_file, overpass_date):
    cams_ds = xr.open_dataset(cams_file).rename({'tcco2': 'cams'})
    cams_ds = adjusting_longitude(cams_ds)
    cams_ds = cams_ds.sel(time=overpass_date)
    cams_ds.close()
    return cams_ds