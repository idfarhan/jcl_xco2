"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import os, glob
import xarray as xr

def era_target_file(era_directory, target_date_str):
    """
    Find the ERA file corresponding to the target date in the specified directory.

    Args:
        era_directory (str): The path to the directory containing ERA files.
        target_date_str (str): The target date in 'YYYY-MM-dd' format.

    Returns:
        str or None: The path to the ERA file corresponding to the target date, or None if no matching file is found.
    """
    try:
        # Extract year and month from the date string
        year, month = target_date_str.split('-')[:2]

        # Create a pattern to match files for the specified year and month
        pattern = f"{era_directory}/era_{year}_{month}*.nc"

        # Search for files matching the pattern
        matches = glob.glob(pattern)

        # Return the first match or None if no matches are found
        return matches[0] if matches else None
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred:", e)
        return None

def adjusting_longitude(ds):

    """
    Adjust longitudes in an xarray dataset to the range of (-180, 180).

    Parameters:
    - ds (xarray.Dataset): The input xarray dataset containing longitude values.

    Returns:
    - ds (xarray.Dataset): A modified xarray dataset with longitudes adjusted to the range (-180, 180).

    This function takes an xarray dataset with longitude values and adjusts them to fit within the range
    of -180 degrees to 180 degrees. Longitudes greater than 180 degrees are shifted to their equivalent values
    within the specified range by subtracting 360 degrees.

    Note:
    - This function assumes that the input dataset contains a 'longitude' variable.
    - The adjusted longitude values are stored in a new variable called '_longitude_adjusted.'
    - The adjusted dataset has its 'longitude' dimension replaced, sorted, and the original 'longitude' variable dropped.

    Example usage:
    adjusted_ds = adjusting_longitude(input_ds)
    """

    ds['_longitude_adjusted'] = xr.where(
        ds['longitude'] > 180,
        ds['longitude'] - 360,
        ds['longitude'])
    ds = (
    ds
        .swap_dims({'longitude': '_longitude_adjusted'})
        .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
        .drop('longitude'))
    ds = ds.rename({'_longitude_adjusted': 'longitude'})
    return ds

def era_read_file(era_dir, overpass_date):
    era_file = era_target_file(era_dir, overpass_date)
    era_ds_monthly = xr.open_dataset(era_file)
    era_ds_monthly = adjusting_longitude(era_ds_monthly)
    era_ds = era_ds_monthly.sel(time=overpass_date)
    era_ds_monthly.close()
    return era_ds


