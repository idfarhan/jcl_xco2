"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import glob
import os
import xarray as xr

def odiac_target_file(odiac_directory, target_date_str):
    """
    Find the ODIAC file corresponding to the target date in the specified directory.

    Args:
        odiac_directory (str): The path to the directory containing ODIAC files.
        target_date_str (str): The target date in 'YYYY-MM-dd' format.

    Returns:
        str or None: The path to the ODIAC file corresponding to the target date, or None if no matching file is found.
    """
    try:
        # Extract year and month from the date string
        year, month = target_date_str.split('-')[:2]

        # Convert the year to the format used in the file names after 'intl_' ('YY')
        year_short = year[2:]

        # Create a pattern to match files for the specified year and month
        # Note: Adjusted pattern to match your file naming convention
        pattern = f"{odiac_directory}/{year}/odiac2022_1km_excl_intl_{year_short}{month}*.tif"

        # Search for files matching the pattern
        matches = glob.glob(pattern, recursive=True)

        # Return the first match or None if no matches are found
        return matches[0] if matches else None
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred:", e)
        return None
    
def odiac_read_file(odiac_dir, overpass_date):
    odiac_file = odiac_target_file(odiac_dir, overpass_date)
    odiac_ds = xr.open_dataset(odiac_file).rename({'band_data': 'odiac'})
    odiac_ds.close()
    return odiac_ds