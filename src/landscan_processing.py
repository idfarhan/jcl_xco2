"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import glob
import xarray as xr

def landscan_target_file(landscan_directory, target_date_str):
    """
    Find the Landscan file corresponding to the target date in the specified directory.

    Args:
        landscan_directory (str): The path to the directory containing Landscan files.
        target_date_str (str): The target date in 'YYYY-MM-dd' format.

    Returns:
        str or None: The path to the Landscan file corresponding to the target date, or None if no matching file is found.
    """
    try:
        # Extract year and month from the date string
        year, month = target_date_str.split('-')[:2]

        # Create a pattern to match files for the specified year and month
        pattern = f"{landscan_directory}/landscan-global-{year}.tif"

        # Search for files matching the pattern
        matches = glob.glob(pattern)

        # Return the first match or None if no matches are found
        return matches[0] if matches else None
    except Exception as e:
        # Handle any unexpected errors
        print("An error occurred:", e)
        return None
    
def landscan_read_file(landscan_dir, overpass_date):
    landscan_file = landscan_target_file(landscan_dir, overpass_date)
    landscan_ds = xr.open_dataset(landscan_file).rename({'band_data': 'landscan'})
    landscan_ds.close()
    return landscan_ds

