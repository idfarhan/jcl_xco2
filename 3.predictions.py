"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from src.era5_processing import era_read_file
from src.cams_processing import cams_0p75_read_file
from src.odiac_processing import odiac_read_file
from src.modis_processing import modis_ndvi_read_file
from src.landscan_processing import landscan_read_file
from src.gfed_processing import gfed_read_file
from sklearn.preprocessing import StandardScaler
import joblib


def doy(year):

    '''
    Generate a list of dates for a given year.
    
    Args:
    - year (int): The year for which to generate the list of dates.
    
    Returns:
    - list: List of date strings in "YYYY-MM-DD" format for each day of the year.
    '''

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    current_date = start_date
    date_list = []
    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += datetime.timedelta(days=1)

    return date_list

def predictions(cams_file, era_dir, odiac_dir, ndvi_dir, landscan_dir, gfed_dir, trained_model, year, output_dir):

    '''

    Generate XCO2 predictions using various data sources and a trained machine learning model.
    
    Args:
    - cams_file (str): Path to the CAMS XCO2 data file (.nc file).
    - era_dir (str): Directory containing ERA5 data files (.nc files).
    - odiac_dir (str): Directory containing ODIAC data files (.tif files).
    - ndvi_dir (str): Directory containing MODIS NDVI data files (.hdf files).
    - landscan_dir (str): Directory containing Landscan data files (.tif files).
    - gfed_dir (str): Directory containing GFED data files (.hdf files).
    - trained_model (str): Path to the trained machine learning model file (.pkl file).
    - year (int): Year for which predictions are to be generated.
    - output_dir (str): Directory where the output files will be saved.
    
    '''
    
    new_lats = np.arange(-89.95, 90, 0.1)
    new_lons = np.arange(-179.95, 180, 0.1)

    model = joblib.load(trained_model)

    days = doy(year)
    for d in days:

        filename = f'model_v1.0_xco2_{d}.nc'
        output_filename = os.path.join(output_dir, filename)

        # Check if the output file already exists
        if os.path.exists(output_filename):
            print(f"File already exists for {d}, skipping...")
            continue
    
    odiac_ds = odiac_read_file(odiac_dir, d).rename(
        {'x': 'longitude', 'y': 'latitude'}).drop_vars(['band', 'spatial_ref'])
    odiac_interp = odiac_ds.interp(latitude=new_lats, longitude=new_lons,
                                   kwargs={"fill_value": "extrapolate"})
    ndvi_ds = modis_ndvi_read_file(ndvi_dir, d)
    ndvi_interp = ndvi_ds.interp(latitude=new_lats, longitude=new_lons,
                                 kwargs={"fill_value": "extrapolate"})
    ndvi_interp['ndvi'] = xr.where((ndvi_interp['ndvi'] < 0) | ndvi_interp['ndvi'].isnull(),
                                   -3000, ndvi_interp['ndvi'])
    landscan_ds = landscan_read_file(landscan_dir, d).rename(
        {'x': 'longitude', 'y': 'latitude'}).drop_vars(['band', 'spatial_ref'])
    landscan_interp = landscan_ds.interp(latitude=new_lats, longitude=new_lons,
                                         kwargs={"fill_value": "extrapolate"})
    landscan_interp['landscan'] = xr.where((landscan_interp['landscan'] < 0) | landscan_interp['landscan'].isnull(),
                                           0, landscan_interp['landscan'])
    gfed_ds = gfed_read_file(gfed_dir, d)
    gfed_interp = gfed_ds.interp(latitude=new_lats, longitude=new_lons,
                                 kwargs={"fill_value": "extrapolate"})
    cams_tmp = cams_0p75_read_file(cams_file, d)
    
    predicted_df = pd.DataFrame()
    for cams_time in cams_tmp.time.values:
        era_ds = era_read_file(era_dir, d).sel(time=cams_time).drop_vars('time')
        era_interp = era_ds.interp(latitude=new_lats, longitude=new_lons,
                                   kwargs={"fill_value": "extrapolate"})
        cams_ds = (cams_tmp).sel(time=cams_time).drop_vars('time')
        cams_interp = cams_ds.interp(latitude=new_lats, longitude=new_lons,
                                     kwargs={"fill_value": "extrapolate"})

        predicting_ds = xr.merge([era_interp, cams_interp, odiac_interp, ndvi_interp, landscan_interp, gfed_interp])
        predicting_df = predicting_ds.to_dataframe().reset_index()
        X_df = predicting_df.drop(['longitude', 'latitude', 'band'], axis=1)

        std = StandardScaler()
        X_tmp = std.fit_transform(X_df)
        X_tmp_df = pd.DataFrame(X_tmp, columns = X_df.columns)
        print('Predicting', cams_time)
        prediction = model.predict(X_tmp_df)
        predicted_df_tmp = pd.DataFrame({
            'longitude': predicting_df['longitude'],
            'latitude': predicting_df['latitude'],
            'time': cams_time,
            'XCO2': prediction})
        
        predicted_df = pd.concat([predicted_df, predicted_df_tmp], ignore_index=True)
    
    predicted_ds = xr.Dataset.from_dataframe(predicted_df.set_index(['time', 'latitude', 'longitude']))
    predicted_ds['XCO2'].attrs['units'] = 'ppm'
    predicted_ds['XCO2'].attrs['long_name'] = 'Column-averaged dry-air mole fraction of CO2'
    predicted_ds['time'].attrs['long_name'] = 'Time'
    predicted_ds['longitude'].attrs['units'] = 'degrees_east'
    predicted_ds['longitude'].attrs['long_name'] = 'Longitude'
    predicted_ds['latitude'].attrs['units'] = 'degrees_north'
    predicted_ds['latitude'].attrs['long_name'] = 'Latitude'
    
    return predicted_ds.to_netcdf(output_filename)

def main():
    # Define directories and parameters
    cams_file = ''
    era_dir = ''
    odiac_dir = ''
    ndvi_dir = ''
    landscan_dir = ''
    gfed_dir = ''
    trained_model = ''
    year = 2019
    output_dir = ''
    
    predictions(cams_file, era_dir, odiac_dir, ndvi_dir, landscan_dir,
                gfed_dir, trained_model, year, output_dir)
    
if __name__ == "__main__":
    main()