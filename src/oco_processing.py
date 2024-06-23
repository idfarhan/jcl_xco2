"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""

import os, glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from .reference_grid import ref_grid
from shapely.geometry import Point


def oco_data_files(oco_dir, years):
    oco_files = []
    for year in years:
        year_str = str(year)[2:4]
        files_for_year = glob.glob(f'{oco_dir}/oco2_LtCO2_{year_str}*.nc4')
        oco_files.extend(files_for_year)
    oco_files.sort()
    return oco_files


def oco_gridding(oco_file, grid):
    oco_ds = xr.open_dataset(oco_file)
    oco_df = oco_ds[['time', 'longitude', 'latitude', 'xco2',
                     'xco2_quality_flag']].to_dataframe().reset_index()
    overpass_date = oco_df['time'].dt.date.iloc[0]
    overpass_date = overpass_date.strftime('%Y-%m-%d')
    print('Processing', overpass_date)
    oco_df = oco_df[oco_df['xco2_quality_flag'] != 1]
    oco_df.drop('xco2_quality_flag', axis=1, inplace=True)
    oco_df['time'] = pd.to_datetime(oco_df['time']).dt.tz_localize('UTC')
    oco_df['rounded_time'] = oco_df['time'].dt.floor('H')

    geometry = [Point(xy) for xy in zip(oco_df['longitude'], oco_df['latitude'])]
    oco_gdf = gpd.GeoDataFrame(oco_df, geometry=geometry)
    oco_gdf.crs = 'EPSG:4326'
    joined_gdf = gpd.sjoin(oco_gdf, grid, how='inner', predicate='within')
    oco_mean = joined_gdf.groupby(['rounded_time', 'lon', 'lat']).agg({'xco2':'mean'}).reset_index()
    oco_mean.rename(columns = {'lon': 'longitude', 'lat': 'latitude'}, inplace=True)
    
    oco_ds.close()

    return overpass_date, oco_mean
