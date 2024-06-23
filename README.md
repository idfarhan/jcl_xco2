# jcl_xco2
A machine learning model to produce XCO2 dataset

Input Datasets:

1. OCO-2 XCO2 (.nc4 files).    
2. ERA-5 hourly wind vector (.nc files)    
3. CAMS-EGG4 XCO2 (.nc file)
4. ODIAC (.tif files)
5. MODIS NDVI (.hdf files).
6. Landscan population density (.tif files).
7. GFED emissions (.hdf files).

Procedure:
1. Set directories of input files in "1.data_preparation.py" and execute it to produce a training data.
2. Use the training data as an input in "2.training_model.py" to train the machine learning model.
3. Set the directories of the input files in "3.predictions.py" and execute it to generate XCO2 dataset.

For queries, please contact Dr. Farhan Mustafa (fmustafa@ust.hk).


