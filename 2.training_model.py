"""
JCL XCO2 Model: A machine-learning-based model to predict high spatiotemporal global XCO2 dataset
Author: Dr. Farhan Mustafa
Email: fmustafa@ust.hk, idfarhan@gmail.com
Fok Ying Tun Research Institute (FYTRI), Hong Kong University of Science and Technology (HKUST)  
"""


import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

def training_model(training_data):

    '''
    Train an XGBoost regression model to predict XCO2 dataset.

    Parameters:
    training_data (str): Path to the CSV file containing the training data.

    Returns:
    The trained model in .pkl format.
    '''

    cols = ['time', 'longitude', 'latitude', 'xco2', 'u10', 'v10', 'cams', 'odiac', 'ndvi', 'landscan', 'gfed']
    df = pd.read_csv(training_data, names=cols)
    df.dropna(inplace=True)

    X_tmp = df.drop(['time', 'longitude', 'latitude', 'xco2'], axis=1)
    std = StandardScaler()
    X_data = std.fit_transform(X_tmp)
    X = pd.DataFrame(X_data, columns = X_tmp.columns)

    y = df[['xco2']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1742)

    ''' For hyperparameter tuning

    cv_params = {'n_estimators': [6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 1500, 'max_depth': 10, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'tree_method':'gpu_hist'}
    fit_params = {'early_stopping_rounds': 50}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=10, verbose=1, n_jobs=-1)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('each_iteration:{0}'.format(evalute_result))
    print('best_params：{0}'.format(optimized_GBM.best_params_))
    print('best_score:{0}'.format(optimized_GBM.best_score_))

    cv_params = {'max_depth': [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14]}
    other_params = {'learning_rate': 0.2, 'n_estimators': 8000, 'max_depth': 9, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method':'gpu_hist'}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=10, verbose=1, n_jobs=-1)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('each_iteration:{0}'.format(evalute_result))
    print('best_params：{0}'.format(optimized_GBM.best_params_))
    print('best_score:{0}'.format(optimized_GBM.best_score_))

    cv_params = {'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    other_params = {'learning_rate': 0.2, 'n_estimators': 5000, 'max_depth': 9, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'tree_method':'gpu_hist'}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=10, verbose=1, n_jobs=22)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('each_iteration:{0}'.format(evalute_result))
    print('best_params：{0}'.format(optimized_GBM.best_params_))
    print('best_score:{0}'.format(optimized_GBM.best_score_))

    cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 8000, 'max_depth': 7, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                    'tree_method':'gpu_hist'}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=10, verbose=1, n_jobs=-1)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('each_iteration:{0}'.format(evalute_result))
    print('best_params：{0}'.format(optimized_GBM.best_params_))
    print('best_score:{0}'.format(optimized_GBM.best_score_))

    '''

    XGBR = xgb.XGBRegressor(n_jobs=-1, n_estimators=8000, max_depth=7, min_child_weight=1, learning_rate=0.1)
    XGBR.fit(X_train, y_train)
    print('Predicting results')
    y_pred = XGBR.predict(X_test)
    
    print(f'RMSE : {np.sqrt(mean_absolute_error(y_test, y_pred))}')
    print(f'R2 : {r2_score(y_test, y_pred)}')

    print('The machine learning model is successfully trained.')

    return joblib.dump(XGBR, 'trained_model.pkl')

def main():
    training_data = 'training_data.csv'
    training_model(training_data)

if __name__ == "__main__":
    main()