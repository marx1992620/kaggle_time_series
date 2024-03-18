import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
# Modelling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import RandomizedSearchCV
import pickle


def split_dataset(merged_df_encoded):
    train_set = merged_df_encoded.loc[merged_df_encoded['year'].isin([2013, 2014, 2015, 2016])]
    valid_set = merged_df_encoded.loc[merged_df_encoded['year'] == 2017]
    print(train_set.shape)
    print(valid_set.shape)

    return train_set, valid_set


def data_seperation(train_set,valid_set):
    # For the training set
    x_train = train_set.drop('sales', axis=1)
    y_train = train_set['sales']
    # For the evaluation set
    x_valid = valid_set.drop('sales', axis=1)
    y_valid = valid_set['sales']

    # Initialize the results dataframe
    result_df = pd.DataFrame(columns=['Model', 'RMSLE', 'RMSE', 'MSE', 'MAE'])
    return x_train, y_train, x_valid, y_valid, result_df


def linear_regression(x_train,y_train,x_valid,y_valid):
    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    linear_predictions = linear_model.predict(x_valid)

    # Calculate metrics
    linear_mse = mean_squared_error(y_valid, linear_predictions)
    linear_mae = mean_absolute_error(y_valid, linear_predictions)

    # Apply the absolute value function to both y_eval and lr_predictions
    y_valid_abs = abs(y_valid)
    linear_predictions_abs = abs(linear_predictions)

    # Calculate the Root Mean Squared Logarithmic Error (RMSLE)
    linear_rmsle = np.sqrt(mean_squared_log_error(y_valid_abs, linear_predictions_abs))

    # Create a DataFrame to store results for Linear Regression
    results = pd.DataFrame({'Model': ['Linear Regression'],
                                'RMSLE': [linear_rmsle],
                                'RMSE': [np.sqrt(linear_mse)],
                                'MSE': [linear_mse],
                                'MAE': [linear_mae]}).round(2)

    # Print the results dataframe
    print(results)
    x_test = pd.read_csv("data/test/df_test.csv")
    res = linear_model.predict(x_test)
    print(len(res))


    test_results = pd.DataFrame({'sales': res})
    test_results.to_csv('./submission_2.csv',index=False)
