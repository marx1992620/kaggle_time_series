# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import seaborn as sns

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
from preprocess import merge_df
from data_mining import visualize_df, stationary_test, t_test, check_sales_dates, check_sales_relation, extract_date, reset_category, feature_scaling, one_hot_encoding
from data_modeling import split_dataset, data_seperation, linear_regression
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression



if __name__ == "__main__":
    merged_df, df_test, df_oil, df_holidays_events, df_stores, df_transactions = merge_df()
    visualize_df(merged_df,df_oil)
    # stationary_test(merged_df)
    # t_test(merged_df)
    # check_sales_dates(merged_df)
    # check_sales_relation(merged_df)
    merged_df_copy,x_test = extract_date(merged_df)
    reset_category(x_test)
    feature_scaling(x_test)
    x_test_encoded = one_hot_encoding(x_test)
    x_test_encoded.drop(['holiday_type_nan'], axis=1, inplace=True)
    x_test_encoded = x_test_encoded.fillna(0)
    # x_test_encoded.insert(45, 'holiday_type_Transfer', 0)
    # x_test_encoded.insert(44, 'holiday_type_Event', 0)
    # x_test_encoded.insert(44, 'holiday_type_Bridge', 0)
    # x_test_encoded.insert(44, 'holiday_type_Additional', 0)
    x_test_encoded.to_csv('data/test/x_test.csv',index=False)

    reset_category(merged_df_copy)
    feature_scaling(merged_df_copy)
    merged_df_encoded = one_hot_encoding(merged_df_copy)
    train_set, valid_set = split_dataset(merged_df_encoded)
    x_train, y_train, x_valid, y_valid, result_df = data_seperation(train_set,valid_set)
    linear_regression(x_train,y_train,x_valid,y_valid)

    # df_train_new,df_test_new = prework(df_train_new,df_test_new)
    # print(f"{'-'*10} df_train_new {'-'*10}")
    # check_na(df_train_new)
    # print(f"{'-'*10} df_test_new {'-'*10}")
    # check_na(df_test_new)


