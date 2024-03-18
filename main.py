# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import seaborn as sns

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
from preprocess import merge_df
from dig_data import visualize_df,stationarity_test
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression



if __name__ == "__main__":
    merged_df, df_test, df_oil, df_holidays_events, df_stores, df_transactions = merge_df()
    visualize_df(merged_df,df_oil)
    stationarity_test(merged_df)
    # df_train_new,df_test_new = prework(df_train_new,df_test_new)
    # print(f"{'-'*10} df_train_new {'-'*10}")
    # check_na(df_train_new)
    # print(f"{'-'*10} df_test_new {'-'*10}")
    # check_na(df_test_new)


