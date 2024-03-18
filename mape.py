import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from etl import merge_df, check_na, prework, holiday, split_date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


def build_random_forest(x_data,y_data,data_frame):
    x_ = data_frame[x_data]
    y_ = data_frame[y_data]
    model = RandomForestRegressor(random_state=1,max_depth=100)
    a_x,b_x,a_y,b_y = train_test_split(x_,y_,random_state=1)
    model.fit(a_x,a_y)
    predictions = model.predict(b_x)
    delta = mean_absolute_error(b_y,predictions)
    print(f"mean_absolute_error delta: {delta}")


def Forecast(x_feature,y_feature):
    model = RandomForestRegressor(random_state=1,max_depth=100)
    model.fit(df_train_new[x_feature],df_train_new[y_feature])
    aim = df_test_new[x_feature]
    predictions = model.predict(aim)
    print(predictions)
    res = pd.DataFrame(predictions)
    path = './submission.csv'
    res.to_csv(path)


if __name__ == "__main__":
    print(f"{'='*10} start {'='*10}")
    df_train_new,df_test_new = merge_df()
    print(f"df_train_new shape: {df_train_new.shape} , df_test_new shape: {df_test_new.shape}")

    df_train_new , df_test_new = prework(df_train_new,df_test_new)
    print("\nprework done !\n")

    print(f"df_train_new shape: {df_train_new.shape} , df_test_new shape: {df_test_new.shape}")
    # print(f"{'-'*10} df_train_new {'-'*10}")
    # check_na(df_train_new)
    # print(f"{'-'*10} df_test_new {'-'*10}")
    # check_na(df_test_new)
    # print(df_test_new.isnull().sum())
    df_train_new = split_date(df_train_new)
    df_test_new = split_date(df_test_new)
    df_train_new = holiday(df_train_new)
    df_test_new = holiday(df_test_new)
    print(f"df_train_new shape: {df_train_new.shape} , df_test_new shape: {df_test_new.shape}")
    
    # check correlation
    numeric_x = ['sales','onpromotion','dcoilwtico','transactions']
    numeric_info = df_train_new[numeric_x]
    num_correlation = numeric_info.corr(method='pearson')
    print(num_correlation)

    # plt.figure(figsize=(10,8))
    # sns.heatmap(num_correlation,annot=True)
    # print(sns.heatmap(num_correlation,annot=True))
    # num_correlation
    df_train_new = df_train_new.drop(columns='dcoilwtico')
    df_test_new = df_test_new.drop(columns='dcoilwtico')

    weekday_sales_means = df_train_new.groupby('weekday').agg({'sales':'mean'}).reset_index().sort_values(by='sales',ascending=False)
    print(weekday_sales_means)
    # sns.set()
    # plt.figure(figsize=(60,10))
    # sns.barplot(x=weekday_sales_means['weekday'], y=weekday_sales_means, color='r')
    # plt.legend()
    # plt.title('weekday: Comparson with Mean', fontsize=20)

    weekend_sales_means = df_train_new.groupby('weekend').agg({'sales':'mean'}).reset_index().sort_values(by='sales',ascending=False)
    print(weekend_sales_means)
    # sns.set()
    # plt.figure(figsize=(60,10))
    # sns.barplot(x=weekend_sales_means['weekend'], y=weekend_sales_means, color='r')
    # plt.legend()
    # plt.title('weekend: Comparison with Mean', fontsize=20)
    # drop weekday feature as it is similar to weekend
    df_train_new = df_train_new.drop(columns='weekday')
    df_test_new = df_test_new.drop(columns='weekday')

    dum_cat_col = ['store_nbr','family', 'locale', 'locale_name', 'city', 'state', 'store_type', 'cluster']
    train_data = pd.get_dummies(df_train_new, columns=dum_cat_col)
    test_data = pd.get_dummies(df_test_new, columns=dum_cat_col)
    print(train_data.head(3))

    # test data lack columns
    disappear_col_test = list(set(train_data.columns) - set(test_data.columns))
    print(disappear_col_test)
    disappear_col_test.remove('sales')
    for i in disappear_col_test:
        test_data[i] = 0
    col_order = train_data.columns.to_list()
    col_order.remove('sales')
    test_data = test_data[col_order]

    # x_feature = ['id','date','store_nbr','family','onpromotion','holiday_type','locale','locale_name','description','transferred',
    #           'city','state','store_type','cluster','transactions']
    # y_feature = 'sales'
    x = train_data.loc[:,'onpromotion':]
    y = train_data['sales']
    # k = the expected nums of features
    k_best = SelectKBest(f_regression,k=80).fit(x,y)
    keep_cols = k_best.get_feature_names_out().tolist()
    # filter needed columns
    filter_train_data_without_sales = train_data[keep_cols]
    filter_test_data = test_data[keep_cols]

    target = train_data['sales']
    from sklearn.model_selection import train_test_split
    x_train, x_test , y_train, y_test = train_test_split(filter_train_data_without_sales,target,test_size=0.2)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_log_error
    linear = LinearRegression()
    # model
    linear_model = linear.fit(x_train,y_train)
    # predic test data
    predictions = linear_model.predict(x_test)
    # estimate the RMSLE of prediction
    print("Valid RMSLE:", mean_squared_log_error(predictions,y_test)**0.5)

    res = linear_model.predict(filter_test_data)
    print(len(res))

    sample_submission = pd.read_csv("data/test/test.csv")
    sample_submission = sample_submission.drop(columns=['date','store_nbr','family','onpromotion'])
    print(sample_submission.shape)
    print(sample_submission.head())
    sample_submission['sales'] = res
    sample_submission.to_csv('./submission.csv',index=False)

    print("\nBuild model\n")

    # build_random_forest(x_feature,y_feature,df_train_new)
    print(f"{'-'*10} Forcasting {'-'*10}")
    # Forecast(x_feature,y_feature)
    print("done!")
