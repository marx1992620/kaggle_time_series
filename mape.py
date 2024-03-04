import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from etl import merge_df, check_na, prework


def build_random_forest(x_data,y_data,data_frame):
    x_ = data_frame[x_data]
    y_ = data_frame[y_data]
    model = RandomForestRegressor(random_state=1,max_depth=100)
    a_x,b_x,a_y,b_y = train_test_split(x_,y_,random_state=1)
    model.fit(a_x,a_y)
    predictions = model.predict(b_x)
    delta = mean_absolute_error(b_y,predictions)
    print(f"mean_absolute_error delta: {delta}")


def Forecast():
    model = RandomForestRegressor(random_state=1,max_depth=100)
    model.fit(df_train_new(x),df_train_new(y))
    aim = df_test_new(x)
    predictions = model.predict(aim)
    res = pd.DataFrame(predictions)
    path = './submission.csv'
    res.to_csv(path)


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
    print("======== start ========")
    df_train_new,df_test_new = merge_df()
    print(f"df_train_new shape: {df_train_new.shape} , df_test_new shape: {df_test_new.shape}")
    x_feature = ['id','date','store_nbr','family','onpromotion','Daily_holiday_type','Daily_holiday_locale','Daily_holiday_locale_name','Daily_holiday_description','Daily_holiday_transferred',
              'dcoilwtico','store_city','store_state','store_type','store_cluster','transactions']
    y_feature = 'sales'
    df_train_new2 , df_test_new2 = prework(df_train_new,df_test_new)
    print(f"df_train_new2 shape: {df_train_new2.shape} , df_test_new2 shape: {df_test_new2.shape}")
    print("----------- df_train_new -----------")
    check_na(df_train_new2)
    print("----------- df_test_new ------------")
    check_na(df_test_new2)
    build_random_forest(x_feature,y_feature,df_train_new)
    print("------ forcasting ------")
    Forecast(x_feature,y_feature)
    print("done!")
