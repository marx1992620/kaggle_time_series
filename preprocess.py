# import numpy as np
import pandas as pd
# import os
import matplotlib.pyplot as plt

def draw(x,y):
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.plot(x,y)
    plt.show()


def merge_df():
    df = pd.read_csv('data/train/oil.csv')
    # print(df.isna().sum())
    # draw(df['date'],df['dcoilwtico'])
    df_oil_fill = df.fillna(method="backfill")

    df_train = pd.read_csv("data/train/train.csv")
    df_test = pd.read_csv("data/test/test.csv")
    df_holidays_events = pd.read_csv("data/train/holidays_events.csv")
    df_holidays_events.rename(columns={'type':'holiday_type'},inplace=True)
    df_stores = pd.read_csv("data/train/stores.csv")
    df_stores.rename(columns={'type':'store_type'},inplace=True)
    df_transactions = pd.read_csv("data/train/transactions.csv")

    # Converting the 'date' column in the datasets to datetime format
    # Train dataset
    df_train['date'] = pd.to_datetime(df_train['date'])

    # Test dataset
    df_test['date'] = pd.to_datetime(df_test['date'])

    # Holiday Events dataset
    df_holidays_events['date'] = pd.to_datetime(df_holidays_events['date'])

    # Oil dataset
    df_oil_fill['date'] = pd.to_datetime(df_oil_fill['date'])

    # Transactions dataset
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    # Confirm the data type of the 'date' column after transformation
    print('Date Column Data Type After Transformation:') 
    print('='*30)
    print("Train dataset:", df_train['date'].dtype)
    print("Test dataset:", df_test['date'].dtype)
    print("Holiday Events dataset:", df_holidays_events['date'].dtype)
    print("Oil dataset:", df_oil_fill['date'].dtype)
    print("Transactions dataset:", df_transactions['date'].dtype)

    min_date = df_train['date'].min()
    max_date = df_train['date'].max()
    expected_dates = pd.date_range(start=min_date, end=max_date)
    missing_dates = expected_dates[~expected_dates.isin(df_train['date'])]
    if len(missing_dates) == 0:
        print("The date data is complete.")
    else:
        print("The date data is incomplete. missing dates:")
        print(missing_dates)

    # Check for missing values in the datasets
    datasets = {'train': df_train, 'test': df_test, 'holiday events': df_holidays_events, 'oil': df_oil_fill, 'stores': df_stores, 'transactions': df_transactions, }

    # def show_missing_values(datasets):
    #     for name, data in datasets.items():
    #         print(f"Missing values in the {name.capitalize()} dataset:")
    #         print(data.isnull().sum())
    #         print('=' * 30)
    #         print()

    # show_missing_values(datasets)
    # # Complete the missing dates in the train dataset
    # # Create an index of the missing dates as a DatetimeIndex object
    # missing_dates = pd.Index(['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25'], dtype='datetime64[ns]')
    # Create a DataFrame with the missing dates, using the 'date' column
    missing_data = pd.DataFrame({'date': missing_dates})
    # Concatenate the original train dataset and the missing data DataFrame
    # ignore_index=True ensures a new index is assigned to the resulting DataFrame
    df_train = pd.concat([df_train, missing_data], ignore_index=True)
    # Sort the DataFrame based on the 'date' column in ascending order
    df_train.sort_values('date', inplace=True)

    min_date = df_train['date'].min()
    max_date = df_train['date'].max()
    expected_dates = pd.date_range(start=min_date, end=max_date)
    missing_dates = expected_dates[~expected_dates.isin(df_train['date'])]
    if len(missing_dates) == 0:
        print("The date data is complete.")



    # train data
    df_train_new = pd.merge(df_train,df_oil_fill,how='left',on='date')
    # draw(df_train_new['date'],df_train_new['dcoilwtico'])
    df_train_new = df_train_new.fillna(method='pad')
    # draw(df_train_new['date'],df_train_new['dcoilwtico'])
    df_train_new = pd.merge(df_train_new,df_holidays_events,how='left',on='date')
    # print(df_train_new.isna().sum())
    df_train_new = df_train_new.fillna("Empty")
    # print(df_train_new.isna().sum())
    # df_train_new.info(show_counts=True)
    df_train_new = pd.merge(df_train_new,df_stores,how='left',on='store_nbr')
    # print(df_train_new.isna().sum())
    df_train_new = pd.merge(df_train_new,df_transactions,how='left',on=['date','store_nbr'])
    # print(df_train_new.isna().sum())
    df_train_new['transactions'] = df_train_new['transactions'].fillna(0)
    # print(df_train_new.isna().sum())
    df_train_new['date'] = df_train_new['date'].astype('datetime64[ns]')
    df_train_new['date'].dtype
    print("="*10,"train data","="*10)
    print(df_train_new.isna().sum())
    df_train_new.head()

    # test data
    print("="*10,"test data","="*10)
    df_test_new = pd.merge(df_test,df_oil_fill,how='left',on='date')
    # print(df_test_new.isna().sum())
    # draw(df_test_new['date'],df_test_new['dcoilwtico'])
    df_test_new = df_test_new.fillna(method='backfill')
    # draw(df_test_new['date'],df_test_new['dcoilwtico'])
    # print(df_test_new.isna().sum())
    df_test_new = pd.merge(df_test_new,df_holidays_events,how='left',on='date')
    df_test_new = df_test_new.fillna("Empty")
    
    df_test_new = pd.merge(df_test_new,df_stores,how='left',left_on='store_nbr',right_on='store_nbr')
    df_test_new = pd.merge(df_test_new,df_transactions,how='left',on=['date','store_nbr'])
    df_test_new['transactions'] = df_test_new['transactions'].fillna(0)
    df_test_new['date'] = df_test_new['date'].astype('datetime64[ns]')
    df_test_new['date'].dtype
    print(df_test_new.isna().sum())

    return df_train_new, df_test_new


def check_na(df):
    print(f"length: {len(df)}")
    for i in df.columns:
        a = df[i].describe()
        print(f"Name: {i} Rate: {100*a['count']/len(df[i])}")


def split_date(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df.loc[df['weekday'] < 5,'weekend'] = 0
    df.loc[df['weekday'] >= 5,'weekend'] = 1
    return df


def holiday(df):
    df['is_holiday'] = 0
    df.loc[df.weekday > 4 , 'is_holiday'] = 1
    df.loc[df['holiday_type'] == 'Work Day', 'is_holiday'] = 0
    df.loc[df['holiday_type'] == 'Transfer', 'is_holiday'] = 1
    df.loc[df['holiday_type'] == 'Bridge', 'is_holiday'] = 1
    df.loc[(df['holiday_type'] == 'Holiday') & (df.transferred == False), 'is_holiday'] = 1
    df.loc[(df['holiday_type'] == 'Holiday') & (df.transferred == True), 'is_holiday'] = 0
    df = df.drop(columns=['holiday_type','description','transferred'])
    return df

def prework(df_train_new,df_test_new):
    # df_train_new.drop_duplicates(subset='id',keep='first',inplace=True)
    # df_train_new.dropna(axis=0,inplace=True)
    # df_train_new['date'] = df_train_new['date'].apply(lambda X: int(str(X).split('-')[0] + str(X).split('-')[1] + str(X).split('-')[2]))

    df_train_new['family'] = pd.factorize(df_train_new['family'])[0].astype(int)
    df_train_new['holiday_type'] = pd.factorize(df_train_new['holiday_type'])[0].astype(int)
    df_train_new['locale'] = pd.factorize(df_train_new['locale'])[0].astype(int)
    df_train_new['locale_name'] = pd.factorize(df_train_new['locale_name'])[0].astype(int)
    df_train_new['description'] = pd.factorize(df_train_new['description'])[0].astype(int)
    df_train_new['transferred'] = pd.factorize(df_train_new['transferred'])[0].astype(int)
    df_train_new['city'] = pd.factorize(df_train_new['city'])[0].astype(int)
    df_train_new['state'] = pd.factorize(df_train_new['state'])[0].astype(int)
    df_train_new['store_type'] = pd.factorize(df_train_new['store_type'])[0].astype(int)

    # df_test_new.drop_duplicates(subset='id',keep='first',inplace=True)
    # df_test_new.dropna(axis=0,inplace=True)
    # df_test_new['date'] = df_test_new['date'].apply(lambda X: int(str(X).split('-')[0] + str(X).split('-')[1] + str(X).split('-')[2]))
    df_test_new['family'] = pd.factorize(df_test_new['family'])[0].astype(int)
    df_test_new['holiday_type'] = pd.factorize(df_test_new['holiday_type'])[0].astype(int)
    df_test_new['locale'] = pd.factorize(df_test_new['locale'])[0].astype(int)
    df_test_new['locale_name'] = pd.factorize(df_test_new['locale_name'])[0].astype(int)
    df_test_new['description'] = pd.factorize(df_test_new['description'])[0].astype(int)
    df_test_new['transferred'] = pd.factorize(df_test_new['transferred'])[0].astype(int)
    df_test_new['city'] = pd.factorize(df_test_new['city'])[0].astype(int)
    df_test_new['state'] = pd.factorize(df_test_new['state'])[0].astype(int)
    df_test_new['store_type'] = pd.factorize(df_test_new['store_type'])[0].astype(int)

    return df_train_new, df_test_new


if __name__ == "__main__":
    print(f"{'='*10} start {'='*10}")
    df_train_new,df_test_new = merge_df()
    df_train_new,df_test_new = prework(df_train_new,df_test_new)
    # print(f"{'-'*10} df_train_new {'-'*10}")
    # check_na(df_train_new)
    # print(f"{'-'*10} df_test_new {'-'*10}")
    # check_na(df_test_new)
    # print("done!")
