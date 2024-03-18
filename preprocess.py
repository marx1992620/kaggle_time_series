import pandas as pd
import matplotlib.pyplot as plt


def draw(x,y):
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.plot(x,y)
    plt.show()


def show_missing_values(dataframe):
    for df_name, data in dataframe.items():
        print('=' * 30)
        print(f"Check missing values of {df_name} dataset:")
        print(data.isnull().sum())


def check_dates(dataframe):
    min_date = dataframe['date'].min()
    max_date = dataframe['date'].max()
    expected_dates = pd.date_range(start=min_date, end=max_date)
    missing_dates = expected_dates[~expected_dates.isin(dataframe['date'])]
    if len(missing_dates) == 0:
        print("The date data is complete.")
    else:
        print("The date data is incomplete. missing dates:")
        print(missing_dates)
    # The date data is incomplete. missing dates:
    # DatetimeIndex(['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25'], dtype='datetime64[ns]', freq=None)


def insert_dates(dataframe):
    min_date = dataframe['date'].min()
    max_date = dataframe['date'].max()
    expected_dates = pd.date_range(start=min_date, end=max_date)
    missing_dates = expected_dates[~expected_dates.isin(dataframe['date'])]
    # Create an index of the missing dates as a DatetimeIndex object
    # Create a DataFrame with the missing dates, using the 'date' column
    missing_data = pd.DataFrame({'date': missing_dates})
    # Concatenate the original train dataset and the missing data DataFrame
    # ignore_index=True ensures a new index is assigned to the resulting DataFrame
    dataframe = pd.concat([dataframe, missing_data], ignore_index=True)
    # Sort the DataFrame based on the 'date' column in ascending order
    dataframe.sort_values('date', inplace=True)

    return dataframe


def merge_df():

    df_train = pd.read_csv("data/train/train.csv")
    df_test = pd.read_csv("data/test/test.csv")
    df_oil = pd.read_csv('data/train/oil.csv')
    df_holidays_events = pd.read_csv("data/train/holidays_events.csv")
    df_stores = pd.read_csv("data/train/stores.csv")
    df_transactions = pd.read_csv("data/train/transactions.csv")

    # Check for missing values in the datasets
    datasets_1 = {'train': df_train, 'test': df_test, 'holiday events': df_holidays_events, 'oil': df_oil, 'stores': df_stores, 'transactions': df_transactions}
    show_missing_values(datasets_1)

    # fill df_oil missing data
    draw(df_oil['date'],df_oil['dcoilwtico'])
    df_oil = df_oil.fillna(method="backfill")
    draw(df_oil['date'],df_oil['dcoilwtico'])

    # Converting the 'date' column in the datasets to datetime format
    datasets_2 = {'train': df_train, 'test': df_test, 'holiday events': df_holidays_events, 'oil': df_oil, 'stores': df_stores, 'transactions': df_transactions}
    for df_name, dataframe in datasets_2.items():
        if 'date' in dataframe.columns:
            dataframe['date'] = pd.to_datetime(dataframe['date'])
            print(f"{df_name} date: {dataframe['date'].dtype}")

    # check date data
    check_dates(df_train)
    # insert dates
    df_train = insert_dates(df_train)
    check_dates(df_train)

   # rename the type column
    df_holidays_events.rename(columns={'type':'holiday_type'},inplace=True)
    df_stores.rename(columns={'type':'store_type'},inplace=True)
    # merge train dataframe
    merged_df = df_train.merge(df_stores,on='store_nbr',how='inner')
    merged_df = merged_df.merge(df_transactions,on=['date','store_nbr'],how='inner')
    merged_df = merged_df.merge(df_holidays_events,on='date',how='inner')
    merged_df = merged_df.merge(df_oil,on='date',how='inner')

    datasets_3 = {'merged_df': merged_df, 'test': df_test}
    show_missing_values(datasets_3)
    print(f"merged train df shape:{merged_df.shape}")
    print(merged_df.head())
    print()
    print(f"merged test df shape:{df_test.shape}")
    print(df_test.head())
    print()
    print(merged_df.describe().T)

    merged_df.to_csv('data/train/merged_df.csv',index=False)
    df_test.to_csv('data/test/df_test.csv',index=False)

    return merged_df, df_test, df_oil, df_holidays_events, df_stores, df_transactions


if __name__ == "__main__":
    merged_df, df_test, df_oil, df_holidays_events, df_stores, df_transactions = merge_df()

