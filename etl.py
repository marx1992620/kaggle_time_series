# import numpy as np
import pandas as pd
# import os
# import matplotlib.pyplot as plt



def merge_df():
    df = pd.read_csv('data/train/oil.csv')
    df_oil_fill = df.fillna(method="pad")

    df_train = pd.read_csv("data/train/train.csv")
    df_test = pd.read_csv("data/test/test.csv")

    df_holidays_events = pd.read_csv("data/train/holidays_events.csv")
    df_holidays_events.rename(columns={'date':'date',
                             'type':'Daily_holiday_type',
                             'locale':'Daily_holiday_locale',
                             'locale_name':'Daily_holiday_locale_name',
                             'description':'Daily_holiday_description',
                             'transferred':'Daily_holiday_transferred'},
                             inplace=True)

    df_stores = pd.read_csv("data/train/stores.csv")
    df_stores.rename(columns={'store_nbr':'store_nbr',
                             'city':'store_city',
                             'state':'store_state',
                             'type':'store_type',
                             'cluster':'store_cluster'},
                             inplace=True)
    df_transactions = pd.read_csv("data/train/transactions.csv")
    df_transactions.rename(columns={'transactions':'Daily_transactions'})

    df_train_new = pd.merge(df_train,df_holidays_events,how='left',left_on='date',right_on='date')
    df_train_new = pd.merge(df_train_new,df_oil_fill,how='left',left_on='date',right_on='date')
    df_train_new = pd.merge(df_train_new,df_stores,how='left',left_on='store_nbr',right_on='store_nbr')
    df_train_new = pd.merge(df_train_new,df_transactions,how='left',on=['date','store_nbr'])

    df_test_new = pd.merge(df_test,df_holidays_events,how='left',left_on='date',right_on='date')
    df_test_new = pd.merge(df_test_new,df_oil_fill,how='left',left_on='date',right_on='date')
    df_test_new = pd.merge(df_test_new,df_stores,how='left',left_on='store_nbr',right_on='store_nbr')
    df_test_new = pd.merge(df_test_new,df_transactions,how='left',on=['date','store_nbr'])

    return df_train_new, df_test_new


def check_na(df):
    print(f"length: {len(df)}")
    for i in df.columns:
        a = df[i].describe()
        print(f"Name: {i} Rate: {100*a['count']/len(df[i])}")


def prework():
    df_train_new.drop_duplicates(subset='id',keep='first',inplace=True)
    df_train_new.dropna(axis=0,inplace=True)
    df_train_new['date'] = df_train_new['date'].apply(lambda X: int(str(X).split('-')[0] + str(X).split('-')[1] + str(X).split('-')[2]))
    df_train_new['family'] = pd.factorize(df_train_new['family'])[0].astype(int)
    df_train_new['Daily_holiday_type'] = pd.factorize(df_train_new['Daily_holiday_type'])[0].astype(int)
    df_train_new['Daily_holiday_locale'] = pd.factorize(df_train_new['Daily_holiday_locale'])[0].astype(int)
    df_train_new['Daily_holiday_locale_name'] = pd.factorize(df_train_new['Daily_holiday_locale_name'])[0].astype(int)
    df_train_new['Daily_holiday_description'] = pd.factorize(df_train_new['Daily_holiday_description'])[0].astype(int)
    df_train_new['Daily_holiday_transferred'] = pd.factorize(df_train_new['Daily_holiday_transferred'])[0].astype(int)
    df_train_new['store_city'] = pd.factorize(df_train_new['store_city'])[0].astype(int)
    df_train_new['store_state'] = pd.factorize(df_train_new['store_state'])[0].astype(int)
    df_train_new['store_type'] = pd.factorize(df_train_new['store_type'])[0].astype(int)

    df_test_new.drop_duplicates(subset='id',keep='first',inplace=True)
    df_test_new.dropna(axis=0,inplace=True)
    df_test_new['date'] = df_test_new['date'].apply(lambda X: int(str(X).split('-')[0] + str(X).split('-')[1] + str(X).split('-')[2]))
    df_test_new['family'] = pd.factorize(df_test_new['family'])[0].astype(int)
    df_test_new['Daily_holiday_type'] = pd.factorize(df_test_new['Daily_holiday_type'])[0].astype(int)
    df_test_new['Daily_holiday_locale'] = pd.factorize(df_test_new['Daily_holiday_locale'])[0].astype(int)
    df_test_new['Daily_holiday_locale_name'] = pd.factorize(df_test_new['Daily_holiday_locale_name'])[0].astype(int)
    df_test_new['Daily_holiday_description'] = pd.factorize(df_test_new['Daily_holiday_description'])[0].astype(int)
    df_test_new['Daily_holiday_transferred'] = pd.factorize(df_test_new['Daily_holiday_transferred'])[0].astype(int)
    df_test_new['store_city'] = pd.factorize(df_test_new['store_city'])[0].astype(int)
    df_test_new['store_state'] = pd.factorize(df_test_new['store_state'])[0].astype(int)
    df_test_new['store_type'] = pd.factorize(df_test_new['store_type'])[0].astype(int)


if __name__ == "__main__":
    print("======== start ========")
    df_train_new,df_test_new = merge_df()
    prework()
    # print("----------- df_train_new -----------")
    # check_na(df_train_new)
    # print("----------- df_test_new ------------")
    # check_na(df_test_new)
    print("done!")
