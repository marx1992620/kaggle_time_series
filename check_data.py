import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


for dirname, _, filenames in os.walk('data/train'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def draw(x,y):
    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.plot(x,y)
    plt.show()


df = pd.read_csv("data/test/test.csv")
print("test:",df.shape)
print(df.head())
print("++++++++++++++++++++++")
# print(df.isnull().sum())
print(df.describe())
print("================")


df = pd.read_csv("data/train/train.csv")
print("train:",df.shape)
print(df.head())
print("++++++++++++++++++++++")
# print(df.isnull().sum())
print(df.describe())
print("================")


df = pd.read_csv('data/train/oil.csv')
print("oil:",df.shape)
print(df.head())
print("++++++++++++++++++++++")
# print(df.isnull().sum())
print(df.describe())
# print("------------------")
# print(df[df.dcoilwtico.isnull()])
print("================")
# draw(df['date'],df['dcoilwtico'])
df_fill = df.fillna(method="pad")
# draw(df_fill['date'],df_fill['dcoilwtico'])


df = pd.read_csv('data/train/stores.csv')
print("stores:",df.shape)
print(df.head())
print("++++++++++++++++++++++")
# print(df.isnull().sum())
print(df.describe())
print("================")

df = pd.read_csv('data/train/transactions.csv')
print("transactions:",df.shape)
print(df.head())
print("++++++++++++++++++++++")
# print(df.isnull().sum())
print(df.describe())
print("================")


df = pd.read_csv('data/train/holidays_events.csv')
print("holidays_events:",df.shape)
print(df.head())
print("++++++++++++++++++++++")
# print(df.isnull().sum())
print(df.describe())
print("================")

# ax=df.plot.kde()
