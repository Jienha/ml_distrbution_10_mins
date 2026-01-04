import pandas as pd
import numpy as np
import os
import warnings 
warnings.filterwarnings('ignore')

path = r'C:\Users\piers\Desktop\home\Sviluppo\ml_distrbution_10_mins\data\train.csv'

df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])

# all dates:
all_dates = pd.date_range(
    start=df['date'].min(),
    end=df['date'].max(),
    freq='D'
)

all_stores = df['store_nbr'].unique()

all_families = df['family'].unique()

full_index = pd.MultiIndex.from_product(
    [all_dates, all_stores, all_families],
    names=['date', 'store_nbr', 'family']
)

df = (
    df
    .set_index(['date', 'store_nbr', 'family'])
    .reindex(full_index)
    .reset_index()
)

# fillna con 0:
df['sales'] = df['sales'].fillna(0)
df['onpromotion'] = df['onpromotion'].fillna(0)


print(df.groupby(['store_nbr', 'family'])['date'].nunique().describe())



df['store_nbr_id'] = df['store_nbr'].astype('category').cat.codes
df['family_id'] = df['family'].astype('category').cat.codes

for lag in [1, 7, 14, 28]:
    df[f'sales_lag_{lag}'] = df.groupby(['store_nbr_id','family_id'])['sales'].shift(lag)

for lag in [1, 7, 14, 28]:
    df[f'onpromotion_lag_{lag}'] = df.groupby(['store_nbr_id','family_id'])['onpromotion'].shift(lag)

print(df.tail())
