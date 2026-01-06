import pandas as pd
import numpy as np
import os
import warnings 
warnings.filterwarnings('ignore')


### Reading Data:
path = r'C:\Users\piers\Desktop\home\Sviluppo\ml_distrbution_10_mins\data\train.csv'

df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])

# dates range:
all_dates = pd.date_range(
    start=df['date'].min(),
    end=df['date'].max(),
    freq='D'
)

class TimeSeriesFormat:
    def __init__(self, data, date_col):
        self.data = data
        self.date_col = date_col


    def reindex(self, index_cols:list):

        all_dates = pd.date_range(
            start=self.data[self.date_col].min(),
            end=self.data[self.date_col].max(),
            freq='D'
        )

        all_uniques_values = [
            self.data[col_name].unique()
            for col_name in index_cols
        ]
        
        full_index = pd.MultiIndex.from_product([all_dates, *all_uniques_values], names=[self.date_col, *index_cols])
        
        return (
            self.data
            .set_index([self.date_col, *index_cols]) # set index
            .reindex(full_index) # reindex
            .reset_index()
        )

    def fillna(self, index_cols:list):
        
    
    def to_category(self):
        pass

    def add_lags(self):
        pass

    def basic_transformation(self):
        pass


# all stores:
all_stores = df['store_nbr'].unique()

# all families of product
all_families = df['family'].unique()

# create a multiIndex as a product of dates/store/family:
full_index = pd.MultiIndex.from_product(
    [all_dates, all_stores, all_families],
    names=['date', 'store_nbr', 'family']
)

# Reindex the dataframe with the new index:
df = (
    df
    .set_index(['date', 'store_nbr', 'family']) # set index
    .reindex(full_index) # reindex
    .reset_index()
)

# fillna con 0:
df['sales'] = df['sales'].fillna(0)
df['onpromotion'] = df['onpromotion'].fillna(0)

# convert into category value:
df['store_nbr_id'] = df['store_nbr'].astype('category').cat.codes
df['family_id'] = df['family'].astype('category').cat.codes

# sales lags prev 1-7-14-28 days:
for lag in [1, 7, 14, 28]:
    df[f'sales_lag_{lag}'] = df.groupby(['store_nbr_id','family_id'])['sales'].shift(lag)

# onpromotion lags prev 1-7-14-28 days:
for lag in [1, 7, 14, 28]:
    df[f'onpromotion_lag_{lag}'] = df.groupby(['store_nbr_id','family_id'])['onpromotion'].shift(lag)

print(df.tail())
