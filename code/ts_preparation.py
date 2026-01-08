import pandas as pd
import os
import warnings 
from pathlib import Path

warnings.filterwarnings('ignore')


def time_series_reindex(dataframe:pd.DataFrame, date_col:str, freq:str='D', extra_cols:list=None) -> pd.DataFrame:
    """Standardize multi-index time series dataframe.

    Args:
        dataframe (pd.DataFrame): Time series dataframe
        date_col (str): Date column's name
        freq (str, optional): Date frequecy - Daily/Weekly/Monthly/Hourly. Defaults to 'D' = daily.
        extra_cols (list, optional): List of others extra columns used for creating the index. Defaults to None.

    Returns:
        pd.DataFrame:  Time series reindex dataframe
    """
    all_dates = pd.date_range(
            start=dataframe[date_col].min(),
            end=dataframe[date_col].max(),
            freq=freq
        )

    if extra_cols != None: 
        # All unique values for each extra columns:
        all_uniques_values = [dataframe[col].unique() for col in extra_cols]
        # Unique index as a product of all unique values
        full_index = pd.MultiIndex.from_product([all_dates, *all_uniques_values], names=[date_col, *extra_cols])
        
        return (
            dataframe
            .set_index([date_col, *extra_cols]) # set index
            .reindex(full_index) # reindex
            .reset_index()
        )
    
    else:
        full_index = all_dates
        return (
            dataframe
            .set_index([date_col]) # set index
            .reindex(full_index) # reindex
            .reset_index()
        )
    
def fillna_lop(dataframe:pd.DataFrame, target_cols:list, val=0, inplace=False) -> pd.DataFrame:
        if inplace:
            for col in target_cols:
                dataframe[col] = dataframe[col].fillna(val) 
            return
        else:
            dataframe_copy = dataframe.copy()
            for col in target_cols:
                dataframe_copy[col] = dataframe_copy[col].fillna(val) 
            return dataframe_copy
        
def feature_lags(dataframe:pd.DataFrame, target:str, lags:list=[1], groupby_columns:list=None, inplace=False) -> pd.DataFrame:
    if inplace:
        if groupby_columns != None:
            for lag in lags:
                dataframe[target+'_lag_{}'.format(lag)] = dataframe.groupby(groupby_columns)[target].shift(lag)
        else:
            for lag in lags:
                dataframe[target+'_lag_{}'.format(lag)] = dataframe[target].shift(lag)
        return

    else:
        dataframe_copy = dataframe.copy()
        if groupby_columns != None:
            for lag in lags:
                dataframe_copy[target+'_lag_{}'.format(lag)] = dataframe_copy.groupby(groupby_columns)[target].shift(lag)
        else:
            for lag in lags:
                dataframe_copy[target+'_lag_{}'.format(lag)] = dataframe_copy[target].shift(lag)

        return dataframe_copy
    

if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    # standardize praject path:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    DATA_TRAIN_PATH = os.path.join(PROJECT_ROOT,"data","train.csv")

    # reading dataset:
    ts = pd.read_csv(DATA_TRAIN_PATH)
    ts['date'] = pd.to_datetime(ts['date'])

    EXTRA_COLS = ['store_nbr', 'family']
    DATE = 'date'
    FREQ = 'D'
    TARGET = 'sales'
    ALL_TARGET_COLS = ['sales', 'onpromotion']
    FILLNA_VAL = 0

    ts_reindex = time_series_reindex(
        dataframe=ts, 
        date_col=DATE, 
        freq=FREQ, 
        extra_cols=EXTRA_COLS
    )

    ts_reindex_filled = fillna_lop(ts_reindex, target_cols=ALL_TARGET_COLS, val=FILLNA_VAL, inplace=False)
    # fillna_lop(ts_reindex, target_cols=ALL_TARGET_COLS, val=FILLNA_VAL, inplace=True)

    ts_ready = feature_lags(
        dataframe=ts_reindex_filled, 
        target=TARGET, 
        lags=[1,2,3,9], 
        groupby_columns=EXTRA_COLS, 
        inplace=False
    )


    filtered = ts_ready[(ts_ready['store_nbr'] == 1) & (ts_ready['family'].isin(ts_ready['family'].unique()[:10]))]
    # print(filtered.sort_values(['store_nbr', 'family', 'date']))
    
    sns.lineplot(x="date", y="sales",
             hue="family", style="family",
             data=filtered)
    plt.show()


