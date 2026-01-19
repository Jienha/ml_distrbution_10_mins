import pandas as pd
import numpy as np
import os
import warnings 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import code.dataset_preparation as tprep
from hybrid_model_karada import HybridModelKarada
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error, mean_absolute_error, mean_squared_error



# import dataframe and process it:
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

# set and normalize the multi-index timeseries:
ts_reindex = tprep.time_series_reindex(
    dataframe=ts, 
    date_col=DATE, 
    freq=FREQ, 
    extra_cols=EXTRA_COLS
)

# NaN value filling:
ts_reindex_filled = tprep.fillna_lop(ts_reindex, target_cols=ALL_TARGET_COLS, val=FILLNA_VAL, inplace=False)
# fillna_lop(ts_reindex, target_cols=ALL_TARGET_COLS, val=FILLNA_VAL, inplace=True)

# Add feature lags:
ts_ready = tprep.feature_lags(
    dataframe=ts_reindex_filled, 
    target=TARGET, 
    lags=[1,2,3,4,5,6,7, 14, 28], 
    groupby_columns=EXTRA_COLS, 
    inplace=False
)

# remove first n_records null:
ts_ready = ts_ready.dropna()

# filtered = ts_ready[(ts_ready['store_nbr'] == 1) & (ts_ready['family'].isin(ts_ready['family'].unique()[:10]))]
filtered = ts_ready[(ts_ready['store_nbr'] == 1) & (ts_ready['family'].isin(ts_ready['family'].unique()[5:6]))]
# print(filtered.sort_values(['store_nbr', 'family', 'date']))

unique_sorted_dates = filtered['date'].unique()
first_seventy_percent = int(unique_sorted_dates.shape[0]*0.7)
training_dates, test_dates = unique_sorted_dates[:first_seventy_percent], unique_sorted_dates[first_seventy_percent:]

train_cols = filtered.drop(['date', 'store_nbr', 'family', 'sales'], axis=1).columns
target_cols = 'sales'
y_all = filtered[[target_cols]]


train, test = filtered[filtered['date'].isin(training_dates)], filtered[filtered['date'].isin(test_dates)]

filtered['class'] = np.where(filtered['date'].isin(training_dates), 'train', 'test')

X_train, X_test = train[train_cols], test[train_cols]
y_train, y_test = train[[target_cols]], test[[target_cols]]
    

# training models:
lr = LinearRegression()
rfr = RandomForestRegressor(max_depth=3)

model = HybridModelKarada(lr, rfr, multioutput=False)
model.fit(X_train, X_train, y_train)
y_fit = model.predict(X_train, X_train).values
y_pred = model.predict(X_test, X_test).values

y_estim = np.concatenate([y_fit, y_pred])
filtered['model_pred'] = y_estim

filtered_train = filtered[filtered['class'] == 'train']
filtered_test = filtered[filtered['class'] == 'test']

plt.plot(filtered['sales'], alpha=0.5, label='original')
plt.plot(filtered_train['model_pred'],color='C0', label='train')
plt.plot(filtered_test['model_pred'],color='C3', label='test')
plt.legend()
plt.show()
