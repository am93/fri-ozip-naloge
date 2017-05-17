import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# ========================
# Get the data
# ========================
# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

df_train = pd.read_csv("train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

df_train.head()

# ========================
# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
# ========================
ylog_train_all = np.log1p(df_train['price_doc'].values)
id_test = df_test['id']

#df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
#df_test.drop(['id'], axis=1, inplace=True)

# ========================
# Build df_all = (df_train+df_test).join(df_macro)
# ========================
num_train = len(df_train)

df_all_train = pd.merge_ordered(df_train, df_macro, on='timestamp', how='left')
df_all_test = pd.merge_ordered(df_test, df_macro, on='timestamp', how='left')


# Add month-year
month_year = (df_all_train.timestamp.dt.month + df_all_train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all_train['month_year_cnt'] = month_year.map(month_year_cnt_map)
df_all_train['month_year'] = month_year

month_year = (df_all_test.timestamp.dt.month + df_all_test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all_test['month_year_cnt'] = month_year.map(month_year_cnt_map)
df_all_test['month_year'] = month_year


# Add week-year count
week_year = (df_all_train.timestamp.dt.weekofyear + df_all_train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all_train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (df_all_test.timestamp.dt.weekofyear + df_all_test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all_test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all_train['month'] = df_all_train.timestamp.dt.month
df_all_train['dow'] = df_all_train.timestamp.dt.dayofweek

df_all_test['month'] = df_all_test.timestamp.dt.month
df_all_test['dow'] = df_all_test.timestamp.dt.dayofweek

# Other feature engineering
df_all_train['rel_floor'] = df_all_train['floor'] / df_all_train['max_floor'].astype(float)
df_all_train['rel_kitch_sq'] = df_all_train['kitch_sq'] / df_all_train['full_sq'].astype(float)

df_all_test['rel_floor'] = df_all_test['floor'] / df_all_test['max_floor'].astype(float)
df_all_test['rel_kitch_sq'] = df_all_test['kitch_sq'] / df_all_test['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all_train.drop(['timestamp'], axis=1, inplace=True)
df_all_train.drop(['market_shop_km'], axis=1, inplace=True)
df_all_train.drop(['green_part_5000'], axis=1, inplace=True)

df_all_test.drop(['timestamp'], axis=1, inplace=True)
df_all_test.drop(['market_shop_km'], axis=1, inplace=True)
df_all_test.drop(['green_part_5000'], axis=1, inplace=True)


# ========================
# Deal with categorical values
# ========================
df_numeric = df_all_train.select_dtypes(exclude=['object'])
df_obj = df_all_train.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values_train = pd.concat([df_numeric, df_obj], axis=1)

df_numeric = df_all_test.select_dtypes(exclude=['object'])
df_obj = df_all_test.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values_test = pd.concat([df_numeric, df_obj], axis=1)


# save
df_values_train.to_csv('train_processed.csv', index=False)
df_values_test.to_csv('test_processed.csv', index=False)