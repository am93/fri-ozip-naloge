import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ========================
# Get the data
# ========================
# From here: https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity/notebook
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

df_train = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("data/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("data/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

df_train.head()

# ========================
# ylog will be log(1+y), as suggested by https://github.com/dmlc/xgboost/issues/446#issuecomment-135555130
# ========================
ylog_train_all = np.log1p(df_train['price_doc'].values)
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# ========================
# Build df_all = (df_train+df_test).join(df_macro)
# ========================
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)
df_all['month_year'] = month_year


# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)
df_all.drop(['market_shop_km'], axis=1, inplace=True)
df_all.drop(['green_part_5000'], axis=1, inplace=True)

# new
#df_all['rate_1'] = df_all['mortgage_rate'] - df_all['deposits_rate']

# ========================
# Deal with categorical values
# ========================
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

df_values.fillna(0)

# ========================
# Convert to numpy values
# ========================
X_all = df_values.values
print(X_all.shape)

# ========================
# Create a validation set, with last 20% of data
# ========================
X_train = X_all[:num_train]
X_train = np.nan_to_num(X_train)
X_test = X_all[num_train:]
X_test = np.nan_to_num(X_test)

df_columns = df_values.columns


rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rf.fit(X_train, ylog_train_all)
ylog_pred = rf.predict(X_test)

y_pred = np.exp(ylog_pred) - 1
y_pred = y_pred * 1.01

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('rf_sub.csv', index=False)

