import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import sys, getopt

##########################################################
# ---- OPTIONS PARSING --------------------------------
##########################################################
SELECTED_MODELS = ''
ADDITIONAL_VALUE = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], 'hm:v:', ['model=', 'value='])
except getopt.GetoptError:
    print('Error during options parsing...')
    sys.exit(2)

for opt, arg in opts:
    if opt in ('-m', '--model'):
        SELECTED_MODELS = arg
    elif opt in ('-v', '--value'):
        ADDITIONAL_VALUE = int(arg)
    elif opt == '-h':
        print('Available models: randomforest')

##########################################################
# ---- DATA ANALYSIS --------------------------------
##########################################################

##########################################################
# ---- DATA PREPROCESSING --------------------------------
##########################################################

# read input data using pandas
df_train = pd.read_csv("data/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("data/test.csv", parse_dates=['timestamp'])

print(datetime.now(), ' Data read successfully...')

# store train true values and ids for report
y_train = df_train['price_doc']
id_test = df_test['id']

# remove unneeded attributes
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# concat train and test to simplify data preprocessing
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])

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

# handle objects - factorize
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

# handle NAN, INF and > float32 values - sklearn is working internally with float32
df_values = pd.concat([df_numeric, df_obj], axis=1)
df_values.fillna(0)
df_values.info()
df_values = df_values.astype(np.float32)
df_values.info()

print(datetime.now(), ' Data preprocessing finished...')

# get values and split them in train and test back
X_all = df_values.values
X_train = X_all[:num_train]
X_train = np.float32(np.nan_to_num(X_train))
X_test = X_all[num_train:]
X_test = np.float32(np.nan_to_num(X_test))
y_train = np.float32(y_train)

##########################################################
# ---- RANDOM FOREST MODEL -------------------------------
##########################################################
if 'randomforest' in SELECTED_MODELS:
    num_estimators = ADDITIONAL_VALUE

    print(datetime.now(), ' Random forest model with ',num_estimators, ' estimators will be fitted...')

    rf = RandomForestRegressor(n_estimators=num_estimators, n_jobs=-1)
    rf.fit(X_train, y_train)

    print(datetime.now(), ' Random forest model fitted...')

    y_pred = rf.predict(X_test)

    print(datetime.now(), ' Random forest predicitions completed')

    df_predictions = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
    df_predictions.to_csv(str(num_estimators).join(['predictions/rf_basic_','.csv']), index=False)

