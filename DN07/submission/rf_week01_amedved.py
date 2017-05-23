# My Kaggle username: anzzemedved
# My score: 0.31763

import numpy as np
import pandas as pd
from datetime import datetime
import sys, getopt
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

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
        print('Available models: randomforest, gradientboosting, mlpregressor')

if SELECTED_MODELS == '':
    print('No models were selected ! Use -h options to display available models...')

##########################################################
# ---- DATA ANALYSIS --------------------------------
##########################################################
# read input data using pandas
df_train = pd.read_csv("../data/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("../data/test.csv", parse_dates=['timestamp'])

print(datetime.now(), ' Data read successfully...')

# get column names
print(df_train.columns.tolist())

# Price distribution
plt.figure()
plt.hist(df_train.price_doc.values/1000, bins=30)
plt.xlabel('Market price in thousand of rubels')
plt.ylabel('Number of examples')
plt.title('Price distribution')
plt.show()

# Median price based on living area
grouped_df = df_train.groupby('life_sq')['price_doc'].aggregate(np.median).reset_index()
plt.figure()
sb.pointplot(grouped_df.life_sq.values, grouped_df.price_doc.values)
plt.ylabel('Median Price')
plt.xlabel('Living area')
plt.xticks(rotation='vertical')
plt.title('Plot of median price based on living area')
plt.show()

# Missing values (based on Kaggle competition snippet)
missing = df_train.isnull().sum(axis=0).reset_index()
missing.columns = ['column_name', 'missing_count']
missing = missing.ix[missing['missing_count']>0]
missing = missing.sort_values(by=['missing_count'], ascending=True)
ind = np.arange(missing.shape[0])
fig, ax = plt.subplots()
ax.barh(ind, missing.missing_count.values)
ax.set_yticks(ind)
ax.set_yticklabels(missing.column_name.values, rotation='horizontal')
ax.set_xlabel("Number of missing values")
ax.set_title("Missing values analysis")
plt.show()

##########################################################
# ---- DATA PREPROCESSING --------------------------------
# DISCLAIMER: Some code from this section was obtained in
# publicly published scripts on Kaggle competition page
##########################################################

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
# ---- MODELS  -------------------------------------------
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
    df_predictions.to_csv(str(num_estimators).join(['../predictions/rf_basic_','.csv']), index=False)


if 'gradientboosting' in SELECTED_MODELS:
    num_estimators = ADDITIONAL_VALUE

    print(datetime.now(), ' Gradient boosting model with ', num_estimators, ' estimators will be fitted...')

    gb = GradientBoostingRegressor(n_estimators=num_estimators, loss='lad', max_depth=7)
    gb.fit(X_train, y_train)

    print(datetime.now(), ' Gradient boosting model fitted...')

    y_pred = gb.predict(X_test)

    print(datetime.now(), ' Gradient boosting predicitions completed')

    df_predictions = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
    df_predictions.to_csv(str(num_estimators).join(['../predictions/gb_basic_', '.csv']), index=False)

if 'mlpregressor' in SELECTED_MODELS:
    file_index = ADDITIONAL_VALUE

    print(datetime.now(), ' Multi-layer Perceptron regressor will be fitted...')

    mlp = MLPRegressor(verbose=True)
    mlp.fit(X_train, y_train)

    print(datetime.now(), ' Multi-layer Perceptron regressor model fitted...')

    y_pred = mlp.predict(X_test)

    print(datetime.now(), ' Multi-layer Perceptron regressor predicitions completed')

    df_predictions = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
    df_predictions.to_csv(str(file_index).join(['../predictions/mlp_basic_', '.csv']), index=False)
