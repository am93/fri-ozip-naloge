# My Kaggle username: anzzemedved
# My stacking score:
# My model averaging score:
# My best score:

#### !!!!!!!!!!!!!!!!!!!!! ####################################################
# Zaradi tezav pri namestitvi knjiznjice XGBoost sem uporabil
# gradient boosting, ki je na voljo v sklearn
###############################################################################

import numpy as np
import pandas as pd
from datetime import datetime
import sys, getopt
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

##########################################################
# ---- PARAMETERS --------------------------------
##########################################################
num_estimators = 1000
num_fold = 3

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
plt.hist(df_train.price_doc.values / 1000, bins=30)
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
missing = missing.ix[missing['missing_count'] > 0]
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
df_values.fillna(df_values.mean())
df_values.info()
df_values = df_values.astype(np.float32)
df_values.info()

print(datetime.now(), ' Data preprocessing finished...')

# get values and split them in train and test back
df_values -= df_values.min()
df_values /= df_values.max()
X_all = np.float32(np.nan_to_num(df_values.values))
X_train = X_all[:num_train]
X_test = X_all[num_train:]
y_train = np.float32(y_train)

##########################################################
# ---- KERAS  -------------------------------------------
##########################################################
epochs = 150

model = Sequential()
model.add(Dense(296, activation='relu', input_shape=(296,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, init='normal'))
model.compile(optimizer='adam',
              loss='mean_squared_error')
model.summary()

##########################################################
# ---- RANDOM FOREST -------------------------------------
##########################################################
rf = RandomForestRegressor(n_estimators=num_estimators, n_jobs=-1)

##########################################################
# ---- EXTRA TREES   -------------------------------------
##########################################################
gb = GradientBoostingRegressor(n_estimators=num_estimators, loss='lad', max_depth=7)

##########################################################
# ---- CROSS VALIDATION ----------------------------------
##########################################################
X_stacking_train = None
y_stacking_train = None

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(X_train):
    tmp_x_train, tmp_x_test = X_train[train_index], X_train[test_index]
    tmp_y_train, tmp_y_test = y_train[train_index], y_train[test_index]

    # RF
    rf.fit(tmp_x_train, tmp_y_train)
    rf_preds = rf.predict(tmp_x_test)

    # gradient boosting
    gb.fit(tmp_x_train, tmp_y_train)
    gb_preds = gb.predict(tmp_x_test)

    # NN
    model.fit(tmp_x_train, tmp_y_train,
              epochs=epochs,
              verbose=1)
    nn_preds = model.predict(tmp_x_test, verbose=1)

    # merge results
    tmp_res = np.hstack(np.hstack((rf_preds, gb_preds)), nn_preds)

    # save results
    if X_stacking_train is None:
        X_stacking_train = tmp_res
        y_stacking_train = tmp_y_test
    else:
        X_stacking_train = np.vstack((X_stacking_train, tmp_res))
        y_stacking_train = np.vstack((y_stacking_train, tmp_y_test))


##########################################################
# ---- TEST predicitions --------------------------------
##########################################################
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
model.fit(X_train, y_train, epochs=epochs, verbose=1)

rf_preds = rf.predict(X_test)
gb_preds = gb.predict(X_test)
nn_preds = model.predict(X_test)

X_stacking_test = np.hstack(np.hstack((rf_preds, gb_preds)), nn_preds)


##########################################################
# ---- Linear regression for stacking --------------------
##########################################################
lr = LinearRegression(n_jobs=-1)
lr.fit(X_stacking_train, y_stacking_train)
stacking_preds = lr.predict(X_stacking_test)

##########################################################
# ---- Mean predicition of models ------------------------
##########################################################
avg_preds = X_stacking_test.mean(axis=0)

##########################################################
# ---- Save predictions ----------------------------------
##########################################################
stack_predictions = pd.DataFrame({'id': id_test, 'price_doc': stacking_preds.flatten()})
stack_predictions.to_csv('../predictions/stacking.csv', index=False)

avg_predictions = pd.DataFrame({'id': id_test, 'price_doc': avg_preds.flatten()})
avg_predictions.to_csv('../predictions/average.csv', index=False)