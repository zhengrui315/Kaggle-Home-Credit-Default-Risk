import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import copy


# examine fraction of missing value in each attribute
def missing_df(df):
    """
    return a data frame containing the statistics of missing value
    """

    total = df.isnull().sum().sort_values(ascending=False)
    fraction = 100 * total / df.shape[0]

    # keep to two decimal places
    fraction = fraction.apply(lambda x: round(x, 2))

    df_missing = pd.concat([total, fraction], axis=1, keys=['Total', 'Fraction'])
    df_missing.index.name = 'Attributes'

    return df_missing


def cut_missing_fea(df, mis_threshold=60):
    """
    cut off features with ratio of missing values bigger than 60%
    """
    missing = missing_df(df)
    keep_fea = missing[missing['Fraction'] < mis_threshold].index

    return df[keep_fea]


def prepare_data(df, group_var, prefix):
    """
    (1) cut off features with too many missing values;
    (2) one hot encoding for categorical feature;
    (3) aggregate and create new features including sum, mean, std
    (4) count num of group_var for each 'SK_ID_CURR'
    """
    # cut off featues with too many missing values
    df = cut_missing_fea(df)

    # if only one or two categories, factorize()
    for col in df:
        if df[col].dtype == 'object' and len(list(df[col].unique())) <= 2:
            df.loc[:, col], _ = pd.factorize(df[col])

    # one hot encoding if more than two categories, including Null
    df = pd.get_dummies(df, dummy_na=True)

    # compute sum, mean, std for each column
    d1 = df.drop([group_var], axis=1).groupby('SK_ID_CURR').agg(['sum', 'mean', 'std'])
    d1.columns = ['_'.join(col).strip('_ ') for col in d1.columns.values]
    d1.add_prefix(prefix + '_')

    # count group_var for each SK_ID_CURR
    d2 = df[['SK_ID_CURR', group_var]].groupby('SK_ID_CURR').count()

    # merge
    df = pd.merge(d1, d2, left_index=True, right_index=True, how='left')

    # add prefix
    df = df.add_prefix(prefix + '_')

    return df.reset_index()



path = os.getcwd()
path = os.path.join(path, 'data')
#os.listdir(path)


print("reading app_train and app_test...")
app_train_raw = pd.read_csv(path + '/application_train.csv')
app_test_raw = pd.read_csv(path + '/application_test.csv')


# we will first concatenate train and test data to simplify the data wrangling

# make a label as to whether the instance belongs to train to test dataset
app_train_raw['is_train'] = 1
app_test_raw['is_train'] = 0

# target for train data
trainY = app_train_raw['TARGET']
app_train_raw.drop('TARGET',axis=1,inplace=True)


# test id
testID = app_test_raw['SK_ID_CURR']

data = pd.concat([app_train_raw,app_test_raw],axis=0)

assert data.shape[0] == app_train_raw.shape[0] + app_test_raw.shape[0], data.shape[1] == app_train_raw.shape[1]

# credit to Bojan TunguzXGB Simple Features
# https://www.kaggle.com/tunguz/xgb-simple-features/code
# NaN values for DAYS_EMPLOYED: 365.243 -> nan
data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

data['NEW_CREDIT_TO_ANNUITY_RATIO'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
data['NEW_CREDIT_TO_GOODS_RATIO'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
data['NEW_ANNUITY_TO_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['NEW_CREDIT_TO_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']

data['CNT_FAM_MEMBERS'].fillna(data['CNT_FAM_MEMBERS'].median(),inplace=True)
data['NEW_INC_PER_MEMB'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['NEW_INC_PER_CHLD'] = data['AMT_INCOME_TOTAL'] / (1 + data['CNT_CHILDREN'])

data['NEW_EMPLOY_TO_BIRTH_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['NEW_CAR_TO_BIRTH_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_BIRTH']
data['NEW_CAR_TO_EMPLOY_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_EMPLOYED']
data['NEW_PHONE_TO_BIRTH_RATIO'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']

data['NEW_SOURCES_PROD'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
data['NEW_EXT_SOURCES_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
data['NEW_SCORES_STD'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
data['NEW_SCORES_STD'].fillna(data['NEW_SCORES_STD'].mean(),inplace=True)

# cut features with too much values
data = cut_missing_fea(data)


# if only one or two categories, factorize()
for col in data:
    if data[col].dtype == 'object' and len(list(data[col].unique())) <= 2:
        data[col], _ = pd.factorize(data[col])

# one hot encoding if more than two categories, including Null
data = pd.get_dummies(data,dummy_na=True)

del app_train_raw, app_test_raw




print("reading and preparing bureau ...")
bureau = pd.read_csv(path + '/bureau.csv')
bureau_balance = pd.read_csv(path + '/bureau_balance.csv')

bureau_balance['STATUS'] = bureau_balance['STATUS'].replace(['C','X'],'0').apply(lambda x:int(x))
bureau_balance = bureau_balance['STATUS'].groupby(bureau_balance['SK_ID_BUREAU']).sum().to_frame()

# merge the sum of DPD
bureau = pd.merge(bureau,bureau_balance,left_on='SK_ID_BUREAU',right_index=True,how='left')
# print("bureau.shape = ",bureau.shape)


bureau = prepare_data(bureau,group_var='SK_ID_BUREAU',prefix='bureau')
# merge with  data
data = pd.merge(data,bureau,how='left')
del bureau, bureau_balance


print("reading and preparing previous application...")
prev_app = pd.read_csv(path + '/previous_application.csv')

col_DAYS = []
for col in prev_app:
    if 'DAYS' in col:
        col_DAYS.append(col)

for col in col_DAYS:
    prev_app[col].replace(365243,np.nan,inplace=True)
del col_DAYS

prev_app = prepare_data(prev_app,group_var='SK_ID_PREV',prefix='prev_app')
# merge with  data
data = pd.merge(data,prev_app,how='left')
del prev_app


pos_bal = pd.read_csv(path + '/POS_CASH_balance.csv')
pos_bal = prepare_data(pos_bal,group_var='SK_ID_PREV',prefix='pos')
data = pd.merge(data,pos_bal,how='left')
del pos_bal

install_pay = pd.read_csv(path + '/installments_payments.csv')
install_pay = prepare_data(install_pay,group_var='SK_ID_PREV',prefix='install')
data = pd.merge(data,install_pay,how='left')
del install_pay

credit_bal = pd.read_csv(path + '/credit_card_balance.csv')
credit_bal = prepare_data(credit_bal,group_var='SK_ID_PREV',prefix='credit')
data = pd.merge(data,credit_bal,how='left')
del credit_bal


print("filling missing value ...")
missing = missing_df(data)
keep_attr = missing[missing['Fraction']<60].index

data1 = data[keep_attr]
# print("data1.shape = ",data1.shape)
del missing, keep_attr



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score

#### prepare final Train X and Test X dataframes
ignore_features = ['SK_ID_CURR', 'is_train']
relevant_features = [col for col in data1.columns if col not in ignore_features]
trainX = data1[data1['is_train'] == 1][relevant_features]
testX = data1[data1['is_train'] == 0][relevant_features]

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
imputer.fit(trainX)
trainX = imputer.transform(trainX)
testX = imputer.transform(testX)

print("normalizing...")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(trainX)
trainX, testX = scaler.transform(trainX), scaler.transform(testX)

x_train, x_val, y_train, y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=42)


print("training ...")
import lightgbm as lgb
lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)


params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1,
          'min_split_gain':.01, 'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=20)


print("predict and submit...")
test_Y_score = model.predict(testX)
submit = pd.DataFrame({'SK_ID_CURR':testID.values, 'TARGET':test_Y_score})
submit.set_index('SK_ID_CURR')

# Save the submission dataframe
submit.to_csv('lightgbm_sub5.csv', index = False)


