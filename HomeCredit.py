
# coding: utf-8
import os, copy
import pandas as pd
import numpy as np

import arrow

# examine fraction of missing value in each attribute
def missing_df(df):
        """
        return a data frame containing the statistics of missing value
        """
        
        total = df.isnull().sum().sort_values(ascending=False)
        fraction = 100 * total / df.shape[0]
        
        # keep to two decimal places
        fraction = fraction.apply(lambda x: round(x,2))
        
        df_missing = pd.concat([total,fraction], axis=1, keys=['Total','Fraction'])
        df_missing.index.name = 'Attributes'
        
        return df_missing
    
    
def cut_missing_fea(df, mis_threshold=60):
    """
    cut off features with ratio of missing values bigger than the threshold value
    
    Params:
    ----------------
    df:  the dataframe
    mis_threshold: the threshold value in percentage to cut off features
    
    Return:
    ----------------
    dataframe with the remaining features
    """
    missing = missing_df(df)
    keep_fea = missing[missing['Fraction']<mis_threshold].index

    return df[keep_fea]


def one_hot(df):
    # if only one or two categories, factorize()
    for col in df:
        if df[col].dtype == 'object':
            # first fillna with a label, like my name,
            # so in pd.get_dummies(), I don't need to set dummy_na=True
            # which might lead to cols with all zeros
            if df[col].isnull().any():
                df[col].fillna('heyNAN', inplace=True)

            if len(list(df[col].unique())) <= 2:
                df.loc[:, col] = pd.factorize(df[col])[0]

    # one hot encoding if more than two categories, including Null
    cat_columns = [col for col in df.columns if df[col].dtype=='object']
    df = pd.get_dummies(df, columns=cat_columns, dummy_na=False)
    return df


def prepare_data(df, group_var, prefix, mis_threshold=60):
    df = cut_missing_fea(df, mis_threshold)

    num_features = df.select_dtypes(exclude=['object']).columns

    # one hot encoding for categorical features
    df = one_hot(df)

    agg_dict = {}
    for fea in df.columns:
        if fea in (group_var, 'SK_ID_CURR'):
            continue
        elif fea in num_features:
            agg_dict[fea] = ['sum', 'mean', 'max', 'var']
        else:
            df[fea].fillna(0, inplace=True)
            agg_dict[fea] = ['sum', 'mean']

    df1 = df.drop([group_var], axis=1).groupby('SK_ID_CURR').agg(agg_dict)
    df1.columns = ['_'.join(col).strip('_ ') for col in df1.columns.values]
    # count group_var for each SK_ID_CURR
    df2 = df[['SK_ID_CURR', group_var]].groupby('SK_ID_CURR').count()

    # merge
    df = pd.merge(df1, df2, left_index=True, right_index=True)
    del df1, df2
    # add prefix
    df = df.add_prefix(prefix + '_')

    return df.reset_index()


def app_train(path, n_rows):
    app_train_raw = pd.read_csv(path + '/application_train.csv',nrows=n_rows)
    app_test_raw = pd.read_csv(path + '/application_test.csv',nrows=n_rows)

    # make a label indicating whether the instance belongs to train to test dataset
    app_train_raw['is_train'] = 1
    app_test_raw['is_train'] = 0

    # target for train data
    trainY = app_train_raw['TARGET']
    app_train_raw.drop('TARGET', axis=1, inplace=True)

    # test data id
    testID = app_test_raw['SK_ID_CURR']

    # we will first concatenate train and test data to simplify data wrangling
    data = pd.concat([app_train_raw, app_test_raw], axis=0)

    assert data.shape[0] == app_train_raw.shape[0] + app_test_raw.shape[0], data.shape[1] == app_train_raw.shape[1]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    data['CREDIT_TO_ANNUITY_RATIO'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
    data['CREDIT_TO_GOODS_RATIO'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
    data['ANNUITY_TO_INCOME_RATIO'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['CREDIT_TO_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']

    data['CNT_FAM_MEMBERS'].fillna(data['CNT_FAM_MEMBERS'].median(), inplace=True)
    data['INC_PER_MEMB'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
    data['INC_PER_CHLD'] = data['AMT_INCOME_TOTAL'] / (1 + data['CNT_CHILDREN'])

    data['EMPLOY_TO_BIRTH_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['CAR_TO_BIRTH_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_BIRTH']
    data['CAR_TO_EMPLOY_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_EMPLOYED']
    data['PHONE_TO_BIRTH_RATIO'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']

    data['SOURCES_PROD12'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2']
    data['SOURCES_PROD13'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_3']
    data['SOURCES_PROD23'] = data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
    data['SOURCES_PROD123'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
    # data['EXT_SOURCES_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    data['SCORES_STD'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    # data['SCORES_STD'].fillna(data['SCORES_STD'].mean(), inplace=True)

    # cut features with too many missing values
    data = cut_missing_fea(data)

    # one hot encoding
    data = one_hot(data)

    return data, trainY, testID


def bureau(data, path, n_rows):
    bureau = pd.read_csv(path + '/bureau.csv',nrows=n_rows)
    bureau_balance = pd.read_csv(path + '/bureau_balance.csv',nrows=n_rows)

    bureau_balance['STATUS'] = bureau_balance['STATUS'].replace(['C', 'X'], '0').apply(lambda x: int(x))
    bb = bureau_balance['STATUS'].groupby(bureau_balance['SK_ID_BUREAU']).agg(['mean', 'sum', 'count'])
    bb = bb.add_prefix('bureau_balance_')

    # merge bureau and bureau_balance
    bureau = pd.merge(bureau, bb, left_on='SK_ID_BUREAU', right_index=True, how='left')

    bureau_agg = prepare_data(bureau, group_var='SK_ID_BUREAU', prefix='bureau')

    # merge with  data
    data = pd.merge(data, bureau_agg, how='left')

    keep_feat = ['SK_ID_CURR','AMT_CREDIT_MAX_OVERDUE','CNT_CREDIT_PROLONG','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE','AMT_ANNUITY']

    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    bureau_active_agg = bureau_active[keep_feat].groupby('SK_ID_CURR').agg(['mean','max','sum'])
    bureau_active_agg.columns = ['_'.join(col).strip('_ ') for col in bureau_active_agg.columns.values]
    bureau_active_agg = bureau_active_agg.add_prefix('active_')
    data = pd.merge(data, bureau_active_agg.reset_index(), how='left')

    bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    bureau_closed_agg = bureau_closed[keep_feat].groupby('SK_ID_CURR').agg(['mean','max','sum'])
    bureau_closed_agg.columns = ['_'.join(col).strip('_ ') for col in bureau_closed_agg.columns.values]
    bureau_closed_agg = bureau_closed_agg.add_prefix('closed_')
    data = pd.merge(data, bureau_closed_agg.reset_index(), how='left')


    return data

def prev_app(data, path, n_rows):
    prev_app = pd.read_csv(path + '/previous_application.csv',nrows=n_rows)
    col_DAYS = []
    for col in prev_app:
        if 'DAYS' in col:
            col_DAYS.append(col)

    for col in col_DAYS:
        prev_app[col].replace(365243, np.nan, inplace=True)
    prev_app['APP_CREDIT_PERC'] = prev_app['AMT_APPLICATION'] / (1+prev_app['AMT_CREDIT'])

    prev_agg = prepare_data(prev_app, group_var='SK_ID_PREV', prefix='prev_app')

    # merge with  data
    data = pd.merge(data, prev_agg, how='left')
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    prev_approved = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Approved']
    prev_approved_agg = prev_approved.groupby('SK_ID_CURR').agg(num_aggregations)
    prev_approved_agg.columns = ['_'.join(col).strip('_ ') for col in prev_approved_agg.columns.values]
    prev_approved_agg = prev_approved_agg.add_prefix('approved_')
    data = pd.merge(data, prev_approved_agg.reset_index(), how='left')

    prev_refused = prev_app[prev_app['NAME_CONTRACT_STATUS'] == 'Refused']
    prev_refused_agg = prev_refused.groupby('SK_ID_CURR').agg(num_aggregations)
    prev_refused_agg.columns = ['_'.join(col).strip('_ ') for col in prev_refused_agg.columns.values]
    prev_refused_agg = prev_refused_agg.add_prefix('refused_')
    data = pd.merge(data, prev_refused_agg.reset_index(), how='left')

    return data




def POS_CASH(data, path, n_rows):
    pos_bal = pd.read_csv(path + '/POS_CASH_balance.csv',nrows=n_rows)

    pos_bal = prepare_data(pos_bal, group_var='SK_ID_PREV', prefix='pos')

    # merge with  data
    data = pd.merge(data, pos_bal, how='left')

    return data



def install(data, path, n_rows):
    install = pd.read_csv(path + '/installments_payments.csv',nrows=n_rows)

    # Percentage and difference paid in each installtallment (amount paid and installtallment value)
    install['PAYMENT_RATIO'] = install['AMT_PAYMENT'] / (1+install['AMT_INSTALMENT'])
    install['PAYMENT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    install['DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']
    install['DBD'] = install['DAYS_INSTALMENT'] - install['DAYS_ENTRY_PAYMENT']
    install['DPD'] = install['DPD'].apply(lambda x: x if x > 0 else 0)
    install['DBD'] = install['DBD'].apply(lambda x: x if x > 0 else 0)

    install = prepare_data(install, group_var='SK_ID_PREV', prefix='install')

    # merge with  data
    data = pd.merge(data, install, how='left')

    return data


def credit(data, path, n_rows):
    credit_bal = pd.read_csv(path + '/credit_card_balance.csv',nrows=n_rows)

    credit_bal = prepare_data(credit_bal, group_var='SK_ID_PREV', prefix='credit')
    # merge with  data
    data = pd.merge(data, credit_bal, how='left')

    return data


def final_processing(data):
    data = cut_missing_fea(data)

    #### prepare final Train X and Test X dataframes
    ignore_features = ['SK_ID_CURR', 'is_train']
    features = [col for col in data.columns if col not in ignore_features]
    trainX = data[data['is_train'] == 1][features]
    testX = data[data['is_train'] == 0][features]

    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy="median")
    imputer.fit(trainX)
    # trainX, testX become numpy array, not pandas dataframe anymore
    trainX = imputer.transform(trainX)
    testX = imputer.transform(testX)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX, testX = scaler.transform(trainX), scaler.transform(testX)

    # convert back into pandas dataframe
    trainX = pd.DataFrame(trainX,columns=features)
    testX = pd.DataFrame(testX,columns=features)

    return trainX, testX


def xgboost_train(trainX, trainY, testX):
    import xgboost as xgb
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(x_train.values, label=y_train.values.ravel())
    dval = xgb.DMatrix(x_val.values, label = y_val.values.ravel())

    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic','n_thread':4,'eval_metric':'auc'}
    evallist = [(dval,'eval'),(dtrain,'train')]
    num_round = 100
    xgbst = xgb.train(param, dtrain, num_round, evallist)
    xgbst.save_model('xgb_v1.model')



def lgb_train(trainX, trainY, testX):
    import lightgbm as lgb
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    lgb_train = lgb.Dataset(data=x_train, label=y_train)
    lgb_val = lgb.Dataset(data=x_val, label=y_val)
    params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
              'learning_rate': 0.02, 'num_leaves': 40, 'num_iterations': 5000, 'verbose': 0,
              'colsample_bytree': .8, 'subsample': .9, 'max_depth': 8, 'reg_alpha': .1, 'reg_lambda': .08,
              'min_split_gain': .01, 'min_child_weight': 1}

    # model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=100, verbose_eval=20)
    #
    #
    # feat_imp_df = pd.DataFrame()
    # feat_imp_df["feature"] = trainX.columns
    # feat_imp_df["importance"] = model.feature_importance()
    # feat_imp_df.sort_values(by='importance',ascending=False,inplace=True)
    #
    # today = arrow.now().format('YYYYMMDD')
    # feat_imp_df.to_csv('feature_importance_' + today + '.csv', index = False)
    # # print(feat_imp_df.head())
    # imp_feats = feat_imp_df[feat_imp_df['importance']>1]['feature'].values
    # trainX, testX = trainX[imp_feats],testX[imp_feats]

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=100, verbose_eval=20)
    today = arrow.now().format('YYYYMMDD')
    joblib.dump(model, 'lgb_clf_' + today + '.pkl')
    test_Y = model.predict(testX)
    return test_Y

def lgb_KFold(trainX, trainY, testX):
    import lightgbm as lgb
    from sklearn.externals import joblib
    from sklearn.model_selection import StratifiedKFold
    skfolds = StratifiedKFold(n_splits=5, random_state=42)
    test_Y = np.zeros(testX.shape[0])
    for train_id, val_id in skfolds.split(trainX, trainY):
        x_train, x_val = trainX.iloc[train_id], trainX.iloc[val_id]
        y_train, y_val = trainY.iloc[train_id], trainY.iloc[val_id]

        # LightGBM parameters found by Bayesian optimization
        clf = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=48,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=False,
            verbose=0, )

        clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 100)

        test_Y += clf.predict_proba(testX, num_iteration=clf.best_iteration_)[:, 1]/skfolds.n_splits
    return test_Y

def main():
    path = os.getcwd()
    path = os.path.join(path, 'data')
    assert os.path.isdir(path), "the path for data is not valid"

    n_rows = 500

    print("app_train...")
    data, trainY, testID = app_train(path, n_rows)
    assert not np.any(np.isinf(data.values)), " data contains infinity, check data values! "

    print("bureau...")
    data = bureau(data, path, n_rows)
    assert not np.any(np.isinf(data.values)), " data contains infinity, check data values! "

    print("prev_app...")
    data = prev_app(data, path, n_rows)
    assert not np.any(np.isinf(data.values)), " data contains infinity, check data values! "

    print("POS_CASH...")
    data = POS_CASH(data, path, n_rows)
    assert not np.any(np.isinf(data.values)), " data contains infinity, check data values! "

    print("install...")
    data = install(data, path, n_rows)
    assert not np.any(np.isinf(data.values)), " data contains infinity, check data values! "

    print("credit...")
    data = credit(data, path, n_rows)
    assert not np.any(np.isinf(data.values)), " data contains infinity, check data values! "

    print("final...")
    trainX, testX = final_processing(data)

    print("saving data...")
    trainX.to_csv("trainX.csv")
    testX.to_csv("testX.csv")

    ### XGB
    # print("xgb training...")

    ###  LightGBM
    # print("lgb training...")
    # test_Y = lgb_train(trainX, trainY, testX)

    ### Stratified LightGBM
    print("stratified lgb training...")
    test_Y = lgb_KFold(trainX, trainY, testX)


    submit = pd.DataFrame({'SK_ID_CURR': testID.values, 'TARGET': test_Y})
    submit.set_index('SK_ID_CURR')
    # Save the submission dataframe
    today = arrow.now().format('YYYYMMDD')
    submit.to_csv('lightgbm_' + today + '.csv', index = False)


if __name__ == "__main__":
    main()

