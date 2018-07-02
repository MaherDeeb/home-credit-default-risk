# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:19:00 2017

@author: Maher Deeb
"""
#1.0. import important libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import datetime, time

import lightgbm as lgb


#1.1. load the training set and the training lables
def data_preprocessing(test_train_ration,random_state):
    
    try:
        
        df_train_decoded_sorted = pd.read_csv('df_train_bureau_preapp_pos_sorted.csv')
        df_test_decoded_sorted = pd.read_csv('df_test_bureau_preapp_pos_sorted.csv')
        
    except:
        
        df_train = pd.read_csv('df_train_bureau_preapp_pos.csv')
        df_test = pd.read_csv('df_test_bureau_preapp_pos.csv')
        
        df_train_decoded = df_train
        df_test_decoded= df_test
        
        df_total = pd.concat([df_train,df_test])
        df_total.index=range(len(df_total['SK_ID_CURR']))
        for col_i in df_train.columns[df_train.dtypes == 'object']:
            
            df_total[col_i] = df_total[col_i].factorize()[0]
            df_train_decoded[col_i] = df_total.loc[range(df_train.shape[0]),col_i].values
            df_test_decoded[col_i] =  df_total.loc[range(df_train.shape[0],df_train.shape[0]+df_test.shape[0]),col_i].values
            
        #df_train_decoded = df_train_decoded.groupby(['SK_ID_CURR'], axis=0)[df_train_decoded.columns].sum()
        #df_test_decoded = df_test_decoded.groupby(['SK_ID_CURR'], axis=0)[df_test_decoded.columns].sum()
        
        df_corr = df_train_decoded.corr()
        df_corr = df_corr.sort_values(by=['TARGET'], ascending=False)
        
        Col_df_corr = list(df_corr.index)
        col_to_del = list(df_corr[df_corr['TARGET'].isnull()].index)
        
        df_train_decoded_sorted = df_train_decoded[Col_df_corr]
        df_test_decoded_sorted = df_test_decoded[Col_df_corr[1:]]
        
        df_train_decoded_sorted = df_train_decoded_sorted.drop(col_to_del,axis = 1)
        df_test_decoded_sorted = df_test_decoded_sorted.drop(col_to_del,axis = 1)
        
        df_train_decoded_sorted.to_csv('df_train_bureau_preapp_pos_sorted.csv')
        df_test_decoded_sorted.to_csv('df_test_bureau_preapp_pos_sorted.csv')
    
    Y_train = df_train_decoded_sorted['TARGET']
    SK_ID_CURR_test = df_test_decoded_sorted['SK_ID_CURR']
    
    
    df_train_decoded_sorted = df_train_decoded_sorted.drop(['SK_ID_CURR','TARGET'],axis=1)
    df_test_decoded_sorted = df_test_decoded_sorted.drop(['SK_ID_CURR'],axis=1)
    
    
    
    x_train, x_cv, y_train, y_cv= train_test_split(df_train_decoded_sorted,Y_train,
                                                  test_size=test_train_ration,stratify=Y_train,random_state=random_state)
    
    x_train.to_csv('x_train_sr{}.csv'.format(random_state))
    x_cv.to_csv('x_cv_sr{}.csv'.format(random_state))
    y_train.to_csv('y_train_sr{}.csv'.format(random_state))
    y_cv.to_csv('y_cv_sr{}.csv'.format(random_state))
    df_test_decoded_sorted.to_csv('df_test_decoded.csv')
    SK_ID_CURR_test.to_csv('SK_ID_CURR_grouped.csv')
    


def lgb_light(random_state):
    
    
    X_train = pd.read_csv('x_train_sr{}.csv'.format(random_state))
    y_train = pd.read_csv('y_train_sr{}.csv'.format(random_state),header = None)    
    
    X_test = pd.read_csv('x_cv_sr{}.csv'.format(random_state))
    y_test = pd.read_csv('y_cv_sr{}.csv'.format(random_state),header = None)
    
    
    
# =============================================================================
#     std_0 = X_train.std()
#     
#     col_to_delete = list(X_train.columns[std_0==0])
#     
#     for col_i in col_to_delete:
#         
#         X_train = X_train.drop([col_i],axis=1)
#         X_test = X_test.drop([col_i],axis=1)
# =============================================================================
    
    #X_train = X_train.drop(['SK_ID_CURR'],axis=1)
    #X_test = X_test.drop(['SK_ID_CURR'],axis=1)

    d_train_final = lgb.Dataset(X_train, pd.DataFrame(y_train)[1])
    watchlist_final = lgb.Dataset(X_test, pd.DataFrame(y_test)[1])

    params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.3 ,
            'verbose': 0,
            'num_leaves': 150,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 256,
            'max_depth': 10,
            'num_rounds': 600,
            'metric' : 'auc'
        }

    #model_f2 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)

    
    params = {
        'objective': 'binary',
        'boosting': 'dart',
        'learning_rate': 0.1 ,
        'verbose': 0,
        'num_leaves': 31,
        'bagging_fraction': 1,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 1,
        'min_data_in_leaf': 30,
        'feature_fraction_seed': 1,
        'max_bin': 255,
        'max_depth': -1,
        'num_rounds': 2000,
        'metric' : 'auc',
        'gpu_use_dp': True,
        'save_binary': True,
        'scale_pos_weight': 2,
        'drop_rate': 0.02
    }

    model_f2 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)
    

    #y_lgb_te=model_f1.predict(X_test)
    #err_cv_lgb1=np.divide(np.sum(np.square(np.subtract(y_lgb_te,y_test))),len(y_test))
    
    y_lgb_te2=model_f2.predict(X_test)
    err_cv_lgb2=np.divide(np.sum(np.square(np.subtract(y_lgb_te2,pd.DataFrame(y_test)[1]))),len(y_test))
    
        
    return model_f2,y_lgb_te2,err_cv_lgb2#,model_f1,err_cv_lgb1,y_lgb_te


    
def model_pred(clf):
    #X_train = pd.read_csv('X_train.csv')
    X_sub = pd.read_csv('df_test_decoded.csv',encoding='iso-8859-1')

    clf=model_f2
    if clf!=model_f2:
        X_sub = X_sub.fillna(-999999999)
    df_id_submit = pd.read_csv('SK_ID_CURR_grouped.csv',encoding='iso-8859-1',header=None)
    y=clf.predict(X_sub)

    Y_submit=pd.DataFrame()
    Y_submit = pd.concat([df_id_submit[1], pd.DataFrame(y)], axis=1)
    Y_submit.index =Y_submit[1] 
# =============================================================================
    Y_submit.columns=['SK_ID_CURR','TARGET']
    Y_submit=Y_submit.groupby(['SK_ID_CURR'])['TARGET'].mean()
    df_best_submit = pd.read_csv('1529870832_submit.csv')
    df_best_submit.index = df_best_submit.SK_ID_CURR
    df_best_submit = pd.concat([df_best_submit,Y_submit],axis=1)
    df_best_submit.columns=['SK_ID_CURR','TARGET1','TARGET']
    ix = df_best_submit[df_best_submit['TARGET'].isnull()].index.tolist()
    df_best_submit.loc[ix,'TARGET'] = df_best_submit.loc[ix,'TARGET1']
    df_best_submit = df_best_submit.drop(['TARGET1'],axis=1)
    df_best_submit.to_csv('{}_submit.csv'.format(str(round(time.mktime((datetime.datetime.now().timetuple()))))),index=False)
# =============================================================================
    return df_best_submit



random_state = 0
test_train_ration=0.2
#data_preprocessing(test_train_ration,random_state)
model_f2,y_lgb_te2,err_cv_lgb2=lgb_light(random_state)
y_sub_bin_2=model_pred(model_f2)

