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
def data_preprocessing(test_train_ration=0.05):
    
    df_train = pd.read_csv('df_train_bureau_preapp.csv')
    df_test = pd.read_csv('df_test_bureau_preapp.csv')
    
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

    Y_train = df_train_decoded['TARGET']
    SK_ID_CURR_test = df_test_decoded['SK_ID_CURR']
    
    
    #df_train_decoded = df_train_decoded.drop(['SK_ID_CURR','SK_ID_PREV','TARGET'],axis=1)
    #df_test_decoded = df_test_decoded.drop(['SK_ID_CURR','SK_ID_PREV'],axis=1)
    
    #df_train_decoded = df_train_decoded.drop(['SK_ID_CURR','SK_ID_BUREAU','TARGET'],axis=1)
   # df_test_decoded = df_test_decoded.drop(['SK_ID_CURR','SK_ID_BUREAU'],axis=1)
    
    
    df_train_decoded = df_train_decoded.drop(['SK_ID_CURR','TARGET'],axis=1)
    df_test_decoded = df_test_decoded.drop(['SK_ID_CURR'],axis=1)
    
    
    
    
    x_train, x_cv, y_train, y_cv= train_test_split(df_train_decoded,Y_train,
                                                  test_size=test_train_ration,stratify=Y_train,random_state=0)
    
    x_train.to_csv('x_train.csv')
    x_cv.to_csv('x_cv.csv')
    y_train.to_csv('y_train.csv')
    y_cv.to_csv('y_cv.csv')
    df_test_decoded.to_csv('df_test_decoded.csv')
    SK_ID_CURR_test.to_csv('SK_ID_CURR_grouped.csv')
    

def model_train_RF():
    
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv', header = None)
    
    X_train = X_train.fillna(-999)
    y_train = y_train.fillna(-999)
        
    clf = RandomForestClassifier(n_estimators=200,max_depth=50,random_state=0,n_jobs=-1,class_weight='balanced')
    
    clf.fit(X_train, y_train)
    
    print('train:',clf.score(X_train, y_train))
    
    X_test = pd.read_csv('x_cv.csv')
    y_test = pd.read_csv('y_cv.csv', header = None)
    
    X_test = X_test.fillna(-999)
    y_test = y_test.fillna(-999)     
    
    print('test:',clf.score(X_test, y_test))
    
    err_cv=np.divide(np.sum(np.square(np.subtract(clf.predict_proba(X_test)[:,1],y_test.values.T))),len(y_test))
    
    y_rf_te=clf.predict_proba(X_test)[:,1]
    return clf, err_cv,y_rf_te

def lgb_light():
    
    
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv',header = None)    
    
    X_test = pd.read_csv('x_cv.csv')
    y_test = pd.read_csv('y_cv.csv',header = None)
    
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
        'learning_rate': 0.05 ,
        'verbose': 0,
        'num_leaves': 31,
        'bagging_fraction': 1,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 1,
        'min_data_in_leaf': 20,
        'feature_fraction_seed': 1,
        'max_bin': 255,
        'max_depth': -1,
        'num_rounds': 4000,
        'metric' : 'auc',
        'gpu_use_dp': True,
        'save_binary': True,
        #'scale_pos_weight': 2,
        #'drop_rate': 0.02
    }

    model_f2 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)
    

    #y_lgb_te=model_f1.predict(X_test)
    #err_cv_lgb1=np.divide(np.sum(np.square(np.subtract(y_lgb_te,y_test))),len(y_test))
    
    y_lgb_te2=model_f2.predict(X_test)
    err_cv_lgb2=np.divide(np.sum(np.square(np.subtract(y_lgb_te2,pd.DataFrame(y_test)[1]))),len(y_test))
    
        
    return model_f2,y_lgb_te2,err_cv_lgb2#,model_f1,err_cv_lgb1,y_lgb_te


def LDA():

    clf = LinearDiscriminantAnalysis()

    X_train = pd.read_csv('X_train.csv',encoding='iso-8859-1')
    y_train = pd.read_csv('y_train.csv',encoding='iso-8859-1')    
    X_train = X_train.fillna(0)
    y_train = y_train.fillna(0)    
    clf.fit(X_train, y_train)

    X_test = pd.read_csv('x_cv.csv',encoding='iso-8859-1')
    y_test = pd.read_csv('y_cv.csv',encoding='iso-8859-1')
    X_test = X_test.fillna(0)
    y_test = y_test.fillna(0)     
    print('LDA:',clf.score(X_test, y_test))
    y_lda_te=clf.predict_proba(X_test)[:,1]
    
    return clf,y_lda_te
    
def model_pred(clf):
    #X_train = pd.read_csv('X_train.csv')
    X_sub = pd.read_csv('df_test_decoded.csv',encoding='iso-8859-1')
# =============================================================================
# 
#     std_0 = X_train.std()
#     
#     col_to_delete = list(X_train.columns[std_0==0])
#     
#     for col_i in col_to_delete:
#         
#         X_sub = X_sub.drop([col_i],axis=1)
#         
    #X_sub = X_sub.drop(['SK_ID_CURR'],axis=1)
# =============================================================================

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




# =============================================================================
# def stack_features(y_rf_te,y_lgb_te,y_lgb_te2):
#     
#     y_test = pd.read_csv('y_cv.csv',encoding='iso-8859-1')
#     
#     new_fea_test=pd.DataFrame([y_rf_te,y_lgb_te,y_lgb_te2]).transpose()
#  
# 
#     clf_stacked = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=0)
#     
#     clf_stacked.fit(new_fea_test, y_test)
# 
#     print("stacked on test",clf_stacked.score(new_fea_test, y_test))
#     
#     return clf_stacked
#     
# def pred_stacked(clf_stacked,y_sub_bin_rf,y_sub_bin_1,y_sub_bin_2):
#     
#     df_id_submit = pd.read_csv('df_id_submit.csv',encoding='iso-8859-1',header=None)    
# 
#     new_fea_sub=pd.DataFrame([y_sub_bin_rf['target'],y_sub_bin_1['target'],y_sub_bin_2['target']]).transpose()
#     
#     y_sub_bin=clf_stacked.predict_proba(new_fea_sub)[:,1]
#     
#     Y_submit=pd.DataFrame()
#     Y_submit = pd.concat([df_id_submit, pd.DataFrame(y_sub_bin)], axis=1)
# # =============================================================================
#     Y_submit.columns=['SK_ID_CURR','TARGET']
#     Y_submit.to_csv('{}_submit_stacked.csv'.format(str(round(time.mktime((datetime.datetime.now().timetuple()))))),index=False)    
#     
# =============================================================================


    
#data_preprocessing(test_train_ration=0.2)

#clf_rf, err_cv,y_rf_te=model_train_RF()
model_f2,y_lgb_te2,err_cv_lgb2=lgb_light()
#clf_lda,y_lda_te=LDA()


#y_sub_bin_rf=model_pred(clf_rf)
y_sub_bin_2=model_pred(model_f2)
#y_sub_bin_lda=model_pred(clf_lda)

#clf_stacked=stack_features(y_rf_te,y_sub_bin_lda,y_lgb_te2)
#pred_stacked(clf_stacked,y_sub_bin_rf,y_sub_bin_lda,y_sub_bin_2)