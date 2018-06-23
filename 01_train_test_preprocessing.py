# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:38:46 2018

@author: Maher Deeb
"""

import pandas as pd


data_path = 'D:/000_Projects_2018/0002_Development/Kaggle/home-credit-default-risk/data/'

df_train = pd.read_csv(data_path+'application_train.csv',encoding='iso-8859-1')

Y_train = df_train.TARGET
#df_train = df_train.drop(['TARGET'],axis = 1)

df_test = pd.read_csv(data_path+'application_test.csv',encoding='iso-8859-1')
df_id_submit=df_test.SK_ID_CURR


to_delete = ['OWN_CAR_AGE','APARTMENTS_MODE','BASEMENTAREA_MODE','BASEMENTAREA_MEDI','YEARS_BUILD_MODE',
             'YEARS_BUILD_MEDI','COMMONAREA_MODE','COMMONAREA_MEDI','ELEVATORS_MODE','ELEVATORS_MEDI',
             'FLOORSMIN_MODE','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE',
            'NONLIVINGAREA_MODE','NONLIVINGAREA_MEDI','DEF_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','FLAG_MOBIL','FLAG_DOCUMENT_2',
             'FLAG_DOCUMENT_4','FLAG_DOCUMENT_7','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14',
             'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']

df_train=df_train.drop(to_delete,axis=1)
df_test=df_test.drop(to_delete,axis=1)

df_test.to_csv('df_test.csv',index=False)
df_train.to_csv('df_train.csv',index=False)
Y_train.to_csv('Y_train_o.csv',index=False)
df_id_submit.to_csv('df_id_submit.csv',index=False)
