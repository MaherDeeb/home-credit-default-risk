# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:00:34 2018

@author: Maher Deeb
"""

import pandas as pd
    
df_train = pd.read_csv('df_train.csv')
df_bureau = pd.read_csv('df_bureau_decoded.csv')

for col_i in df_bureau.columns[df_bureau.dtypes == 'object']:
    df_bureau[col_i] = df_bureau[col_i].factorize()[0]

df_agg = df_bureau.groupby(['SK_ID_CURR'], axis=0)[df_bureau.columns[2:]].sum()

        
df_train.index = df_train.SK_ID_CURR
#df_bureau.index = df_bureau.SK_ID_CURR

#df_bureau = df_bureau.drop(['SK_ID_CURR'],axis=1)

    
df_train_1 = pd.concat([df_agg,df_train],axis=1)

df_train_2 = df_train_1[~df_train_1['TARGET'].isnull()]

df_train_2.to_csv('df_train_bureau.csv',index=False) 

#del df_train

df_test = pd.read_csv('df_test.csv',encoding='iso-8859-1')

df_test.index = df_test.SK_ID_CURR

df_test_1 = pd.concat([df_agg,df_test],axis=1)


del df_bureau


df_test_2 = df_test_1[~df_test_1['SK_ID_CURR'].isnull()]


df_test.to_csv('df_test_bureau.csv',index=False) 


