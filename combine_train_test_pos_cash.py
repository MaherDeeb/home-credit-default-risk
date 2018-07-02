# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 02:07:53 2018

@author: Maher Deeb
"""

import pandas as pd
    
df_train = pd.read_csv('df_train_bureau_preapp.csv')
df_bureau = pd.read_csv('df_POS_decoded.csv')

for col_i in df_bureau.columns[df_bureau.dtypes == 'object']:
    df_bureau[col_i] = df_bureau[col_i].factorize()[0]

df_agg = df_bureau.groupby(['SK_ID_CURR'], axis=0)[df_bureau.columns[2:]].sum()

#df_agg.columns = [x + "mean" for x in list(df_agg.columns)]

df_train.index = df_train.SK_ID_CURR
#df_bureau.index = df_bureau.SK_ID_CURR

#df_bureau = df_bureau.drop(['SK_ID_CURR'],axis=1)

    
df_train_1 = pd.concat([df_agg[df_agg.columns[1:]],df_train],axis=1)
df_train_2 = df_train_1.dropna(subset=['TARGET']) 

df_train_2.to_csv('df_train_bureau_preapp_pos.csv',index=False) 

del df_train,df_train_1
del df_bureau


df_test = pd.read_csv('df_test_bureau_preapp.csv',encoding='iso-8859-1')

df_test.index = df_test.SK_ID_CURR

df_test_1 = pd.concat([df_agg[df_agg.columns[1:]],df_test],axis=1)
del df_test
del df_agg

df_test_2 = df_test_1.dropna(subset=['SK_ID_CURR']) 
del df_test_1

df_test_2.to_csv('df_test_bureau_preapp_pos.csv',index=False) 