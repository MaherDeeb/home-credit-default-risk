# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:00:34 2018

@author: Maher Deeb
"""

import pandas as pd
    
df_train = pd.read_csv('df_train.csv')
df_bureau = pd.read_csv('df_bureau_decoded.csv')

        
df_train.index = df_train.SK_ID_CURR
df_bureau.index = df_bureau.SK_ID_CURR

df_bureau = df_bureau.drop(['SK_ID_CURR'],axis=1)

    
df_train = pd.concat([df_bureau,df_train],
                     axis=1, join_axes=[df_bureau.index])

df_train = df_train[~df_train['TARGET'].isnull()]
Y_train = df_train.TARGET

df_train.to_csv('df_train_bureau.csv',index=False) 
Y_train.to_csv('Y_train_o.csv',index=False)

del df_train,Y_train

df_test = pd.read_csv('df_test.csv',encoding='iso-8859-1')

df_test.index = df_test.SK_ID_CURR

df_test = pd.concat([df_bureau,df_test],axis=1,
                    join_axes=[df_bureau.index])


del df_bureau


df_test = df_test[~df_test['SK_ID_CURR'].isnull()]

df_id_submit=df_test.SK_ID_CURR


df_test.to_csv('df_test_bureau.csv',index=False) 
df_id_submit.to_csv('df_id_submit.csv',index=False)


