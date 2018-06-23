# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:00:34 2018

@author: Maher Deeb
NOT FINISHED
"""

import pandas as pd
    


def decode_train_test(df_train_b,df_test_b):

    df_train_decoded = df_train_b
    df_test_decoded= df_test_b
        
    df_total = pd.concat([df_train_b,df_test_b])
    if 'SK_ID_CURR' in df_total.columns:
        df_total.index=range(len(df_total['SK_ID_CURR']))
    if 'SK_ID_PREV' in df_total.columns:
        df_total.index=range(len(df_total['SK_ID_PREV']))
    
    for col_i in df_train_b.columns[df_train_b.dtypes == 'object']:
        
        df_total[col_i] = df_total[col_i].factorize()[0]
        df_train_decoded[col_i] = df_total.loc[range(df_train_b.shape[0]),col_i].values
        df_test_decoded[col_i] =  df_total.loc[range(df_train_b.shape[0],df_train_b.shape[0]+df_test_b.shape[0]),col_i].values
        
    if 'SK_ID_CURR' in df_total.columns:    
        
        df_train_decoded = df_train_decoded.groupby(['SK_ID_CURR'], axis=0)[df_train_decoded.columns].mean()
        df_test_decoded = df_test_decoded.groupby(['SK_ID_CURR'], axis=0)[df_test_decoded.columns].mean()
        
    if 'SK_ID_PREV' in df_total.columns:    
        
        df_train_decoded = df_train_decoded.groupby(['SK_ID_PREV'], axis=0)[df_train_decoded.columns].mean()
        df_test_decoded = df_test_decoded.groupby(['SK_ID_PREV'], axis=0)[df_test_decoded.columns].mean()
        
        
    
    return df_train_decoded, df_test_decoded

df_train_b = pd.read_csv('df_train_bureau.csv')
df_test_b = pd.read_csv('df_test_bureau.csv')

df_train_b, df_test_b = decode_train_test(df_train_b,df_test_b)


df_train_p = pd.read_csv('df_train_pa.csv')
df_test_p = pd.read_csv('df_test_pa.csv')

df_train_p,df_test_p = decode_train_test(df_train_p,df_test_p)





