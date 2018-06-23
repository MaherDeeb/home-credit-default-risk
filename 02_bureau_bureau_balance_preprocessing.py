# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:53:10 2018

@author: Maher Deeb
"""

import pandas as pd

data_path = 'D:/000_Projects_2018/0002_Development/Kaggle/home-credit-default-risk/data/'

df_bureau = pd.read_csv(data_path+'bureau.csv')

df_bureau_balance = pd.read_csv(data_path+'bureau_balance.csv')

df_bureau_balance['MONTHS_BALANCE'].unique()
df_bureau.index=df_bureau.SK_ID_BUREAU

for month_balance in df_bureau_balance['MONTHS_BALANCE'].unique():
    
    df_bureau_balance_new=df_bureau_balance[df_bureau_balance['MONTHS_BALANCE']==month_balance]
    df_bureau_balance_new.index=df_bureau_balance_new.SK_ID_BUREAU
    new_col = list(df_bureau.columns)
    new_col.append('mb'+str(month_balance))
    new_col.append('stat'+str(month_balance))
    df_bureau[new_col] = pd.concat([df_bureau,df_bureau_balance_new[['MONTHS_BALANCE','STATUS']]],axis=1, join_axes=[df_bureau.index])

df_bureau_decoded = df_bureau 

for col_i in df_bureau.columns[df_bureau.dtypes == 'object']:
    
    df_bureau_decoded[col_i] = df_bureau_decoded[col_i].factorize()[0]
    

df_bureau.to_csv('df_bureau.csv',index=False)
df_bureau_decoded.to_csv('df_bureau_decoded.csv',index=False)