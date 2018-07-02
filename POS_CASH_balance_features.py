# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 22:23:39 2018

@author: Maher Deeb
"""

import pandas as pd
import numpy as np
data_path = 'D:/000_Projects_2018/0002_Development/Kaggle/home-credit-default-risk/data/'

def _extract_features(df,df_train):
    
    df_ohc = pd.get_dummies(df.MONTHS_BALANCE)
    col = df_ohc.columns
    df_ohc.columns = [str(x) + "MONTHS_BALANCE" for x in col]
    df = pd.concat([df,df_ohc],axis = 1)
    
    df.CNT_INSTALMENT = df.CNT_INSTALMENT.fillna(0)

    df_ohc = pd.get_dummies(df.CNT_INSTALMENT)
    col = df_ohc.columns
    df_ohc.columns = [str(x) + "CNT_INSTALMENT" for x in col]
    df = pd.concat([df,df_ohc],axis = 1)
    
    df.CNT_INSTALMENT_FUTURE = df.CNT_INSTALMENT_FUTURE.fillna(-1)
    
    df['INS_curr_futu'] = df.CNT_INSTALMENT + df.CNT_INSTALMENT_FUTURE
    df['INS_currvsfutu'] = df.CNT_INSTALMENT/(1+df.CNT_INSTALMENT_FUTURE)


    df_ohc = pd.get_dummies(df.CNT_INSTALMENT_FUTURE)
    col = df_ohc.columns
    df_ohc.columns = [str(x) + "CNT_INSTALMENT_FUTURE" for x in col]
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.NAME_CONTRACT_STATUS)
    col = df_ohc.columns
    df_ohc.columns = [str(x) + "NAME_CONTRACT_STATUS" for x in col]
    df = pd.concat([df,df_ohc],axis = 1)
    
    df['SK_DPD_30'] = (df.SK_DPD >= 30)*1
    df['SK_DPD_90'] = ((df.SK_DPD >= 90) & (df.SK_DPD < 30))*1
    df['SK_DPD_180'] = ((df.SK_DPD >= 180) & (df.SK_DPD < 90))*1
    df['SK_DPD_365'] = ((df.SK_DPD >= 365) & (df.SK_DPD < 180))*1
    df['SK_DPD_720'] = ((df.SK_DPD >= 720) & (df.SK_DPD < 365))*1
    df['SK_DPD_1400'] = ((df.SK_DPD >= 1400) & (df.SK_DPD < 720))*1
    df['SK_DPD_2800'] = ((df.SK_DPD >= 2800))*1
    
    df['SK_DPD_DEF_30'] = (df.SK_DPD_DEF >= 30)*1
    df['SK_DPD_DEF_90'] = ((df.SK_DPD_DEF >= 90) & (df.SK_DPD_DEF < 30))*1
    df['SK_DPD_DEF_180'] = ((df.SK_DPD_DEF >= 180) & (df.SK_DPD_DEF < 90))*1
    df['SK_DPD_DEF_365'] = ((df.SK_DPD_DEF >= 365) & (df.SK_DPD_DEF < 180))*1
    df['SK_DPD_DEF_720'] = ((df.SK_DPD_DEF >= 720) & (df.SK_DPD_DEF < 365))*1
    df['SK_DPD_DEF_1400'] = ((df.SK_DPD_DEF >= 1400) & (df.SK_DPD_DEF < 720))*1
    df['SK_DPD_DEF_2800'] = ((df.SK_DPD_DEF >= 2800))*1
    
    df['SK_DPD_DEF_SK_DPD'] = df.SK_DPD_DEF + df.SK_DPD
    df['SK_DPD_DEFvsSK_DPD'] = df.SK_DPD_DEF/(1+df.SK_DPD)
    
    for col_i in df.columns[df.dtypes == 'object']:
#         
        df[col_i] = df[col_i].factorize()[0]
        
    for col_i in df.columns:
        
        if sum(df[col_i]) is None:
            
            df = df.drop([col_i],axis=1)
#     
    Y_train = df_train.TARGET
    
    df.index = df.SK_ID_CURR
#     
    df = pd.concat([Y_train,df],axis = 1,join_axes=[df.index])
#     
    df = df[~df.TARGET.isnull()]
#     
    df.index = range(len(df.SK_ID_CURR))
    df_agg = df.groupby(['SK_ID_CURR'], axis=0)[df.columns].sum()

    
    print(abs(df_agg.corr().TARGET))
    print(sum(abs(df_agg.corr().TARGET)))

    return df

df_train = pd.read_csv(data_path+'application_train.csv')

df_POS = pd.read_csv(data_path+'POS_CASH_balance.csv')
#
df_POS = _extract_features(df_POS,df_train)

df_POS.to_csv('df_POS_decoded.csv',index=False) 
