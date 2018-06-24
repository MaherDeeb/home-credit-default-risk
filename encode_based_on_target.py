# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:14:42 2018

@author: Maher Deeb
"""

import pandas as pd

df_train = pd.read_csv('df_train_bureau.csv',encoding='iso-8859-1')

# need to be incoded
df_train.loc[df_train['NAME_CONTRACT_TYPE']=='Cash loans','NAME_CONTRACT_TYPE_Tencode'] \
 = df_train[df_train['NAME_CONTRACT_TYPE']=='Cash loans'].TARGET.sum()/ df_train.TARGET.sum() 
df_train.loc[df_train['NAME_CONTRACT_TYPE']=='Revolving loans','NAME_CONTRACT_TYPE_Tencode'] \
= df_train[df_train['NAME_CONTRACT_TYPE']=='Revolving loans'].TARGET.sum()/df_train.TARGET.sum() 

df_train.loc[df_train['CODE_GENDER']==0,'CODE_GENDER_Tencode'] \
 = df_train[df_train['CODE_GENDER']==0].TARGET.sum()  
df_train.loc[df_train['CODE_GENDER']==1,'CODE_GENDER_Tencode'] \
 = df_train[df_train['CODE_GENDER']==1].TARGET.sum()  
df_train.loc[df_train['CODE_GENDER']==2,'CODE_GENDER_Tencode'] \
 = df_train[df_train['CODE_GENDER']==2].TARGET.sum()  

df_train.loc[df_train['FLAG_OWN_CAR']==0,'FLAG_OWN_CAR_Tencode'] \
 = df_train[df_train['FLAG_OWN_CAR']==0].TARGET.sum()  
df_train.loc[df_train['FLAG_OWN_CAR']==1,'FLAG_OWN_CAR_Tencode'] \
= df_train[df_train['FLAG_OWN_CAR']==1].TARGET.sum()  

df_train.loc[df_train['FLAG_OWN_REALTY']==0,'FLAG_OWN_REALTY_Tencode'] \
 = df_train[df_train['FLAG_OWN_REALTY']==0].TARGET.sum()  
df_train.loc[df_train['FLAG_OWN_REALTY']==1,'FLAG_OWN_REALTY_Tencode'] \
= df_train[df_train['FLAG_OWN_REALTY']==1].TARGET.sum()  

 

#df_test = pd.read_csv('df_test_bureau.csv',encoding='iso-8859-1')

    df_train = pd.read_csv('df_train.csv')
    df_test = pd.read_csv('df_test.csv')
    
    for col_i in df_train.columns:
        
        values_all = df_train[col_i].unique()
        
        if len(values_all)<100 and col_i != 'TARGET':
        
            for values_i in values_all:
            
            
                df_train.loc[df_train[col_i]==values_i,col_i+'_Tencode'] \
                = df_train[df_train[col_i]==values_i].TARGET.sum()/ df_train.TARGET.sum() 
                
                df_test.loc[df_test[col_i]==values_i,col_i+'_Tencode'] \
                = df_train[df_train[col_i]==values_i].TARGET.sum()/ df_train.TARGET.sum()