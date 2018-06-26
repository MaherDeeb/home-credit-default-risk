# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 17:45:26 2018

@author: Maher Deeb
"""

import pandas as pd
import numpy as np
data_path = 'D:/000_Projects_2018/0002_Development/Kaggle/home-credit-default-risk/data/'

def _extract_features(df):
    
    df_ohc = pd.get_dummies(df.NAME_CONTRACT_TYPE)
    df = pd.concat([df,df_ohc],axis = 1)
    df_ohc = pd.get_dummies(df.NAME_TYPE_SUITE)
    df = pd.concat([df,df_ohc],axis = 1)
    df_ohc = pd.get_dummies(df.NAME_INCOME_TYPE)
    df = pd.concat([df,df_ohc],axis = 1)
    
    
    df['incomeVSloan'] = df.AMT_INCOME_TOTAL/df.AMT_CREDIT
    df['incomeVSANNUITY'] = df.AMT_INCOME_TOTAL/df.AMT_ANNUITY
    df['loanVSANNUITY_y'] = df.AMT_CREDIT/df.AMT_ANNUITY
    df['loan-good'] = df.AMT_CREDIT - df.AMT_GOODS_PRICE
    df['loanVSgood'] = df.AMT_CREDIT/df.AMT_GOODS_PRICE
    
    df['EMPLOYEDvsBIRTH'] = df.DAYS_BIRTH/(df.DAYS_EMPLOYED-1)
    df['REGISTRATIONvsBIRTH'] = df.DAYS_BIRTH/(df.DAYS_REGISTRATION-1)
    df['PUBLISHvsBIRTH'] = df.DAYS_BIRTH/(df.DAYS_ID_PUBLISH-1)
    df['PUBLISHvsEMPLOYED'] = df.DAYS_EMPLOYED/(df.DAYS_ID_PUBLISH-1)
    df['PUBLISHvsREGISTRATION'] = df.DAYS_REGISTRATION/(df.DAYS_ID_PUBLISH-1)
    df['EMPLOYEDvsREGISTRATION'] = df.DAYS_REGISTRATION/(df.DAYS_EMPLOYED-1)
    df.OWN_CAR_AGE = df.OWN_CAR_AGE.fillna(0)
    df['OWN_CAR_AGEvsBIRTH'] = df.OWN_CAR_AGE/df.DAYS_BIRTH
    df['OWN_CAR_AGEvsREGISTRATION'] = df.OWN_CAR_AGE/(df.DAYS_REGISTRATION-1)
    df['OWN_CAR_AGEvsPUBLISH'] = df.OWN_CAR_AGE/(df.DAYS_ID_PUBLISH-1)
    df['OWN_CAR_AGEvsEMPLOYED'] = df.OWN_CAR_AGE/(df.DAYS_EMPLOYED-1)
    df['OWN_CAR_AGEvsloanVSANNUITY_y'] = df.OWN_CAR_AGE/(df.loanVSANNUITY_y-1)
    
    df['sum_FALG_1'] = df.FLAG_EMP_PHONE + df.FLAG_WORK_PHONE
    df['sum_FALG_2'] = df.FLAG_CONT_MOBILE + df.FLAG_PHONE + df.FLAG_EMAIL
    df['sum_FALG_3'] = df.sum_FALG_1 + df.sum_FALG_2
    
    df.OCCUPATION_TYPE.fillna('unknown')
    df.CNT_FAM_MEMBERS = df.CNT_FAM_MEMBERS.fillna(0)
    
    df['Family-childern'] = df.CNT_FAM_MEMBERS + df.CNT_CHILDREN
    df['Familyvschildern'] = df.CNT_FAM_MEMBERS / (df.CNT_CHILDREN+1)

    df['incomevschild'] = df.AMT_INCOME_TOTAL/(df.CNT_CHILDREN+1)
    
    df['score_region_compare_b'] = df.REGION_RATING_CLIENT > df.REGION_RATING_CLIENT_W_CITY
    df['score_region_compare_s'] = df.REGION_RATING_CLIENT + df.REGION_RATING_CLIENT_W_CITY

    df['weekend'] = ((df.WEEKDAY_APPR_PROCESS_START == 'SATURDAY') | (df.WEEKDAY_APPR_PROCESS_START == 'SUNDAY'))
    dayOfWeek={'MONDAY':1, 'TUESDAY':2, 'WEDNESDAY':0, 'THURSDAY':-2, 'FRIDAY':-1, 'SATURDAY':0.25, 'SUNDAY':0.75}
    df['weekday'] = df['WEEKDAY_APPR_PROCESS_START'].map(dayOfWeek)
    
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START >= 18,'weekday'].index)
    df.loc[time_index,'period'] = 'evening'
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START < 18,'weekday'].index)
    df.loc[time_index,'period'] = 'afternoon'
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START < 12,'weekday'].index)
    df.loc[time_index,'period'] = 'morning'
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START < 6,'weekday'].index)
    df.loc[time_index,'period'] = 'night'
    #df.period=df.period.factorize()[0]
    
    df['wrong_address'] = df.REG_REGION_NOT_LIVE_REGION + df.REG_REGION_NOT_WORK_REGION + \
    df.LIVE_REGION_NOT_WORK_REGION + df.REG_CITY_NOT_LIVE_CITY + df.REG_CITY_NOT_WORK_CITY + \
    df.LIVE_CITY_NOT_WORK_CITY
    
    df['industry'] = (df.ORGANIZATION_TYPE.str.contains('Industry'))
    df['trade'] = (df.ORGANIZATION_TYPE.str.contains('Trade'))
    df['Business'] = (df.ORGANIZATION_TYPE.str.contains('Business'))
    df['Transport'] = (df.ORGANIZATION_TYPE.str.contains('Business'))
    
    df['EXT_SOURCE_1_given'] = (df.EXT_SOURCE_1.isnull() == 0)*1
    df.EXT_SOURCE_1 = df.EXT_SOURCE_1.fillna(0)
    df['EXT_SOURCE_2_given'] = (df.EXT_SOURCE_2.isnull() == 0)*1
    df.EXT_SOURCE_2 = df.EXT_SOURCE_2.fillna(0)
    df['EXT_SOURCE_3_given'] = (df.EXT_SOURCE_3.isnull() == 0)*1
    df.EXT_SOURCE_3 = df.EXT_SOURCE_2.fillna(0)
    df['EXT_SOURCE_given'] = df['EXT_SOURCE_1_given'] + df['EXT_SOURCE_2_given'] \
    + df['EXT_SOURCE_3_given'] 
    
    df['EXT_SOURCE_1_b'] = 1*(df.EXT_SOURCE_1 > df.EXT_SOURCE_2) & (df.EXT_SOURCE_1 > df.EXT_SOURCE_3)
    df['EXT_SOURCE_2_b'] = 1*(df.EXT_SOURCE_2 > df.EXT_SOURCE_1) & (df.EXT_SOURCE_2 > df.EXT_SOURCE_3)    
    df['EXT_SOURCE_3_b'] = 1*(df.EXT_SOURCE_3 > df.EXT_SOURCE_1) & (df.EXT_SOURCE_3 > df.EXT_SOURCE_1)    

    df['sum_EXT_SOURCE'] = df.EXT_SOURCE_1 + df.EXT_SOURCE_2 + df.EXT_SOURCE_3
    
    df['building_info'] = df['APARTMENTS_AVG'].fillna(0)
    df['APARTMENTS_AVG'] = df['APARTMENTS_AVG'].fillna(0)
    
    for col_i in range(46,86):
        
        #print(df[df.columns[col_i]].head())
        
        df[df.columns[col_i]] = df[df.columns[col_i]].fillna(0)
        
        df['building_info'] = df['building_info'] + df[df.columns[col_i]]
        
    df.FONDKAPREMONT_MODE = df.FONDKAPREMONT_MODE.fillna('not given')
    df.HOUSETYPE_MODE = df.HOUSETYPE_MODE.fillna('not given')
    
    df.TOTALAREA_MODE = df.TOTALAREA_MODE.fillna(0)
    df['building_info'] = df['building_info'] + df['TOTALAREA_MODE']
    
    df.WALLSMATERIAL_MODE = df.WALLSMATERIAL_MODE.fillna('not given')
    df.EMERGENCYSTATE_MODE = df.EMERGENCYSTATE_MODE.fillna('not given')
    
    df['obs_30_given'] = (df.OBS_30_CNT_SOCIAL_CIRCLE.isnull())*1
    df['def_30_given'] = (df.DEF_30_CNT_SOCIAL_CIRCLE.isnull())*1
    df['30_given'] = df['obs_30_given']+df['def_30_given']
    
    df.OBS_30_CNT_SOCIAL_CIRCLE = df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(-1)
    df.DEF_30_CNT_SOCIAL_CIRCLE = df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(-1)
    
    df['obs_60_given'] = (df.OBS_60_CNT_SOCIAL_CIRCLE.isnull())*1
    df['def_60_given'] = (df.DEF_60_CNT_SOCIAL_CIRCLE.isnull())*1
    df['60_given'] = df['obs_60_given']+df['def_60_given']
    
    df.OBS_60_CNT_SOCIAL_CIRCLE = df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(-1)
    df.DEF_60_CNT_SOCIAL_CIRCLE = df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(-1)
    
    df['sum_30'] = df.OBS_30_CNT_SOCIAL_CIRCLE + df.DEF_30_CNT_SOCIAL_CIRCLE
    df['sum_60'] = df.OBS_60_CNT_SOCIAL_CIRCLE + df.DEF_60_CNT_SOCIAL_CIRCLE
    
    df['sum_30-60'] = df['sum_30'] + df['sum_60']
    df['30-60_given'] = df['60_given'] + df['30_given']

    df.DAYS_LAST_PHONE_CHANGE = df.DAYS_LAST_PHONE_CHANGE.fillna(1)
    
    df['phone_day_change'] = (df.DAYS_LAST_PHONE_CHANGE>=0)*1
    
    df['phonevsBIRTH'] = df.DAYS_BIRTH/(df.DAYS_LAST_PHONE_CHANGE-2)
    df['REGISTRATIONvsphone'] = df.DAYS_REGISTRATION/(df.DAYS_LAST_PHONE_CHANGE-2)
    df['PUBLISHvsphone'] = df.DAYS_ID_PUBLISH/(df.DAYS_LAST_PHONE_CHANGE-2)
    df['EMPLOYEDvsphone'] = df.DAYS_EMPLOYED/(df.DAYS_LAST_PHONE_CHANGE-2)

    df['AMT_REQ_hour_given'] = (df.AMT_REQ_CREDIT_BUREAU_HOUR.isnull())*1
    df['AMT_REQ_hour_0'] = (df.AMT_REQ_CREDIT_BUREAU_HOUR==0)*1

    df.AMT_REQ_CREDIT_BUREAU_HOUR = df.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(-1)

    df['AMT_REQ_day_given'] = (df.AMT_REQ_CREDIT_BUREAU_DAY.isnull())*1
    df['AMT_REQ_day_0'] = (df.AMT_REQ_CREDIT_BUREAU_DAY==0)*1

    df.AMT_REQ_CREDIT_BUREAU_DAY = df.AMT_REQ_CREDIT_BUREAU_DAY.fillna(-1)
    
    df['AMT_REQ_week_given'] = (df.AMT_REQ_CREDIT_BUREAU_WEEK.isnull())*1
    df['AMT_REQ_week_0'] = (df.AMT_REQ_CREDIT_BUREAU_WEEK==0)*1

    df.AMT_REQ_CREDIT_BUREAU_WEEK = df.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(-1)

    df['AMT_REQ_mon_given'] = (df.AMT_REQ_CREDIT_BUREAU_MON.isnull())*1
    df['AMT_REQ_mon_0'] = (df.AMT_REQ_CREDIT_BUREAU_MON==0)*1

    df.AMT_REQ_CREDIT_BUREAU_MON = df.AMT_REQ_CREDIT_BUREAU_MON.fillna(-1)
          
    df['AMT_REQ_qrt_given'] = (df.AMT_REQ_CREDIT_BUREAU_QRT.isnull())*1
    df['AMT_REQ_qrt_0'] = (df.AMT_REQ_CREDIT_BUREAU_QRT==0)*1

    df.AMT_REQ_CREDIT_BUREAU_QRT = df.AMT_REQ_CREDIT_BUREAU_QRT.fillna(-1)
    
    df['AMT_REQ_year_given'] = (df.AMT_REQ_CREDIT_BUREAU_YEAR.isnull())*1
    df['AMT_REQ_year_0'] = (df.AMT_REQ_CREDIT_BUREAU_YEAR==0)*1

    df.AMT_REQ_CREDIT_BUREAU_YEAR = df.AMT_REQ_CREDIT_BUREAU_YEAR.fillna(-1)
   
    df['AMT_REQ_sum'] = df.AMT_REQ_CREDIT_BUREAU_YEAR*365 + df.AMT_REQ_CREDIT_BUREAU_QRT*4*30 + df.AMT_REQ_CREDIT_BUREAU_MON*30 \
    + df.AMT_REQ_CREDIT_BUREAU_WEEK *7 + df.AMT_REQ_CREDIT_BUREAU_DAY
    
    
    df['AMT_REQ_sumvsBIRTH'] = df.DAYS_BIRTH/(df.AMT_REQ_sum+2)
    df['REGISTRATIONvsAMT_REQ_sum'] = df.DAYS_REGISTRATION/(df.AMT_REQ_sum+2)
    df['PUBLISHvsAMT_REQ_sum'] = df.DAYS_ID_PUBLISH/(df.AMT_REQ_sum+2)
    df['EMPLOYEDvsAMT_REQ_sum'] = df.DAYS_EMPLOYED/(df.AMT_REQ_sum+2)

    to_delete = ['OWN_CAR_AGE','APARTMENTS_MODE','BASEMENTAREA_MODE','BASEMENTAREA_MEDI','YEARS_BUILD_MODE',
             'YEARS_BUILD_MEDI','COMMONAREA_MODE','COMMONAREA_MEDI','ELEVATORS_MODE','ELEVATORS_MEDI',
             'FLOORSMIN_MODE','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE',
            'NONLIVINGAREA_MODE','NONLIVINGAREA_MEDI','DEF_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','FLAG_MOBIL','FLAG_DOCUMENT_2',
             'FLAG_DOCUMENT_4','FLAG_DOCUMENT_7','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14',
             'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    
    
    df=df.drop(to_delete,axis=1)

    #print(df[['EMPLOYEDvsAMT_REQ_sum','TARGET']].corr())
    
    return df

df_train = pd.read_csv(data_path+'application_train.csv')

df_train = _extract_features(df_train)

col_i=list(df_train.columns)

for col in col_i:
    try:
        if df_train[col].isin([ np.inf, -np.inf]).sum()>0:
            
            print(col,df_train[col].isin([np.nan, np.inf, -np.inf]).sum())
    except:
        print('does not work')

df_test = pd.read_csv(data_path+'application_test.csv')
df_test.insert(0, 'id', range(len(df_test['OWN_CAR_AGE'])))
df_test = _extract_features(df_test)
df_test=df_test.drop(['id'],axis=1)

col_i=list(df_test.columns)

for col in col_i:
    try:
        if df_test[col].isin([ np.inf, -np.inf]).sum()>0:
            
            print(col,df_test[col].isin([np.nan, np.inf, -np.inf]).sum())
    except:
        print('does not work')
        
df_test.to_csv('df_test.csv',index=False)
df_train.to_csv('df_train.csv',index=False)