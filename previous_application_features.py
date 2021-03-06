# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 01:06:13 2018

@author: Maher Deeb
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:43:57 2018

@author: Maher Deeb
"""

import pandas as pd
import numpy as np
data_path = 'D:/000_Projects_2018/0002_Development/Kaggle/home-credit-default-risk/data/'

def _extract_features(df,df_train):
    
# =============================================================================
    df_ohc = pd.get_dummies(df.NAME_CONTRACT_TYPE)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.WEEKDAY_APPR_PROCESS_START)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.HOUR_APPR_PROCESS_START)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df['AMT_ANNUITY_not_given'] = (df.AMT_ANNUITY.isnull())*1
    df['AMT_ANNUITY_0'] = (df.AMT_ANNUITY==0)*1
    df.AMT_ANNUITY = df.AMT_ANNUITY.fillna(0)
    df['AMT_ANNUITY_0_not_given'] = df['AMT_ANNUITY_not_given'] + df['AMT_ANNUITY_0']
    df['AMT_ANNUITY_log'] = np.log(df.AMT_ANNUITY+1)
    df['AMT_ANNUITY_0.5'] = np.sqrt(df.AMT_ANNUITY)
    df['AMT_ANNUITY_2'] = np.square(df.AMT_ANNUITY/df.AMT_ANNUITY.max())
    
    df['AMT_APPLICATION_0'] = (df.AMT_APPLICATION==0)*1
    df['AMT_APPLICATION_log'] = np.log(df.AMT_APPLICATION+1)
    df['AMT_APPLICATION_0.5'] = np.sqrt(df.AMT_APPLICATION)
    
    df['AMT_ANNUITYvsAMT_APPLICATION'] = df.AMT_APPLICATION/(df.AMT_ANNUITY+1)
    
    df['AMT_CREDIT_0'] = (df.AMT_CREDIT==0)*1
    df['AMT_CREDIT_log'] = np.log(df.AMT_CREDIT+1)
    df['AMT_CREDIT_0.5'] = np.sqrt(df.AMT_CREDIT)
    df['AMT_CREDIT_2'] = np.square(df.AMT_CREDIT/df.AMT_CREDIT.max())

    df['AMT_ANNUITYvsAMT_CREDIT'] = df.AMT_CREDIT/(df.AMT_ANNUITY+1)
    
    df['AMT_CREDITvsAMT_APPLICATION'] = df.AMT_APPLICATION/(df.AMT_CREDIT+1)
    df['AMT_CREDITvsAMT_APPLICATION_L'] = (df['AMT_CREDITvsAMT_APPLICATION']>1)*1
    df['AMT_CREDITvsAMT_APPLICATION_S'] = (df['AMT_CREDITvsAMT_APPLICATION']<1)*1

    df['AMT_DOWN_PAYMENT_not_given'] = (df.AMT_DOWN_PAYMENT.isnull())*1
    df['AMT_DOWN_PAYMENT_0'] = (df.AMT_DOWN_PAYMENT==0)*1
    df.AMT_DOWN_PAYMENT = df.AMT_DOWN_PAYMENT.fillna(0)
    df['AMT_DOWN_PAYMENT_0_not_given'] = df['AMT_DOWN_PAYMENT_not_given'] + df['AMT_DOWN_PAYMENT_0']
    df['AMT_DOWN_PAYMENT_log'] = np.log(df.AMT_DOWN_PAYMENT+1)
    df['AMT_DOWN_PAYMENT_0.5'] = np.sqrt(df.AMT_DOWN_PAYMENT)
    df['AMT_DOWN_PAYMENT_2'] = np.square(df.AMT_DOWN_PAYMENT/df.AMT_DOWN_PAYMENT.max())
    
    df['AMT_ANNUITYvsAMT_DOWN_PAYMENT'] = df.AMT_DOWN_PAYMENT/(df.AMT_ANNUITY+1)
    df['AMT_ANNUITY-AMT_DOWN_PAYMENT'] = df.AMT_DOWN_PAYMENT+df.AMT_ANNUITY
    df['AMT_CREDITvsAMT_DOWN_PAYMENT'] = df.AMT_DOWN_PAYMENT/(df.AMT_CREDIT+1)
    df['AMT_APPLICATIONvsAMT_DOWN_PAYMENT'] = df.AMT_DOWN_PAYMENT/(df.AMT_APPLICATION+1)
    
    df['AMT_GOODS_PRICE_not_given'] = (df.AMT_GOODS_PRICE.isnull())*1
    df['AMT_GOODS_PRICE_0'] = (df.AMT_GOODS_PRICE==0)*1
    df.AMT_GOODS_PRICE = df.AMT_GOODS_PRICE.fillna(0)
    df['AMT_GOODS_PRICE_0_not_given'] = df['AMT_GOODS_PRICE_not_given'] + df['AMT_GOODS_PRICE_0']
    df['AMT_GOODS_PRICE_log'] = np.log(df.AMT_GOODS_PRICE+1)
    df['AMT_GOODS_PRICE_0.5'] = np.sqrt(df.AMT_GOODS_PRICE)
    df['AMT_GOODS_PRICE_2'] = np.square(df.AMT_GOODS_PRICE/df.AMT_GOODS_PRICE.max())

    df['AMT_ANNUITYvsAMT_GOODS_PRICE'] = df.AMT_GOODS_PRICE/(df.AMT_ANNUITY+1)
    df['AMT_CREDITvsAMT_GOODS_PRICE'] = df.AMT_GOODS_PRICE/(df.AMT_CREDIT+1)
    df['AMT_APPLICATIONvsAMT_GOODS_PRICE'] = df.AMT_GOODS_PRICE/(df.AMT_APPLICATION+1)
    df['AMT_DOWN_PAYMENTvsAMT_GOODS_PRICE'] = df.AMT_GOODS_PRICE/(df.AMT_DOWN_PAYMENT+1)
    
    df['weekend_p'] = ((df.WEEKDAY_APPR_PROCESS_START == 'SATURDAY') | (df.WEEKDAY_APPR_PROCESS_START == 'SUNDAY'))
    dayOfWeek={'MONDAY':1, 'TUESDAY':2, 'WEDNESDAY':0, 'THURSDAY':-2, 'FRIDAY':-1, 'SATURDAY':0.25, 'SUNDAY':0.75}
    df['weekday_p'] = df['WEEKDAY_APPR_PROCESS_START'].map(dayOfWeek)
    
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START >= 18,'weekday_p'].index)
    df.loc[time_index,'period_p'] = 'evening'
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START < 18,'weekday_p'].index)
    df.loc[time_index,'period_p'] = 'afternoon'
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START < 12,'weekday_p'].index)
    df.loc[time_index,'period_p'] = 'morning'
    time_index = list(df.loc[df.HOUR_APPR_PROCESS_START < 6,'weekday_p'].index)
    df.loc[time_index,'period_p'] = 'night'
    
    df_ohc = pd.get_dummies(df.period_p)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df.FLAG_LAST_APPL_PER_CONTRACT = (df.FLAG_LAST_APPL_PER_CONTRACT =='Y')*1
    df['flag_contract_flag_day_0'] = (df.FLAG_LAST_APPL_PER_CONTRACT==0)*1 + (df.NFLAG_LAST_APPL_IN_DAY==0)*1
    
    df['RATE_DOWN_PAYMENT_not_given'] = (df.RATE_DOWN_PAYMENT.isnull())*1
    df['RATE_DOWN_PAYMENT_0'] = (df.RATE_DOWN_PAYMENT==0)*1
    df.RATE_DOWN_PAYMENT = df.RATE_DOWN_PAYMENT.fillna(0)
    df['RATE_DOWN_PAYMENT_0_not_given'] = df['RATE_DOWN_PAYMENT_not_given'] + df['RATE_DOWN_PAYMENT_0']
    df['RATE_DOWN_PAYMENT_log'] = np.log(df.RATE_DOWN_PAYMENT+1)
    df['RATE_DOWN_PAYMENT_0.5'] = np.sqrt(df.RATE_DOWN_PAYMENT)
    df['RATE_DOWN_PAYMENT_2'] = np.square(df.RATE_DOWN_PAYMENT/df.RATE_DOWN_PAYMENT.max())
    
    df['RATE_INTEREST_PRIMARY_not_given'] = (df.RATE_INTEREST_PRIMARY.isnull())*1
    df.RATE_INTEREST_PRIMARY = df.RATE_INTEREST_PRIMARY.fillna(0)
    df['RATE_INTEREST_PRIMARY_log'] = np.log(df.RATE_INTEREST_PRIMARY+1)
    df['RATE_INTEREST_PRIMARY_0.5'] = np.sqrt(df.RATE_INTEREST_PRIMARY)
    df['RATE_INTEREST_PRIMARY_2'] = np.square(df.RATE_INTEREST_PRIMARY/df.RATE_INTEREST_PRIMARY.max())
    
    df['rate_INTEREST-DOWN'] = df.RATE_INTEREST_PRIMARY + df.RATE_DOWN_PAYMENT
    df['rate_INTERESTvsDOWN'] = df.RATE_INTEREST_PRIMARY /(1+df.RATE_DOWN_PAYMENT)
    
    df['not_given'] = df.RATE_DOWN_PAYMENT_not_given + df.RATE_INTEREST_PRIMARY_not_given+df.AMT_GOODS_PRICE_not_given+ \
    df.AMT_DOWN_PAYMENT_not_given + df.AMT_ANNUITY_0_not_given
    
    df_ohc = pd.get_dummies(df.NAME_CONTRACT_STATUS)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.NAME_CASH_LOAN_PURPOSE)
    df = pd.concat([df,df_ohc],axis = 1)


    df['DAYS_DECISION_30'] = (df.DAYS_DECISION >= -30)*1
    df['DAYS_DECISION_90'] = ((df.DAYS_DECISION >= -90) & (df.DAYS_DECISION < -30))*1
    df['DAYS_DECISION_180'] = ((df.DAYS_DECISION >= -180) & (df.DAYS_DECISION < -90))*1
    df['DAYS_DECISION_365'] = ((df.DAYS_DECISION >= -365) & (df.DAYS_DECISION < -180))*1
    df['DAYS_DECISION_720'] = ((df.DAYS_DECISION >= -720) & (df.DAYS_DECISION < -365))*1
    df['DAYS_DECISION_1400'] = ((df.DAYS_DECISION >= -1400) & (df.DAYS_DECISION < -720))*1
    df['DAYS_DECISION_2800'] = ((df.DAYS_DECISION >= -2800) & (df.DAYS_DECISION < -1400))*1
    df['DAYS_DECISION_L2800'] = (df.DAYS_DECISION < -2800)*1
    
    df_ohc = pd.get_dummies(df.NAME_PAYMENT_TYPE)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.CODE_REJECT_REASON)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df['NAME_TYPE_SUITE_not_given'] = (df.NAME_TYPE_SUITE.isnull())*1
    df.NAME_TYPE_SUITE = df.NAME_TYPE_SUITE.fillna('not given')
    
    df_ohc = pd.get_dummies(df.NAME_TYPE_SUITE)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.NAME_CLIENT_TYPE)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.NAME_GOODS_CATEGORY)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df_ohc = pd.get_dummies(df.NAME_PORTFOLIO)
    df = pd.concat([df,df_ohc],axis = 1)
    
    df['NFLAG_INSURED_ON_APPROVAL_not_given'] = (df.NFLAG_INSURED_ON_APPROVAL.isnull())*1
    df.NFLAG_INSURED_ON_APPROVAL = df.NFLAG_INSURED_ON_APPROVAL.fillna(-1)

    

# 
#     df['CREDIT_DAY_OVERDUE0'] = (df.CREDIT_DAY_OVERDUE == 0)*1
#     df['CREDIT_DAY_OVERDUEvsDAYS_CREDIT'] = df.CREDIT_DAY_OVERDUE/(df.DAYS_CREDIT-1)
#     df['CREDIT_DAY_OVERDUE-DAYS_CREDIT'] = df.CREDIT_DAY_OVERDUE-df.DAYS_CREDIT
#     df['CREDIT_DAY_OVERDUE--DAYS_CREDIT'] = df.CREDIT_DAY_OVERDUE+df.DAYS_CREDIT
#     
#     df['DAYS_CREDIT_ENDDATE_notgiven'] = (df.DAYS_CREDIT_ENDDATE.isnull())*1
#     ix = list(df.loc[(df.CREDIT_ACTIVE == 'Active') & (df.DAYS_CREDIT_ENDDATE.isnull())].index)
#     df.loc[ix,'DAYS_CREDIT_ENDDATE'] = df.loc[df.DAYS_CREDIT_ENDDATE>0,'DAYS_CREDIT_ENDDATE'].mean()
#     ix = list(df.loc[~(df.CREDIT_ACTIVE == 'Active') & (df.DAYS_CREDIT_ENDDATE.isnull())].index)
#     df.loc[ix,'DAYS_CREDIT_ENDDATE'] = df.loc[df.DAYS_CREDIT_ENDDATE<0,'DAYS_CREDIT_ENDDATE'].mean()
#     
#     df['DAYS_CREDIT_ENDDATEvsDAYS_CREDIT'] = df.DAYS_CREDIT_ENDDATE/abs(df.DAYS_CREDIT-1)
#     df['DAYS_CREDIT_ENDDATE-DAYS_CREDIT'] = df.DAYS_CREDIT_ENDDATE+abs(df.DAYS_CREDIT)
#     df['DAYS_CREDIT_ENDDATEvsCREDIT_DAY_OVERDUE'] = df.DAYS_CREDIT_ENDDATE/(df.CREDIT_DAY_OVERDUE+1)
#     
#     df['DAYS_ENDDATE_FACTvsend_e'] = df.DAYS_ENDDATE_FACT == df.DAYS_CREDIT_ENDDATE
#     df['DAYS_ENDDATE_FACTvsend_b'] = df.DAYS_ENDDATE_FACT > df.DAYS_CREDIT_ENDDATE
#     df['DAYS_ENDDATE_FACTvsend_l'] = df.DAYS_ENDDATE_FACT < df.DAYS_CREDIT_ENDDATE
#     
#     df.DAYS_ENDDATE_FACT = df.DAYS_ENDDATE_FACT.fillna(0)
#     
#     df['DAYS_CREDIT_ENDDATE_30'] = (-df.DAYS_CREDIT_ENDDATE >= -30)*1
#     df['DAYS_CREDIT_ENDDATE_90'] = ((-df.DAYS_CREDIT_ENDDATE >= -90) & (-df.DAYS_CREDIT_ENDDATE < -30))*1
#     df['DAYS_CREDIT_ENDDATE_180'] = ((-df.DAYS_CREDIT_ENDDATE >= -180) & (-df.DAYS_CREDIT_ENDDATE < -90))*1
#     df['DAYS_CREDIT_ENDDATE_365'] = ((-df.DAYS_CREDIT_ENDDATE >= -365) & (-df.DAYS_CREDIT_ENDDATE < -180))*1
#     df['DAYS_CREDIT_ENDDATE_720'] = ((-df.DAYS_CREDIT_ENDDATE >= -720) & (-df.DAYS_CREDIT_ENDDATE < -365))*1
#     df['DAYS_CREDIT_ENDDATE_1400'] = ((-df.DAYS_CREDIT_ENDDATE >= -1400) & (-df.DAYS_CREDIT_ENDDATE < -720))*1
#     df['DAYS_CREDIT_ENDDATE_2800'] = ((-df.DAYS_CREDIT_ENDDATE >= -2800) & (-df.DAYS_CREDIT_ENDDATE < -1400))*1
#     df['DAYS_CREDIT_ENDDATE_L2800'] = (-df.DAYS_CREDIT_ENDDATE < -2800)*1
# 
#     df['AMT_CREDIT_MAX_OVERDUE_notgiven'] = (df.AMT_CREDIT_MAX_OVERDUE.isnull())*1
#     df['AMT_CREDIT_MAX_OVERDUE_0'] = (df.AMT_CREDIT_MAX_OVERDUE == 0)*1
#     df.AMT_CREDIT_MAX_OVERDUE = df.AMT_CREDIT_MAX_OVERDUE.fillna(0)
#     df['AMT_CREDIT_MAX_OVERDUE_vstotal_duration'] = df.AMT_CREDIT_MAX_OVERDUE/(df['DAYS_CREDIT_ENDDATE-DAYS_CREDIT']+1)
#     df['AMT_CREDIT_MAX_OVERDUE_vsover_duration']= df.AMT_CREDIT_MAX_OVERDUE/(df.CREDIT_DAY_OVERDUE+1)
#     
#     df['CNT_CREDIT_PROLONG_0'] = (df.CNT_CREDIT_PROLONG == 0)*1
#     df['CNT_CREDIT_PROLONG_fake_0'] = ((df.CNT_CREDIT_PROLONG == 0) & ((df.AMT_CREDIT_MAX_OVERDUE>0) | (df.CREDIT_DAY_OVERDUE>0)))*1
#     
#     df.AMT_CREDIT_SUM = df.AMT_CREDIT_SUM.fillna(0)
#     df['AMT_CREDIT_SUM0'] = (df.AMT_CREDIT_SUM ==0)*1
#     
#     df['sumvsoverdue'] = df.AMT_CREDIT_MAX_OVERDUE/(df.AMT_CREDIT_SUM+1)
#     df['sum_vstotal_duration'] = df.AMT_CREDIT_SUM/(df['DAYS_CREDIT_ENDDATE-DAYS_CREDIT']+1)
#     df['sum_vsover_duration']= df.AMT_CREDIT_SUM/(df.CREDIT_DAY_OVERDUE+1)
# 
#     df['AMT_CREDIT_SUM_OVERDUE0'] = (df.AMT_CREDIT_SUM_OVERDUE==0)*1
#     df['AMT_CREDIT_SUM_OVERDUEvssum'] = df.AMT_CREDIT_SUM_OVERDUE/(1+df.AMT_CREDIT_SUM)
#     df['AMT_CREDIT_SUM_OVERDUEvsperiod'] = df.AMT_CREDIT_SUM_OVERDUE/(df['DAYS_CREDIT_ENDDATE-DAYS_CREDIT']+1)
#     df['AMT_CREDIT_SUM_OVERDUEvsmaxoverdue'] = df.AMT_CREDIT_SUM_OVERDUE/(df.AMT_CREDIT_MAX_OVERDUE+1)
#     
#     df['AMT_CREDIT_SUM_LIMIT_notgiven'] = (df.AMT_CREDIT_SUM_LIMIT.isnull())*1
#     df['AMT_CREDIT_SUM_LIMIT0'] = (df.AMT_CREDIT_SUM_LIMIT==0)*1
#     df.AMT_CREDIT_SUM_LIMIT = df.AMT_CREDIT_SUM_LIMIT.fillna(0)
#     df['AMT_CREDIT_SUM_LIMITvssum'] = df.AMT_CREDIT_SUM_LIMIT/(1+df.AMT_CREDIT_SUM)
#     df['AMT_CREDIT_SUM_LIMITvsperiod'] = df.AMT_CREDIT_SUM_LIMIT/(df['DAYS_CREDIT_ENDDATE-DAYS_CREDIT']+1)
#     df['AMT_CREDIT_SUM_LIMITvsmaxoverdue'] = df.AMT_CREDIT_SUM_LIMIT/(df.AMT_CREDIT_MAX_OVERDUE+1)
#     
#     
#     df['AMT_CREDIT_SUM_DEBT_notgiven'] = (df.AMT_CREDIT_SUM_DEBT.isnull())*1
#     df['AMT_CREDIT_SUM_DEBT0'] = (df.AMT_CREDIT_SUM_DEBT==0)*1
#     df.AMT_CREDIT_SUM_DEBT = df.AMT_CREDIT_SUM_DEBT.fillna(0)
#     df['AMT_CREDIT_SUM_DEBTvssum'] = df.AMT_CREDIT_SUM_DEBT/(1+df.AMT_CREDIT_SUM)
#     df['AMT_CREDIT_SUM_DEBTvsperiod'] = df.AMT_CREDIT_SUM_DEBT/(df['DAYS_CREDIT_ENDDATE-DAYS_CREDIT']+1)
#     df['AMT_CREDIT_SUM_DEBTvsmaxoverdue'] = df.AMT_CREDIT_SUM_DEBT/(df.AMT_CREDIT_MAX_OVERDUE+1)
#    
# 
# 
#     df_ohc = pd.get_dummies(df.CREDIT_TYPE) 
#     df = pd.concat([df,df_ohc],axis = 1)
#     
#     df['DAYS_CREDIT_UPDATE_30'] = (df.DAYS_CREDIT_UPDATE >= -30)*1
#     df['DAYS_CREDIT_UPDATE_90'] = ((df.DAYS_CREDIT_UPDATE >= -90) & (df.DAYS_CREDIT_UPDATE < -30))*1
#     df['DAYS_CREDIT_UPDATE_180'] = ((df.DAYS_CREDIT_UPDATE >= -180) & (df.DAYS_CREDIT_UPDATE < -90))*1
#     df['DAYS_CREDIT_UPDATE_365'] = ((df.DAYS_CREDIT_UPDATE >= -365) & (df.DAYS_CREDIT_UPDATE < -180))*1
#     df['DAYS_CREDIT_UPDATE_720'] = ((df.DAYS_CREDIT_UPDATE >= -720) & (df.DAYS_CREDIT_UPDATE < -365))*1
#     df['DAYS_CREDIT_UPDATE_1400'] = ((df.DAYS_CREDIT_UPDATE >= -1400) & (df.DAYS_CREDIT_UPDATE < -720))*1
#     df['DAYS_CREDIT_UPDATE_2800'] = ((df.DAYS_CREDIT_UPDATE >= -2800) & (df.DAYS_CREDIT_UPDATE < -1400))*1
#     df['DAYS_CREDIT_UPDATE_L2800'] = (df.DAYS_CREDIT_UPDATE < -2800)*1
# 
#     df['AMT_ANNUITY0'] = (df.AMT_ANNUITY ==0)*1
#     df['AMT_ANNUITY_notgiven'] = (df.AMT_ANNUITY.isnull())*1
#     df['AMT_ANNUITY_notgiven-0'] = df['AMT_ANNUITY0'] + df['AMT_ANNUITY_notgiven'] 
#     df.AMT_ANNUITY = df.AMT_ANNUITY.fillna(0)
#     df['sumvsannuty'] = df.AMT_ANNUITY/(df.AMT_CREDIT_SUM+1)
#     df['overduevsannuty'] = df.AMT_CREDIT_MAX_OVERDUE/(df.AMT_ANNUITY+1)
# =============================================================================
    
# =============================================================================
    #df_train.index = df_train.SK_ID_CURR
#     
#     try:
    #Y_train = df_train.TARGET
#         
#     except:
#         df_train['TARGET'] = df_train.SK_ID_CURR
#         Y_train = df_train.TARGET
#         
    #df.index = df.SK_ID_CURR
#     
    #df = pd.concat([Y_train,df],axis = 1,join_axes=[df.index])
#     
    #df = df[~df.TARGET.isnull()]
#     
#     df.index = range(len(df.SK_ID_CURR))
#     print(df.head())
# =============================================================================
    
# =============================================================================
    for col_i in df.columns[df.dtypes == 'object']:
#         
        df[col_i] = df[col_i].factorize()[0]
#     
    #df_agg = df.groupby(['SK_ID_CURR'], axis=0)[df.columns].sum()
#     
#     col_i=list(df_agg.columns)
# =============================================================================
    
    #for col in col_i:
    
   # print(abs(df_agg.corr().TARGET))
    #print(sum(abs(df_agg.corr().TARGET)))

    #df_agg=df_agg.drop(['TARGET'],axis=1)
    
# =============================================================================
#     df_agg.index = range(len(df_agg.SK_ID_CURR))
# =============================================================================
    
    return df

df_train = pd.read_csv(data_path+'application_train.csv')

df_preapp = pd.read_csv(data_path+'previous_application.csv')
#
df_preapp = _extract_features(df_preapp,df_train)

df_preapp.to_csv('df_preapp_decoded.csv',index=False) 

#df_test = pd.read_csv(data_path+'application_test.csv')

#df_bureau_test = _extract_features(df_bureau,df_test)

#df_bureau_test.to_csv('df_test_bureau.csv',index=False) 

# =============================================================================
# col_i=list(df_bureau.columns)
# 
# for col in col_i:
#     try:
#         if df_bureau[col].isin([ np.inf, -np.inf]).sum()>0:
#             
#             print(col,df_bureau[col].isin([np.nan, np.inf, -np.inf]).sum())
#     except:
#         print('does not work')
# =============================================================================
