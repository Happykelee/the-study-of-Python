# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:16:44 2018

@author: Caizhengting
"""
from pandas import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def PRE(fname):
    
    df = read_csv(fname, header = None)
    cols = list(df.iloc[0:3,:].as_matrix());df.columns = cols
    cols = ['_'.join(col[::-1]) for col in list(df.columns)];df.columns = cols
    
    df = df.iloc[3:,:]
    df = df.apply(to_numeric, errors='ignore')
    df.index = df.custom_custom_patient_id;df.index = df.index.rename('MLID')
    df = df.sort_index()
    return(df)
    
def get_X_all(df):
    for i in range(len(df.columns)) :
        if not('custom' in df.columns[i]) : break
    X_all = df.iloc[:,i:]
    return(X_all)
    
def change_Y_all(y_all):
    y_all.columns = ['label']
    y_all.index = y_all.index.rename('MLID')
    y_all = y_all.sort_index()
    return(y_all)
    
def get_Sd(X_all):
    scaler = StandardScaler().fit(X_all)
    X_Sd = scaler.transform(X_all)
    X_Sd = DataFrame(X_Sd, columns = X_all.columns, index=X_all.index)
    return(X_Sd)
    
def get_CV_num(y_all,run=1000,n_split=5):
    writer_train = ExcelWriter('ID_Train_CV.xlsx')
    writer_test = ExcelWriter('ID_Test_CV.xlsx')
    
    for num in range(run):
        kf = StratifiedKFold(n_splits=n_split, shuffle=True) 
        ID_train = DataFrame()
        ID_test = DataFrame()
        # Stratified k-fold
        i = 1
        for train, test in kf.split(y_all,y_all.label): 
            train = DataFrame(y_all.iloc[train,:].index.rename(str(i)))
            test = DataFrame(y_all.iloc[test,:].index.rename(str(i)))
            ID_train = concat([ID_train, train], axis = 1)
            ID_test = concat([ID_test, test], axis = 1)
            i = i + 1
        ID_train.to_excel(writer_train, str(num))
        ID_test.to_excel(writer_test, str(num))
    writer_train.save()
    writer_test.save()
    
    
    
    