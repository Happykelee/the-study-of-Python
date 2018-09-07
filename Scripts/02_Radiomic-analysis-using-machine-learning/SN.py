# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:10:56 2018

@author: Caizhengting

<small but necessary>

"""
import os
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def ICC(table):
    if type(table) == type(DataFrame()):
        table = array(table)
    n, k = shape(table)
    mean_r = table.mean(axis = 1)
    mean_c = table.mean(axis = 0)
    mean_all = table.mean().mean()

    l_all = square(table-mean_all).sum()
    l_r = square(mean_r-mean_all).sum()*k
    l_c = square(mean_c-mean_all).sum()*n
    l_e = l_all - l_r - l_c

    # v_all = n*k-1
    v_r = n-1
    v_c = k-1
    v_e = v_r*v_c

    MSR = l_r/v_r
    MSC = l_c/v_c
    MSE = l_e/v_e

    ICC = (MSR-MSE)/(MSR+(k-1)*MSE+k*(MSC-MSE)/n)
    
    return(ICC)

def permutation_test(clf,X_all, y_all,real_auc,name,fpath_out = '',n_split=5,run = 1000):
    aucs = []
    kf = StratifiedKFold(n_splits=n_split, shuffle=True)
    
    for num in range(run):
        y_random = DataFrame(random.permutation(y_all),index = y_all.index,columns=y_all.columns)
        model_pred_proba = DataFrame()
        for train, test in kf.split(X_all,y_random): 
            X_train = X_all.iloc[train,:]
            y_train = y_all.iloc[train,:]
            X_test = X_all.iloc[test,:]
            y_test = y_all.iloc[test,:]
            clf.fit(X_train,y_train.label)
            model_pred_proba = concat([model_pred_proba,DataFrame(clf.predict_proba(X_test),index = X_test.index)])
        roc_auc = roc_auc_score(y_random, model_pred_proba.iloc[:,1])
        aucs.append(roc_auc)
    
    pvalue = sum(aucs > real_auc)/run
    fig, ax = plt.subplots(figsize=(9,6))
    ax.hist(aucs,edgecolor="black")
    if pvalue != 0:
        ax.axvline(real_auc, linestyle='--', color='k',label='p = %0.3f' %pvalue)
    else:
        ax.axvline(real_auc, linestyle='--', color='k',label='p < 0.001')
    ax.text(real_auc-0.02, ax.get_ylim()[1]+2.2, round(real_auc,3))
    ax.legend(loc="upper left")

    ax.set_xlabel('AUC')
    ax.set_ylabel('Count')
    ax.set_title('Permutation Test')
    fig.savefig(os.path.join(fpath_out,'_'.join([name,'Permutation.png'])))
    
    return(pvalue)

