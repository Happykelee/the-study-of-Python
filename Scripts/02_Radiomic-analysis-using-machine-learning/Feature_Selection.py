# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:11:00 2018

@author: Caizhengting
"""
import os
import matplotlib
from numpy import *
from pandas import *
from itertools import *
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import Lasso,lasso_path,LassoCV


def LV(X,name,fpath_out = '',ths = 0.9):
    sel_LV = VarianceThreshold(threshold=(ths * (1 - ths)))
    sel_LV.fit_transform(X)
    sel_LVsupport = sel_LV.get_support() # T or F
    sel_FeaName_LV = []
    filter_all = []
    filter_sel = []
    for index, item in enumerate(X.columns):
        filter_all.append(item.split('_')[0])
        if sel_LVsupport[index]:
            sel_FeaName_LV.append(item)
            filter_sel.append(item.split('_')[0])
    
    #---------------------------------------

    temp = matplotlib.rcParams["figure.subplot.left"]
    matplotlib.rcParams["figure.subplot.left"] = temp * 1.2

    count_all = Counter(filter_all)
    count_sel = Counter(filter_sel)
    fig, ax = plt.subplots(figsize=(9,9))
    ax.barh(arange(len(count_all)),list(count_all.values()), 0.3, color ='b',label='the original features')
    ax.barh(arange(len(count_all))+0.3,list(count_sel.values()),0.3,color='cornflowerblue',label='the selected features')

    ax.set_xlabel('Number')
    ax.set_ylabel('Filters')
    ax.set_ylim([-1,len(count_all)+0.6])
    ax.set_title('Features selected by VarianceThreshold')
    ax.set_yticks(arange(len(count_all)))
    ax.set_yticklabels(count_all.keys())
    ax.legend(loc="upper right")

    for i in range(len(count_all)):
        value_all = list(count_all.values())[i]
        value_sel = list(count_sel.values())[i]
        ax.text(value_all+.5, i-.05, value_all)
        ax.text(value_sel+.5, i+.25, value_sel)
    fig.savefig(os.path.join(fpath_out,'_'.join([name,'X_LV.png'])))
    matplotlib.rcParams["figure.subplot.left"] = temp
    
    #----------------------------------------------

    print("Features reduced from {0} to {1} by VarianceThreshold".format(len(X.columns),len(sel_FeaName_LV)))
    X_LV = X[sel_FeaName_LV]
    try:
        X_LV.to_csv(os.path.join(fpath_out,'_'.join([name,'X_LV.csv'])))
    except PermissionError:
        X_LV.to_csv(os.path.join(fpath_out,'_'.join([name,'X_LV_01.csv'])))
        
    return(X_LV)

def KB(X,y,name,fpath_out = ''):
    
    sel_KB = SelectKBest(f_classif,k = 'all').fit(X,y)
    sel_KB_scores = array([sel_KB.scores_,sel_KB.pvalues_]).T
    sel_KB_scores = DataFrame(sel_KB_scores,index = X.columns,columns=['F-statistic','p'])
    sel_KB_scores = sel_KB_scores.sort_values('F-statistic',ascending= False)
    sel_KB_scores = sel_KB_scores[sel_KB_scores['p']< 0.05]
    sel_KB_scores.index = sel_KB_scores.index.rename('Features')

    #------------------------------------------------

    temp = matplotlib.rcParams["figure.subplot.left"]
    matplotlib.rcParams["figure.subplot.left"] = temp * 4.5

    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(arange(len(sel_KB_scores)),sel_KB_scores.iloc[:,0])

    ax.set_xlabel('F-value')
    ax.set_ylabel('Features')
    ax.set_title('F-statistic of features selected by ANOVA')
    ax.set_ylim([-1,len(sel_KB_scores)])
    ax.set_yticks(arange(len(sel_KB_scores)))
    ax.set_yticklabels(sel_KB_scores.index)

    fig.savefig(os.path.join(fpath_out,'_'.join([name,'X_KB.png'])))

    matplotlib.rcParams["figure.subplot.left"] = temp

    #------------------------------------------------

    print("Features reduced from {0} to {1} by SelectKBest".format(len(X.columns),len(sel_KB_scores.index)))
    X_KB = X[sel_KB_scores.index]

    try:
        X_KB.to_csv(os.path.join(fpath_out,'_'.join([name,'X_KB.csv'])))
    except PermissionError:
        X_KB.to_csv(os.path.join(fpath_out,'_'.join([name,'X_KB_01.csv'])))
    try:
        sel_KB_scores.to_csv(os.path.join(fpath_out,'_'.join([name,'X_KB_stats.csv'])))
    except PermissionError:
        sel_KB_scores.to_csv(os.path.join(fpath_out,'_'.join([name,'X_KB_stats_01.csv'])))
        
    #----------------------------------------------
    
    return(X_KB, sel_KB_scores)
	
def Lasso_MSEPath(X_train,y_train,name,fpath_out = ''):

    sel_Lasso = LassoCV(cv=10,max_iter=10000).fit(X_train,y_train)
    # trick
    # sel_Lasso.alpha_ = 10**-1.06
    sel_log_LassoAlpha =  -log10(sel_Lasso.alpha_)
    sel_log_LassoAlphas = -log10(sel_Lasso.alphas_)

    fig, ax = plt.subplots(figsize=(9,6))
    ax.plot(sel_log_LassoAlphas, sel_Lasso.mse_path_, ':')
    ax.plot(sel_log_LassoAlphas, sel_Lasso.mse_path_.mean(axis=-1), 'k',label='Average across the folds', linewidth=2)
    ax.axvline(sel_log_LassoAlpha, linestyle='--', color='k',label='alpha: CV estimate')
    ax.text(sel_log_LassoAlpha-0.1, ax.get_ylim()[1]+0.02, round(sel_log_LassoAlpha,2))
    ax.legend()

    ax.set_title('Mean Square Error Path')
    ax.set_xlabel('-log($\\alpha$)')
    ax.set_ylabel('Mean square error') 

    fig.savefig(os.path.join(fpath_out,'_'.join([name,'X_Lasso_MSEPath.png'])))
    print('Alpha:', sel_Lasso.alpha_)
    
    return(sel_log_LassoAlpha,sel_Lasso.alpha_)
	
def Lasso_LassoPath(X_train,y_train,sel_log_LassoAlpha,name,fpath_out = ''):
    eps = 1e-2 # the smaller it is the longer is the path
    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, eps, fit_intercept=False)
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_LassoAlphas = -log10(alphas_lasso)

    fig, ax = plt.subplots(figsize=(9,6))
    for coef_l, c in zip(coefs_lasso,colors):
        l = ax.plot(neg_log_LassoAlphas, coef_l, c=c)
    ax.axvline(sel_log_LassoAlpha, linestyle='--', color='k',label='alpha: CV estimate')
    ax.text(sel_log_LassoAlpha-0.05, ax.get_ylim()[1]+0.01, round(sel_log_LassoAlpha,2))

    ax.set_xlabel('-log($\\alpha$)')
	# or 
	# ax.set_xlabel(r'-log($\alpha$)')
    ax.set_ylabel('Coefficients')
    ax.set_title('Lasso Path')
    
    fig.savefig(os.path.join(fpath_out,'_'.join([name,'X_LassoPath.png'])))
	
def LASSO(X_train,y_train,sel_Alpha,name,fpath_out = ''):  
    the_Lasso = Lasso(sel_Alpha).fit(X_train,y_train)
    sel_FeaName_Lasso = X_train.columns[the_Lasso.coef_ != 0]
    X_Lasso = X_train[sel_FeaName_Lasso]
    X_predict = DataFrame(the_Lasso.predict(X_train),index = X_Lasso.index)
    sel_Lasso_Coefs = DataFrame(the_Lasso.coef_[the_Lasso.coef_ != 0], index = sel_FeaName_Lasso,columns = ['Coefficients'])
    sel_Lasso_Coefs = sel_Lasso_Coefs.sort_values(by = 'Coefficients')
    sel_Lasso_Coefs.index = sel_Lasso_Coefs.index.rename('Features')

    #---------------------------------
    temp = matplotlib.rcParams["figure.subplot.left"]
    matplotlib.rcParams["figure.subplot.left"] = temp * 2.3

    fig, ax = plt.subplots(figsize=(12,5))
    ax.barh(arange(len(sel_Lasso_Coefs)),sel_Lasso_Coefs.iloc[:,0])

    ax.set_xlabel('Coefficients')
    ax.set_ylabel('Features')
    ax.set_yticks(arange(len(sel_Lasso_Coefs)))
    ax.set_yticklabels(sel_Lasso_Coefs.index)
    ax.set_title("Coefficients in the Lasso Model") 

    fig.savefig(os.path.join(fpath_out,'_'.join([name,'X_Lasso_Coefs.png'])))

    matplotlib.rcParams["figure.subplot.left"] = temp
    
    #------------------------------------------
    sel_Lasso_Coefs = concat([sel_Lasso_Coefs,
                              DataFrame(the_Lasso.intercept_,index = ['intercept(non-feature)'],columns = ['Coefficients'])])
    print(sel_Lasso_Coefs)
    # print('intercept:',the_Lasso.intercept_)
    print("\nFeatures reduced from {0} to {1} by LASSO".format(shape(X_train)[1],len(sel_FeaName_Lasso)))
    
    X_predict.to_csv(os.path.join(fpath_out,'_'.join([name,'X_Lasso_Predict.csv'])))
    try:
        X_Lasso.to_csv(os.path.join(fpath_out,'_'.join([name,'X_Lasso.csv'])))
    except PermissionError:
        X_Lasso.to_csv(os.path.join(fpath_out,'_'.join([name,'X_Lasso_01.csv'])))
    try:
        sel_Lasso_Coefs.to_csv(os.path.join(fpath_out,'_'.join([name,'X_LassoCoefs.csv'])))
    except PermissionError:
        sel_Lasso_Coefs.to_csv(os.path.join(fpath_out,'_'.join([name,'X_LassoCoefs_01.csv'])))
    
    return(X_Lasso,sel_Lasso_Coefs)