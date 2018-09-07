# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:13:24 2018

# UPDATE IN 20180820

@author: Caizhengting
"""

import os
import math
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from scipy.stats import f
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,brier_score_loss

def Find_Optimal_Cutoff(target, predicted, pattern = 0):
    # UPDATE IN 20180810
    # INPUTTED THE DATA STRUCTURE-series
    
    fpr, tpr, threshold = roc_curve(target, predicted, drop_intermediate = False)
    i = arange(len(tpr)) 
    roc = DataFrame({'tf' : Series(tpr-(1-fpr), index=i), 'Y' : Series(tpr-fpr), 
                     'threshold' : Series(threshold)})
    if pattern == 0:
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    else:
        roc_t = roc.iloc[roc.Y.argsort()[len(tpr)-1:len(tpr)]]
    return list(roc_t['threshold'])


def AIC(y_test, y_pred, k, n):
    # Model error obeys independent normal distribution
    if type(y_pred) == type(DataFrame()):
        y_pred = y_pred.iloc[:,0]
    if type(y_test) == type(DataFrame()):
        y_test = y_test.iloc[:,0]
    # change into the series
    
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*math.log(float(SSR)/n)
    return(AICValue)
    
    
def CI_ROC(y_test, y_pred, pattern = 0):
    if type(y_pred) == type(DataFrame()):
        y_pred = y_pred.iloc[:,0]
    if type(y_test) == type(DataFrame()):
        y_test = y_test.iloc[:,0]
    # change into the series
    V10 = list()
    V01 = list()
    
    num0 = array(y_pred[y_test == 0]);n = len(num0)
    num1 = array(y_pred[y_test == 1]);m = len(num1)
    N = m+n
    
    for i in range(m):
        V10.append((sum(num0 < num1[i]) + sum(num0 == num1[i])/2)/n)
    for j in range(n):
        V01.append((sum(num1 > num0[j]) + sum(num1 == num0[j])/2)/m)
    A = sum(V01)/n
    X = int(A*N)
    
    S10 = (sum(multiply(V10,V10))-m*A*A)/(m-1)
    S01 = (sum(multiply(V01,V01))-n*A*A)/(n-1)
    SE = (S10/m+ S01/n) ** 0.5
    
    if pattern == 0:
        # Wald method
        CI_lower = A - SE * 1.96
        CI_upper = A + SE * 1.96
        method = 'Wald method'
        
    elif pattern == 1:
        # exact binomial method(seems little bug)
        C1 = (N-X+1)/X
        F1 = f.ppf(0.025,2*(N-X+1),2*X)
        CI_lower = 1/(1+C1/F1)
        C2 = (N-X)/(X+1)
        F2 = f.ppf(0.975,2*(N-X),2*(X+1))
        CI_upper = 1/(1+C2/F2)
        method = 'exact binomial method'
        
    elif pattern == 2:
        # modified Wald method
        P = (X+2)/(N+4)
        W = 2*(P*(1-P)/(N+4)) ** 0.5
        CI_lower = A - W
        CI_upper = A + W
        method = 'modified Wald method'
        
    elif pattern == 3:
        # bootstrap
        n_bootstraps = 1000
        bootstrapped_scores = []
        rng = random.RandomState(0)
        for i in range(n_bootstraps):
            indice0 = rng.randint(0, n, n)
            indice1 = rng.randint(0, m, m)
            prob = append(num0[indice0],num1[indice1])
            true = append(repeat(0,n),repeat(1,m))
            s = roc_auc_score(true, prob)
            bootstrapped_scores.append(s)
        SE = (sum((bootstrapped_scores-mean(bootstrapped_scores)) ** 2)/(n_bootstraps-1)) ** 0.5
        CI_lower = A - SE * 1.96
        CI_upper = A + SE * 1.96
        method = 'bootstrap method'
    return(method,CI_lower,CI_upper)
    
#--------------------------------------------   
    
def final_model_oneCV_pred(model, X_train_group, y_train_group, X_test_group, y_test_group):
    # cross validation of prediction

    X_all = concat([X_test_group[0],X_train_group[0]]).sort_index()
    y_all = concat([y_test_group[0],y_train_group[0]]).sort_index()
    
    
    model_pred_proba = DataFrame()
    model_pred = DataFrame()
    model_pred_proba_group = list()
    model_pred_group = list()
    

    for cv in range(len(X_train_group)):
        X_train = X_train_group[cv]
        y_train = y_train_group[cv]
        X_test = X_test_group[cv]
        y_test = y_test_group[cv]
        model.fit(X_train, y_train)
        model_params = model.get_params()

        pred_tmp = DataFrame(model.predict(X_test),index = X_test.index,columns=['label'])
        pred_proba_tmp = DataFrame(model.predict_proba(X_test),index = X_test.index)
        cutoff = Find_Optimal_Cutoff(y_test, pred_proba_tmp.iloc[:,1], pattern=1)
        pred_tmp[pred_proba_tmp.iloc[:,1] >= cutoff] = 1
        pred_tmp[pred_proba_tmp.iloc[:,1] < cutoff] = 0
        
        model_pred_group.append(pred_tmp)
        model_pred_proba_group.append(pred_proba_tmp)
        model_pred = concat([model_pred,pred_tmp])
        model_pred_proba = concat([model_pred_proba,pred_proba_tmp])
    
    model_pred = model_pred.sort_index()
    model_pred_proba = model_pred_proba.sort_index()
    cutoff = Find_Optimal_Cutoff(y_all, model_pred_proba.iloc[:,1], pattern=1)
    model_pred[model_pred_proba.iloc[:,1] >= cutoff] = 1
    model_pred[model_pred_proba.iloc[:,1] < cutoff] = 0
    
    return(model_params,cutoff,model_pred_group,model_pred_proba_group,model_pred,model_pred_proba)
    
def final_model_aic(model, X_all, y_all):
    
    model_pred_proba = DataFrame()
    model_pred = DataFrame()
    
    tprs = []
    aucs = []
    mean_fpr = linspace(0, 1, 100)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True) 
    # Stratified k-fold
    
    for train, test in kf.split(y_all,y_all.label): 
        X_train = X_all.iloc[train,:]
        y_train = y_all.iloc[train,:].label
        X_test = X_all.iloc[test,:]
        y_test = y_all.iloc[test,:].label
        model.fit(X_train, y_train)
        
        pred_tmp = DataFrame(model.predict(X_test),index = X_test.index)
        pred_proba_tmp = DataFrame(model.predict_proba(X_test),index = X_test.index)
        model_pred = concat([model_pred,pred_tmp])
        model_pred_proba = concat([model_pred_proba,pred_proba_tmp]) 
        
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba_tmp.iloc[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    
    mean_tpr = mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    model_pred = model_pred.sort_index()
    model_pred_proba = model_pred_proba.sort_index()
    
    fpr, tpr, _ = roc_curve(y_all, model_pred_proba.iloc[:,1])
    roc_auc = auc(fpr, tpr)
    
    num_smp,num_fea = X_all.shape
    aic = AIC(y_all, model_pred, num_fea, num_smp)
    
    return(roc_auc,aic,mean_auc)
    

def final_model_oneCV_FoldIndex(name, y_test_group, model_pred_group, model_pred_proba_group, fpath_out):
    # ROC and index for mean of folds

    sens = []
    pres = []
    spcs = []   
    accs = []
    Fs   = []
    Brs  = []  
    
    tprs = []
    aucs = []
    fig, ax = plt.subplots(figsize=(9,6))
    mean_fpr = linspace(0, 1, 100)
    i = 0
    
    for cv in range(len(y_test_group)):
        y_test = y_test_group[cv]
        pred_tmp = model_pred_group[cv]
        pred_proba_tmp = model_pred_proba_group[cv]
              
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, pred_proba_tmp.iloc[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        i += 1
        
        TN, FP, FN, TP = confusion_matrix(y_test, pred_tmp).ravel()
        sens.append(recall_score(y_test, pred_tmp)) # sensitivity = recall
        pres.append(precision_score(y_test, pred_tmp))
        spcs.append(TN / (TN + FP))
        accs.append(accuracy_score(y_test, pred_tmp))
        Fs.append(f1_score(y_test, pred_tmp))
        Brs.append(brier_score_loss(y_test, pred_proba_tmp.iloc[:,1], pos_label=y_test.max()))
                
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
    
    mean_tpr = mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr);std_auc = std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    
    std_tpr = std(tprs, axis=0)
    tprs_upper = minimum(mean_tpr + std_tpr, 1)
    tprs_lower = maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic(ROC)')
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(fpath_out,'_'.join([name,'ROC-CVMean.png'])))
    
    #------------------------------------------------------------
        
    mean_sens = mean(sens);std_sens = std(sens)
    mean_pres = mean(pres);std_pres = std(pres)
    mean_spcs = mean(spcs);std_spcs = std(spcs)
    mean_accs = mean(accs);std_accs = std(accs)
    mean_Fs = mean(Fs);std_Fs = std(Fs)
    mean_Brs = mean(Brs);std_Brs = std(Brs)

    print('------------------%s_Fold------------------' %name)
    print('mean sensitivity/recall = %.3f ± %0.3f ' %(mean_sens,std_sens))
    print('mean precision = %.3f ± %0.3f' %(mean_pres,std_pres))
    print('mean specificity = %.3f ± %0.3f ' %(mean_spcs,std_spcs))
    print('mean accuracy = %.3f ± %0.3f' %(mean_accs,std_accs))
    print('mean F1 = %.3f ± %0.3f' %(mean_Fs,std_Fs))
    print('mean Brier = %.3f ± %0.3f' %(mean_Brs,std_Brs))
    print('mean AUC = %.3f ± %0.3f' %(mean_auc,std_auc))

    f_name ='_'.join([name,'statistics.txt'])
#    if os.path.exists(os.path.join(fpath_out,f_name)):
#        f = open(os.path.join(fpath_out,f_name), 'a')
#    else:
    f = open(os.path.join(fpath_out,f_name), 'w')
    f.write('------------------%s_Fold------------------' %name)
    f.write('\nmean sensitivity/recall = %.3f ± %0.3f' %(mean_sens,std_sens))
    f.write('\nmean precision = %.3f ± %0.3f' %(mean_pres,std_pres))
    f.write('\nmean specificity = %.3f ± %0.3f' %(mean_spcs,std_spcs))
    f.write('\nmean accuracy = %.3f ± %0.3f' %(mean_accs,std_accs))
    f.write('\nmean F1 = %.3f ± %0.3f' %(mean_Fs,std_Fs))
    f.write('\nmean Brier = %.3f ± %0.3f' %(mean_Brs,std_Brs))
    f.write('\nmean AUC = %.3f ± %0.3f' %(mean_auc,std_auc))
    f.write('\n\n')
    f.close()
    
    return(mean_sens,mean_pres,mean_spcs,mean_accs,mean_Fs,mean_Brs,mean_auc,
           std_sens,std_pres,std_spcs,std_accs,std_Fs,std_Brs,std_auc)
 
    
def final_model_oneCV_AllIndex(name,label, y_all, model_pred, model_pred_proba, fpath_out, CI_pattern):
    # ROC and index for all samples of cross validation
    n_classes = len(label)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_all, model_pred_proba.iloc[:,i], pos_label = i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        tpr[i][0] = 0.0

    fig, ax = plt.subplots(figsize=(9,6))
    lw = 2
    ax.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve for {0}(AUC = {1:.3f})'.format(label[0],roc_auc[0]))
    ax.plot(fpr[1], tpr[1], color='cornflowerblue', lw=lw, label='ROC curve for {0}(AUC = {1:.3f})'.format(label[1],roc_auc[1]))
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.legend(loc="lower right")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic(ROC)')
    fig.savefig(os.path.join(fpath_out,'_'.join([name,'ROC.png'])))
    
    #------------------------------------------------

    model_report = classification_report(y_all, model_pred,target_names=label)
    model_matrix = confusion_matrix(y_all, model_pred)
    TN, FP, FN, TP = model_matrix.ravel()
    recall = recall_score(y_all, model_pred) # sensitivity = recall
    precision = precision_score(y_all, model_pred)
    specificity = TN / (TN + FP)
    accuracy = accuracy_score(y_all, model_pred)
    F1 = f1_score(y_all, model_pred)
    Brier = brier_score_loss(y_all, model_pred_proba.iloc[:,1], pos_label=y_all.max())
    method,CI_lower,CI_upper = CI_ROC(y_all, model_pred_proba.iloc[:,1], CI_pattern)
    
    print('------------------%s_All------------------' %name)
    print(model_report)
    print(model_matrix)
    print('sensitivity/recall = %.3f ' %recall)
    print('precision = %.3f' %precision)
    print('specificity = %.3f ' %specificity)
    print('accuracy = %.3f' %accuracy)
    print('F1 = %.3f' %F1)
    print('Brier = %.3f' %Brier)
    print('AUC = {:0.3f} (95%CI:{:0.3f}-{:0.3f}) by {:s}'.format(roc_auc[0],CI_lower,CI_upper,method))

    f_name ='_'.join([name,'statistics.txt'])
    f = open(os.path.join(fpath_out,f_name), 'a')
    f.write('\n------------------%s_All------------------\n' %name)
    f.write(model_report)
    f.write('\n[Confusion Matrix]\n')
    f.write('\t\tPredict_0\tPredict_1\nReal_0    {0}\t      {1}\nReal_1    {2}\t     {3}\n'.format(TN, FP, FN, TP))
    f.write('sensitivity/recall = %.3f ' %recall)
    f.write('\nprecision = %.3f' %precision)
    f.write('\nspecificity = %0.3f' %specificity)
    f.write('\naccuracy = %.3f' %accuracy)
    f.write('\nF1 = %.3f' %F1)
    f.write('\nBrier = %.3f' %Brier)    
    f.write('\nAUC = {:0.3f} (95%CI:{:0.3f}-{:0.3f}) by {:s}'.format(roc_auc[0],CI_lower,CI_upper,method))
    
    return(roc_auc[0],recall,precision,specificity,accuracy,F1,Brier)


def final_model_oneCV_All(model,name,label, X_train_group, y_train_group, X_test_group, y_test_group,fpath_out = '',CI_pattern = 0):
    # pipelien of cross validation
    
    X_all = concat([X_test_group[0],X_train_group[0]]).sort_index()
    y_all = concat([y_test_group[0],y_train_group[0]]).sort_index()
    
    model_params,cutoff,model_pred_group,model_pred_proba_group,model_pred,model_pred_proba = final_model_oneCV_pred(
            model, X_train_group, y_train_group, X_test_group, y_test_group)
    
    model_pred.to_csv(os.path.join(fpath_out,'_'.join([name,'pred.csv'])))
    model_pred_proba.to_csv(os.path.join(fpath_out,'_'.join([name,'pred_proba.csv'])))
    
    FoldIndex = final_model_oneCV_FoldIndex(name, y_test_group, model_pred_group, model_pred_proba_group, fpath_out)
    AllIndex = final_model_oneCV_AllIndex(name,label, y_all, model_pred, model_pred_proba, fpath_out, CI_pattern)
    
    
    print('cutoff of ROC = %.3f(Youden)\n' %cutoff[0])
    f_name ='_'.join([name,'statistics.txt'])
    f = open(os.path.join(fpath_out,f_name), 'a')
    f.write('\ncutoff of ROC = %.3f(Youden)' %cutoff[0])
    f.write('\n\n')
    
    f.write('\nthe Params of %s\n' %name)
    for key in model_params:
        f.write('\t{0}:{1}\n'.format(key,model_params[key]))
    f.write('\n\n\n\n')
    f.close()
    
    return(FoldIndex,AllIndex)

    
def final_model_groupCV_All(model,name,X_all,y_all,fpath_out = '',run = 100,n_split=5,
                           ID_train_file = 'ID_Train_CV.xlsx', ID_test_file = 'ID_Test_CV.xlsx'):
    
    writer_pred = ExcelWriter(os.path.join(fpath_out,'_'.join([name,'pred_group.xlsx'])))
    writer_pred_proba = ExcelWriter(os.path.join(fpath_out,'_'.join([name,'pred_proba_group.xlsx'])))
    
    fig, ax = plt.subplots(figsize=(9,6))
    tprs = []
    aucs = []
    mean_fpr = linspace(0, 1, 100)    
    
    for r in range(run):

        X_train_group = []
        y_train_group = []
        X_test_group = []
        y_test_group = []

        ID_test = read_excel(ID_test_file,r,header=0,index_col=0)
        ID_train = read_excel(ID_train_file,r,header=0,index_col=0)

        for i in range(n_split):
            id_test = ID_test.iloc[:,i].dropna(axis = 0).apply(int)
            id_train = ID_train.iloc[:,i].dropna(axis = 0).apply(int)
            X_test_group.append(X_all.reindex(id_test))
            X_train_group.append(X_all.reindex(id_train))
            y_test_group.append(y_all.reindex(id_test).label)
            y_train_group.append(y_all.reindex(id_train).label)

        tmp1,tmp2,tmp3,tmp4,model_pred,model_pred_proba = final_model_oneCV_pred(model, X_train_group, y_train_group, X_test_group, y_test_group)
        model_pred.to_excel(writer_pred, str(r))
        model_pred_proba.to_excel(writer_pred_proba, str(r))
        
        fpr, tpr, thresholds = roc_curve(y_all, model_pred_proba.iloc[:,1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

    mean_tpr = mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)

    std_tpr = std(tprs, axis=0)
    tprs_upper = minimum(mean_tpr + std_tpr, 1)
    tprs_lower = maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic(ROC)')
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(fpath_out,'_'.join([name,'ROC-allMean.png'])))
    
    writer_pred.save()
    writer_pred_proba.save()
    
    return(mean_auc,std_auc)
