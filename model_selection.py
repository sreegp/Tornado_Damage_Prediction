#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sklearn
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, EditedNearestNeighbours
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

class models_and_hyperparameter_tuning:
    def Logistic_Regression(x_train, y_train, C_values, penalty):
        results = []
        skf = StratifiedKFold(n_splits= 5, random_state=0)
        for train_index, test_index in tqdm(skf.split(x_train, y_train)):
            X_cv_train, X_cv_validate = pd.DataFrame(x_train).iloc[train_index], pd.DataFrame(x_train).iloc[test_index]
            Y_cv_train, Y_cv_validate = pd.DataFrame(y_train).iloc[train_index], pd.DataFrame(y_train).iloc[test_index]
            X_cv_train_balanced, Y_cv_train_balanced = RandomOverSampler().fit_resample(X_cv_train, Y_cv_train.values.ravel())
            for i in C_values:
                for j in penalty:
                    dt = LogisticRegression(penalty = j, C = i, solver = 'liblinear').fit(X_cv_train_balanced, np.ravel(Y_cv_train_balanced))
                    y_pred = dt.predict(X_cv_validate)
                    fpr, tpr, thresholds = metrics.roc_curve(Y_cv_validate, y_pred, pos_label = 1)
                    row = [i ,j, metrics.auc(fpr, tpr)]
                    results.append(row)
        # Summarizing results to get the best hyperparameters
        results = pd.DataFrame(results)
        results = results.rename(columns={ 0: 'C_values', 1: 'Penalty', 2: 'AUC'})
        mean_AUC = results.groupby(['C_values','Penalty']).mean().reset_index()
        standard_deviation_AUC = results.groupby(['C_values','Penalty']).std().reset_index()
        mean_AUC = mean_AUC.rename(columns={'AUC': 'Mean_AUC'})
        mean_AUC['Standard_Deviation_AUC'] = standard_deviation_AUC['AUC']
        AUC = mean_AUC
        return AUC[AUC['Mean_AUC'] == AUC['Mean_AUC'].max()]
    
    def Decision_Trees(x_train, y_train, min_samples_split_values, maxdepth):
        results = []
        skf = StratifiedKFold(n_splits= 5, random_state=0)
        for train_index, test_index in tqdm(skf.split(x_train, y_train)):
            X_cv_train, X_cv_validate = pd.DataFrame(x_train).iloc[train_index], pd.DataFrame(x_train).iloc[test_index]
            Y_cv_train, Y_cv_validate = pd.DataFrame(y_train).iloc[train_index], pd.DataFrame(y_train).iloc[test_index]
            X_cv_train_balanced, Y_cv_train_balanced = RandomOverSampler().fit_resample(X_cv_train, Y_cv_train.values.ravel())
            for i in min_samples_split_values:
                for j in maxdepth:
                    dt = DecisionTreeClassifier(min_samples_split = i, max_depth = j).fit(X_cv_train_balanced, np.ravel(Y_cv_train_balanced))
                    y_pred = dt.predict(X_cv_validate)
                    fpr, tpr, thresholds = metrics.roc_curve(Y_cv_validate, y_pred, pos_label = 1)
                    row = [i ,j, metrics.auc(fpr, tpr)]
                    results.append(row) 
        # Summarizing results to get the best hyperparameters
        results = pd.DataFrame(results)
        results = results.rename(columns={ 0: 'Min_Samples_Split', 1: 'Max_Depth', 2: 'AUC'})
        mean_AUC = results.groupby(['Min_Samples_Split','Max_Depth']).mean().reset_index()
        standard_deviation_AUC = results.groupby(['Min_Samples_Split','Max_Depth']).std().reset_index()
        mean_AUC = mean_AUC.rename(columns={'AUC': 'Mean_AUC'})
        mean_AUC['Standard_Deviation_AUC'] = standard_deviation_AUC['AUC']
        AUC = mean_AUC
        return AUC[AUC['Mean_AUC'] == AUC['Mean_AUC'].max()]
    
    def Random_Forest(x_train, y_train, estimators, max_depth, min_split):
        results = []
        skf = StratifiedKFold(n_splits=3, random_state=0)
        for train_index, test_index in tqdm(skf.split(x_train, y_train)):
            X_cv_train, X_cv_validate = pd.DataFrame(x_train).iloc[train_index], pd.DataFrame(x_train).iloc[test_index]
            Y_cv_train, Y_cv_validate = pd.DataFrame(y_train).iloc[train_index], pd.DataFrame(y_train).iloc[test_index]
            X_cv_train_balanced, Y_cv_train_balanced = RandomOverSampler().fit_resample(X_cv_train, Y_cv_train.values.ravel())
            for i in estimators:
                for j in max_depth:
                    for k in min_split:
                        dt = RandomForestClassifier(n_estimators = i, max_depth = j, min_samples_split = k).fit(X_cv_train_balanced, np.ravel(Y_cv_train_balanced))
                        y_pred = dt.predict(X_cv_validate)
                        fpr, tpr, thresholds = metrics.roc_curve(Y_cv_validate, y_pred, pos_label = 1)
                        row = [i ,j, k, metrics.auc(fpr, tpr)]
                        results.append(row) 
        # Summarizing results to get the best hyperparameters
        results = pd.DataFrame(results)
        results = results.rename(columns={ 0: 'No_of_Estimators', 1: 'Max_Depth', 2: 'Min_Split', 3: 'AUC'})
        mean_AUC = results.groupby(['No_of_Estimators','Max_Depth','Min_Split']).mean().reset_index()
        standard_deviation_AUC = results.groupby(['No_of_Estimators','Max_Depth', 'Min_Split']).std().reset_index()
        mean_AUC = mean_AUC.rename(columns={'AUC': 'Mean_AUC'})
        mean_AUC['Standard_Deviation_AUC'] = standard_deviation_AUC['AUC']
        AUC = mean_AUC
        return AUC[AUC['Mean_AUC'] == AUC['Mean_AUC'].max()]
    
