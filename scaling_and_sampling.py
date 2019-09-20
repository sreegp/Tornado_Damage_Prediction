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

class scaling_train_test_split:
    def train_and_test_split(data):
        # Sort data in ascending order
        data = data.sort_values(by=['BEGIN_DATE_TIME'])
        total_length = len(data)
        training_length = int(total_length*0.8)
        training_data = data.iloc[:training_length]
        validation_length = int(total_length*0.15)
        validation_data = data.iloc[training_length:training_length+validation_length]
        test_data = data.iloc[validation_length:]
        return training_data, validation_data, test_data
    
    def scaling(training_data, validation_data, test_data):
        training_data_X = training_data.drop(columns=['binary_casualties', 'BEGIN_DATE_TIME'], axis = 1)
        scaled_training_data_X = preprocessing.scale(training_data_X)
        validation_data_X = validation_data.drop(columns=['binary_casualties', 'BEGIN_DATE_TIME'], axis = 1)
        scaled_validation_data_X = preprocessing.scale(validation_data_X)
        test_data_X = test_data.drop(columns=['binary_casualties', 'BEGIN_DATE_TIME'], axis = 1)
        scaled_test_data_X = preprocessing.scale(test_data_X)
        training_data_Y = training_data['binary_casualties']
        validation_data_Y = validation_data['binary_casualties']
        test_data_Y = test_data['binary_casualties']
        return scaled_training_data_X, training_data_Y, scaled_validation_data_X, validation_data_Y, scaled_test_data_X, test_data_Y
    
class sampling_and_baseline_model:
    def baseline_log_reg_model(x_train, y_train, x_val, y_val):
        clf = LogisticRegression(solver = 'liblinear').fit(x_train, y_train)
        pred = clf.predict(x_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred, pos_label = 1)
        print(metrics.auc(fpr, tpr))
        print(confusion_matrix(y_val, pred))
    def random_oversampling(x_train, y_train, x_val, y_val):
        x_train_resampled_rand, y_train_resampled_rand = RandomOverSampler().fit_resample(x_train, y_train)
        clf = LogisticRegression(solver = 'liblinear').fit(x_train_resampled_rand, y_train_resampled_rand)
        pred = clf.predict(x_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred, pos_label = 1)
        print(metrics.auc(fpr, tpr))
        print(confusion_matrix(y_val, pred))
    def SMOTE_oversampling(x_train, y_train, x_val, y_val):
        x_train_resampled_SMOTE, y_train_resampled_SMOTE = SMOTE().fit_resample(x_train, y_train)
        clf = LogisticRegression(solver = 'liblinear').fit(x_train_resampled_SMOTE, y_train_resampled_SMOTE)
        pred = clf.predict(x_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred, pos_label = 1)
        print(metrics.auc(fpr, tpr))
        print(confusion_matrix(y_val, pred))
    def ADASYN_oversampling(x_train, y_train, x_val, y_val):
        x_train_resampled_ADASYN, y_train_resampled_ADASYN = ADASYN().fit_resample(x_train, y_train)
        clf = LogisticRegression(solver = 'liblinear').fit(x_train_resampled_ADASYN, y_train_resampled_ADASYN)
        pred = clf.predict(x_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred, pos_label = 1)
        print(metrics.auc(fpr, tpr))
        print(confusion_matrix(y_val, pred))
    def random_undersampling(x_train, y_train, x_val, y_val):
        x_train_unsampled_rand, y_train_unsampled_rand = RandomUnderSampler().fit_resample(x_train, y_train)
        clf = LogisticRegression(solver = 'liblinear').fit(x_train_unsampled_rand, y_train_unsampled_rand)
        pred = clf.predict(x_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred, pos_label = 1)
        print(metrics.auc(fpr, tpr))
        print(confusion_matrix(y_val, pred))
    def Edited_NN_undersampling(x_train, y_train, x_val, y_val):
        x_train_unsampled_ENN, y_train_unsampled_ENN = EditedNearestNeighbours().fit_resample(x_train, y_train)
        clf = LogisticRegression(solver = 'liblinear').fit(x_train_unsampled_ENN, y_train_unsampled_ENN)
        pred = clf.predict(x_val)
        fpr, tpr, thresholds = metrics.roc_curve(y_val, pred, pos_label = 1)
        print(metrics.auc(fpr, tpr))
        print(confusion_matrix(y_val, pred))