#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:42:08 2022

@author: lucatoscano
"""
#Import Libraries
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from get_features import get_features
from AdaBoost import Adaboost_Algorithm


def get_dataset(csv_signal,csv_bkg):
    """This function transform the dataset contained in the csv file in numpy array for the training"""
    
    columns=['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_M','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P1_PT','P2_IPCHI2_OWNPV','P2_PT','m_corr','nSPDHits']

    #Read the file cvs
    dataset_signal = read_csv(csv_signal, names=columns)
    dataset_bkg = read_csv(csv_bkg, names=columns)
    
    print('Size of dataset of signal in the csv: {}'.format(dataset_signal.shape))
    print('Size of dataset of background in the csv: {}'.format(dataset_bkg.shape))
    
    
    data_frame_signal, data_frame_bkg = get_features(dataset_signal, dataset_bkg)
    

    #Transform the dataset in numpy array
    #data_frame_signal = pd.DataFrame(dataset_signal)
    #data_frame_bkg = pd.DataFrame(dataset_bkg)
    array_signal = np.array(data_frame_signal.values)
    array_bkg = np.array(data_frame_bkg.values)


    
    print('Size of numpy array of signal: {}'.format(array_signal.shape))
    print(array_signal)
    print('\n''\n')
    print('Size of numpy array of background:{} '.format(array_bkg.shape))
    print(array_bkg)

    #Create the columns with the labels of signal =1 and background=-1
    column_of_label_signal = np.ones(array_signal.shape[0])
    column_of_label_bkg = np.empty(array_bkg.shape[0],dtype = int)
    column_of_label_bkg.fill(-1)
    
    print('Size of new column with signal labels: {}'.format(column_of_label_signal.size))
    print(column_of_label_signal)
    print('Size of new column with background labels: {}'.format(column_of_label_bkg.size))
    print(column_of_label_bkg)
    

    #Merge the array with the signal and the array with the bkg for both labels and features
    array_y = np.concatenate((column_of_label_signal,column_of_label_bkg), axis=0)
    array_X = np.concatenate((array_signal,array_bkg), axis=0)
    print('Final array of the labels: \n')
    print(array_y)
    print('Final array of the features: \n')
    print(array_X)

    #split the array in test sample and training sample
    X_train, X_validation, y_train, y_validation = train_test_split(array_X, array_y, test_size=0.20, random_state=1)

    return X_train, X_validation, y_train, y_validation


def accurancy (label_true, label_predicted):
    """This function which gives the accurancy of the algorithm"""
    
    accurancy = np.sum(label_true == label_predicted) / len(label_true) 
    return accurancy


def training (X_train, X_validation, y_train,  y_validation):
    """This function performes the training of the algorithm"""
    
    classification = Adaboost_Algorithm(n_of_classifier = 5)
    classification.fit(X_train, y_train)
    acc=accurancy(y_validation,y_train)
    print('the accurancy of the algorithm is : ',acc)






