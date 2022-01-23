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
from AdaBoost import Adaboost_Algorithm
from sklearn.model_selection import train_test_split


def get_dataset(csv_signal,csv_bkg):
    """This function transform the dataset contained in the csv file in numpy array for the training"""
    columns=['B_M','P1_PT','P2_PT','m_corr','nSPDHits']

    #read the file cvs
    dataset_signal = read_csv(csv_signal, names=columns)
    dataset_bkg = read_csv(csv_bkg, names=columns)
    
    print('Size of dataset of signal in the csv: {}'.format(dataset_signal.shape))
    print('Size of dataset of background in the csv: {}'.format(dataset_bkg.shape))

    #transform the dataset in numpy array
    data_frame_signal = pd.DataFrame(dataset_signal)
    data_frame_bkg = pd.DataFrame(dataset_bkg)
    array_signal = np.array(data_frame_signal.values)
    array_bkg = np.array(data_frame_bkg.values)

    #remove the first row with the name of the features
    array_signal_training = array_signal[1:,:]
    array_bkg_training = array_bkg[1:,:]
    
    print('Size of numpy array of signal: {}'.format(array_signal_training.shape))
    print(array_signal_training)
    print('\n''\n')
    print('Size of numpy array of background:{} '.format(array_bkg_training.shape))
    print(array_bkg_training)

    #Create the columns with the labels of signal =1 and background=-1
    column_of_label_signal = np.ones(array_signal_training.shape[0])
    column_of_label_bkg = np.empty(array_bkg_training.shape[0],dtype = int)
    column_of_label_bkg.fill(-1)
    
    print('Size of new column with signal labels: {}'.format(column_of_label_signal.size))
    print(column_of_label_signal)
    print('Size of new column with background labels: {}'.format(column_of_label_bkg.size))
    print(column_of_label_bkg)
    

    #merge the array array with the signal and the array with the bkg for both labels and features
    array_y = np.concatenate((column_of_label_signal,column_of_label_bkg), axis=0)
    array_X = np.concatenate((array_signal_training,array_bkg_training), axis=0)
    print('Final array of the signal: \n')
    print(array_y)
    print('Final array of the background: \n')
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






