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

columns=['B_M','P1_PT','P2_PT','m_corr','nSPDHits']

file_signal='Kpi_2018_signal.csv'
file_bkg='pipi_2018_bkg.csv'
file_test='pipi_2018_test.csv'

dataset_signal = read_csv(file_signal, names=columns)
dataset_bkg = read_csv(file_bkg, names=columns)
dataset_test = read_csv(file_test, names=columns)



# Explore the dataset
def scan_dataset(dataset, column = 'B_M'):
    """Show the content of the dataset given as argoument and the size of one column."""
    # shape
    print('Size of data: {}'.format(dataset.shape))
    print('Number of events: {}'.format(dataset.shape[0]))
    print('Number of columns: {}'.format(dataset.shape[1]))
    # show the first n rows of the dataset
    #print(dataset.head(5))
    # column distribution distribution
    print(dataset.groupby(column).size())
    
    print ('\nList of features in dataset:')
    for col in dataset.columns:
       print(col)


#transform the dataset in numpy array
data_frame_signal=pd.DataFrame(dataset_signal)
data_frame_bkg=pd.DataFrame(dataset_bkg)
array_signal =np.array(data_frame_signal.values)
array_bkg = np.array(data_frame_bkg.values)


#remove the first row with the name of the features
array_signal_training=array_signal[1:,:]
array_bkg_training=array_bkg[1:,:]
print('size of numpy array of signal : ',array_signal_training.shape)
print(array_signal_training)
print('\n''\n')
print('size of numpy array of background : ',array_bkg_training.shape)
print(array_bkg_training)


#Create the columns with the labels of signal =1 and background=-1
column_of_label_signal=np.ones(array_signal_training.shape[0])
print('size of new column with signal labels: ',column_of_label_signal.size)
print(column_of_label_signal)
column_of_label_bkg=np.empty(array_bkg_training.shape[0],dtype = int)
column_of_label_bkg.fill(-1)
print('size of new column with background labels: ',column_of_label_bkg.size)
print(column_of_label_bkg)



#merge the array array with the signal and the array with the bkg for both labels and features
array_y = np.concatenate((column_of_label_signal,column_of_label_bkg), axis=0)
array_X = np.concatenate((array_signal_training,array_bkg_training), axis=0)
print(array_y)
print(array_X)

#split the array in test sample and training sample
X_train, X_validation, y_train, y_validation = train_test_split(array_X, array_y, test_size=0.20, random_state=1)

#Function which gives the accurancy of the algorithm
def accurancy (label_true, label_predicted):
    """Function which gives the accurancy of the algorithm"""
    accurancy = np.sum(label_true == label_predicted) / len(label_true) 
    return accurancy


classification = Adaboost_Algorithm(n_of_classifier = 5)
classification.fit(X_train, y_train)
acc=accurancy(y_validation,y_train)
print('the accurancy of the algorithm is : ',acc)






