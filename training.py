#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 23:33:26 2022

@author: lucatoscano
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:42:08 2022

@author: lucatoscano
"""
#Import Libraries
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from get_features import get_features
from AdaBoost import Adaboost_Algorithm
from sklearn import datasets
import pickle

def get_dataset(csv_signal,csv_bkg):
    """This function transform the dataset contained in the csv file in numpy array for the training"""
    
    columns=['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_M','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P1_PT','P2_IPCHI2_OWNPV','P2_PT','m_corr','nSPDHits']

    #Read the file cvs
    dataset_signal = read_csv(csv_signal, names=columns)
    dataset_bkg = read_csv(csv_bkg, names=columns)
    
    print('Size of dataset of signal in the csv: {}'.format(dataset_signal.shape))
    print('Size of dataset of background in the csv: {}'.format(dataset_bkg.shape))
    
    #Call the get_features function for compute the columns of the features from the datasets
    data_frame_signal, data_frame_bkg = get_features(dataset_signal, dataset_bkg)
    

    #Transform the dataset in numpy array
    #data_frame_signal = pd.DataFrame(dataset_signal)
    #data_frame_bkg = pd.DataFrame(dataset_bkg)
    array_signal = np.array(data_frame_signal.values)
    array_bkg = np.array(data_frame_bkg.values)


    
    #print('Size of numpy array of signal: {}'.format(array_signal.shape))
    #print(array_signal)
    #print('\n''\n')
    #print('Size of numpy array of background:{} '.format(array_bkg.shape))
    #print(array_bkg)

    #Create the columns with the labels of signal =1 and background=-1
    column_of_label_signal = np.ones(array_signal.shape[0])
    column_of_label_bkg = np.empty(array_bkg.shape[0],dtype = int)
    column_of_label_bkg.fill(-1)
    
    #print('Size of new column with signal labels: {}'.format(column_of_label_signal.size))
    #print(column_of_label_signal)
    #print('Size of new column with background labels: {}'.format(column_of_label_bkg.size))
    #print(column_of_label_bkg)
    

    #Merge the array with the signal and the array with the bkg for both labels and features
    array_y = np.concatenate((column_of_label_signal,column_of_label_bkg), axis=0)
    array_X = np.concatenate((array_signal,array_bkg), axis=0)
    #print('Final array of the labels: \n')
    #print(array_y)
    #print('Final array of the features: \n')
    #print(array_X)

    #split the array in test sample and training sample
    X_train, X_validation, y_train, y_validation = train_test_split(array_X, array_y, test_size=0.20, random_state=1)
    print('Size of X_train: {} \n'.format(X_train.shape))
    print('Size of X_validation: {} \n'.format(X_validation.shape))
    print('Size of y_train: {} \n'.format(y_train.shape))
    print('Size of y_validation: {} \n'.format(y_validation.shape))


    return X_train, X_validation, y_train, y_validation


def accurancy (label_true, label_predicted):
    """This function which gives the accurancy of the algorithm"""
    
    accurancy = np.sum(label_true == label_predicted) / len(label_true) 
    return accurancy


def cutter (array_to_cut, first_row, last_row):
    """this function cut the array gives as argument"""
    
    array_cutted=np.delete(array_to_cut, slice(first_row, last_row), axis=0)
    print('Size of cutted array: {} \n'.format(array_cutted.shape))
    return array_cutted


def training (csv_signal,csv_bkg, n_classifier):
    """This function performes the training of the algorithm"""
    
    X_train, X_validation, y_train, y_validation = get_dataset(csv_signal,csv_bkg)
    classification = Adaboost_Algorithm(n_classifier)
    
    X_train = cutter(X_train, 0, 160000)
    #X_validation = cutter(X_validation, 0, 40000)
    y_train = cutter(y_train, 0, 160000)
    #y_validation = cutter(y_validation, 0, 40000)
    
    print('The training is starting ...')
    
    classification.fit(X_train, y_train)
    y_pred = classification.predict(X_validation)
    acc=accurancy(y_validation,y_pred)
    print('The accurancy of the algorithm is : ',acc)
    
    # save the model to disk
    filename = 'Ada_classification.sav'
    pickle.dump(classification, open(filename, 'wb'))

#data = datasets.load_breast_cancer()
#X, y = data.data, data.target
#y[y == 0] = -1

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
#classification = Adaboost_Algorithm(5)
#classification.fit(X_train, y_train)
#y_pred = classification.predict(X_test)
#acc=accurancy(y_test,y_pred)
#print('The accurancy of the algorithm is : ',acc)


# load the model from disk
#X_train, X_validation, y_train, y_validation = get_dataset('Kpi_2018_signal.csv','pipi_2018_bkg.csv')
#loaded_classification = pickle.load(open('Ada_classification.sav', 'rb'))
#y_pred = loaded_classification.predict(X_validation)
#acc=accurancy(y_validation,y_pred)
#print('The accurancy of the algorithm is : ',acc)




