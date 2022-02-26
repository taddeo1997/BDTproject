#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:42:08 2022

@author: lucatoscano
"""
#Import Libraries
import numpy as np
from pandas import read_csv
from get_features import get_features
import pickle
import matplotlib.pyplot as plt



def Classification (csv_dataset, model):
    columns=['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_M','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P1_PT','P2_IPCHI2_OWNPV','P2_PT','m_corr','nSPDHits','D0_M']

    raw_dataset = read_csv(csv_dataset, names=columns)
    
    #the invariant mass of D0 is not used for the traonign, only for show the final results
    D0_mass_column = raw_dataset.iloc[:,-1:]
    
    array_D0_mass_column = np.array(D0_mass_column.values)
    
    raw_dataset.drop('D0_M',axis=1, inplace=True)
    
    #the 10 kinematic variable are computed from the raw database by the function get_features
    dataset = get_features(raw_dataset)
    
    array_dataset = np.array(dataset.values)
    
    
    
    # load the model from disk
    loaded_classification = pickle.load(open(model, 'rb'))
    array_predicted_label = loaded_classification.predict(array_dataset)
    
    
    #the arrays of the features, the D0 mass and the predicted labels are merged to extract the value of signal and bkg
    classified_dataset =  np.c_[ array_dataset, array_D0_mass_column,array_predicted_label ]  
 
    
    #the +1 label is for the signal, -1 for bkg
    signal_dataset = classified_dataset[np.where(classified_dataset[:,-1] ==1)]
    bkg_dataset = classified_dataset[np.where(classified_dataset[:,-1] ==-1)]
    
    # sum of the lenghts of the signal_dataset and bkg dataset must be equal to the lenght of the classified_database
    def test_weight_normalization():
        
        assert len(signal_dataset)+len(bkg_dataset)==len(classified_dataset)
    
 
    plt.hist(array_D0_mass_column, 50, label='signal+bkg' )
    plt.hist(signal_dataset[:,-2], 50, label='signal')
    plt.xlim([1820, 1950])
    plt.xlabel("D0 invariant mass")
    plt.ylabel("Events")
    plt.legend(loc='upper right')
    plt.title("Distribution D0 invariant mass (signal+bkg)")
    plt.show



















