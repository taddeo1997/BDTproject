#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:22:02 2022

@author: lucatoscano
"""
import math
import pandas as pd
import numpy as np

def get_features(dataset):
    """This function compute the variable needed for thetraining from the variable in the dataset"""
    

    #remove the [] to the columns B_DTF_chi2 and B_DTF_nDOF
    dataset['B_DTF_chi2'] = dataset['B_DTF_chi2'].str.replace('[','',regex=True)
    dataset['B_DTF_chi2'] = dataset['B_DTF_chi2'].str.replace(']','',regex=True)
 
    
    dataset['B_DTF_nDOF'] = dataset['B_DTF_nDOF'].str.replace('[','',regex=True)
    dataset['B_DTF_nDOF'] = dataset['B_DTF_nDOF'].str.replace(']','',regex=True)

    
    dataset['B_DTF_chi2'] = pd.to_numeric(dataset['B_DTF_chi2'], downcast='float')
  
    
    dataset['B_DTF_nDOF'] = pd.to_numeric(dataset['B_DTF_nDOF'], downcast='float')
 
    #compute the new columns
    dataset['log(D0_FDCHI2_OWNPV)'] = dataset.apply(lambda column: np.log(column.D0_FDCHI2_OWNPV), axis=1)
    dataset['log(P1_IPCHI2_OWNPV)'] = dataset.apply(lambda column: np.log(column.P1_IPCHI2_OWNPV), axis=1)
    dataset['log(P2_IPCHI2_OWNPV)'] = dataset.apply(lambda column: np.log(column.P2_IPCHI2_OWNPV), axis=1)
    dataset['(D0_ENDVERTEX_Z - B_ENDVERTEX_Z) / sqrt(D0_ENDVERTEX_ZERR**2 + B_ENDVERTEX_ZERR**2)'] = dataset.apply(lambda column: (column.D0_ENDVERTEX_Z - column.B_ENDVERTEX_Z) / math.sqrt(column.D0_ENDVERTEX_ZERR**2 + column.B_ENDVERTEX_ZERR**2), axis=1)
    dataset[' B_DTF_chi2/B_DTF_nDOF'] = dataset.apply(lambda column:  column.B_DTF_chi2/column.B_DTF_nDOF, axis=1)
    

    
    dataset.drop(['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P2_IPCHI2_OWNPV'], axis=1, inplace=True)
    
    return dataset
    
    
    
    
    
    
    
   