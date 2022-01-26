#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:22:02 2022

@author: lucatoscano
"""
import math
import pandas as pd
from pandas import read_csv
import numpy as np

def get_features(signal, bkg):
    """This function compute the variable needed for thetraining from the variable in the dataset"""
    
    #columns=['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_M','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P1_PT','P2_IPCHI2_OWNPV','P2_PT','m_corr','nSPDHits']

    #Read the file cvs
    #dataset_signal = read_csv('Kpi_2018_signal.csv', names=columns)
    #dataset_bkg = read_csv('pipi_2018_bkg.csv', names=columns)
    
    #Transform the dataset in pandas data frame
    #data_frame_signal = pd.DataFrame(dataset_signal)
    #data_frame_bkg = pd.DataFrame(dataset_bkg)
    data_frame_signal = signal
    data_frame_bkg = bkg
    
    print("Initial DataFrame:")
    print(data_frame_signal, "\n")
    print(data_frame_bkg, "\n")
    
    #remove the {} to the columns B_DTF_chi2 and B_DTF_nDOF
    data_frame_signal['B_DTF_chi2'] = data_frame_signal['B_DTF_chi2'].str.replace('[','',regex=True)
    data_frame_signal['B_DTF_chi2'] = data_frame_signal['B_DTF_chi2'].str.replace(']','',regex=True)
    data_frame_bkg['B_DTF_chi2'] = data_frame_bkg['B_DTF_chi2'].str.replace('[','',regex=True)
    data_frame_bkg['B_DTF_chi2'] = data_frame_bkg['B_DTF_chi2'].str.replace(']','',regex=True)
    
    data_frame_signal['B_DTF_nDOF'] = data_frame_signal['B_DTF_nDOF'].str.replace('[','',regex=True)
    data_frame_signal['B_DTF_nDOF'] = data_frame_signal['B_DTF_nDOF'].str.replace(']','',regex=True)
    data_frame_bkg['B_DTF_nDOF'] = data_frame_bkg['B_DTF_nDOF'].str.replace('[','',regex=True)
    data_frame_bkg['B_DTF_nDOF'] = data_frame_bkg['B_DTF_nDOF'].str.replace(']','',regex=True)
    
    data_frame_signal['B_DTF_chi2'] = pd.to_numeric(data_frame_signal['B_DTF_chi2'], downcast='float')
    data_frame_bkg['B_DTF_chi2'] = pd.to_numeric(data_frame_bkg['B_DTF_chi2'], downcast='float')
    
    data_frame_signal['B_DTF_nDOF'] = pd.to_numeric(data_frame_signal['B_DTF_nDOF'], downcast='float')
    data_frame_bkg['B_DTF_nDOF'] = pd.to_numeric(data_frame_bkg['B_DTF_nDOF'], downcast='float')
    
    print(data_frame_signal['B_DTF_chi2'])
    print(data_frame_bkg['B_DTF_chi2'])
    print(data_frame_signal['B_DTF_nDOF'])
    print(data_frame_bkg['B_DTF_nDOF'])
    
    #compute the new columns
    data_frame_signal['log(D0_FDCHI2_OWNPV)'] = data_frame_signal.apply(lambda column: np.log(column.D0_FDCHI2_OWNPV), axis=1)
    data_frame_signal['log(P1_IPCHI2_OWNPV)'] = data_frame_signal.apply(lambda column: np.log(column.P1_IPCHI2_OWNPV), axis=1)
    data_frame_signal['log(P2_IPCHI2_OWNPV)'] = data_frame_signal.apply(lambda column: np.log(column.P2_IPCHI2_OWNPV), axis=1)
    data_frame_signal['(D0_ENDVERTEX_Z - B_ENDVERTEX_Z) / sqrt(D0_ENDVERTEX_ZERR**2 + B_ENDVERTEX_ZERR**2)'] = data_frame_signal.apply(lambda column: (column.D0_ENDVERTEX_Z - column.B_ENDVERTEX_Z) / math.sqrt(column.D0_ENDVERTEX_ZERR**2 + column.B_ENDVERTEX_ZERR**2), axis=1)
    data_frame_signal[' B_DTF_chi2/B_DTF_nDOF'] = data_frame_signal.apply(lambda column:  column.B_DTF_chi2/column.B_DTF_nDOF, axis=1)
    
    data_frame_bkg['log(D0_FDCHI2_OWNPV)'] = data_frame_bkg.apply(lambda column: np.log(column.D0_FDCHI2_OWNPV), axis=1)
    data_frame_bkg['log(P1_IPCHI2_OWNPV)'] = data_frame_bkg.apply(lambda column: np.log(column.P1_IPCHI2_OWNPV), axis=1)
    data_frame_bkg['log(P2_IPCHI2_OWNPV)'] = data_frame_bkg.apply(lambda column: np.log(column.P2_IPCHI2_OWNPV), axis=1)
    data_frame_bkg['(D0_ENDVERTEX_Z - B_ENDVERTEX_Z) / sqrt(D0_ENDVERTEX_ZERR**2 + B_ENDVERTEX_ZERR**2)'] = data_frame_bkg.apply(lambda column: (column.D0_ENDVERTEX_Z - column.B_ENDVERTEX_Z) / math.sqrt(column.D0_ENDVERTEX_ZERR**2 + column.B_ENDVERTEX_ZERR**2), axis=1)
    data_frame_bkg[' B_DTF_chi2/B_DTF_nDOF'] = data_frame_bkg.apply(lambda column:  column.B_DTF_chi2/column.B_DTF_nDOF, axis=1)
    
    
    print("DataFrame after the addition of new columns")
    print(data_frame_signal, "\n")
    print(data_frame_bkg, "\n")
    
    data_frame_signal.drop(['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P2_IPCHI2_OWNPV'], axis=1, inplace=True)
    data_frame_bkg.drop(['B_ENDVERTEX_Z','B_ENDVERTEX_ZERR','B_DTF_chi2','B_DTF_nDOF','D0_ENDVERTEX_Z','D0_ENDVERTEX_ZERR','D0_FDCHI2_OWNPV','P1_IPCHI2_OWNPV','P2_IPCHI2_OWNPV'], axis=1, inplace=True)       
    
    print("DataFrame after the removal of old columns")
    print(data_frame_signal, "\n")
    print(data_frame_bkg, "\n")
    
    return data_frame_signal, data_frame_bkg
    
    
    
    
    
    
    
   