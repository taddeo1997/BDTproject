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



