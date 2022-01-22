#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:58:05 2022

@author: lucatoscano
"""
import numpy as np

#Create the lcass of the decision stump used in the AdaBoost algorithm
class DecisionStump:
    
    def __init__(self):
        #says if the sample is classified as +1 or -1 for the given threshold
        #if I want to flip the error I must flip the polarity
        self.polarity = 1
        self.feature_index = None
        #threshold of the feature, for the decision of the stump
        self.threshold = None
        #the amount of say of a stump indicates how well the stamp classifies
        self.amount_of_say = None
    
    #I want to compare a feature of the sample X, set a thersold and set +1
    #or -1 if the event has X larger or smaller then the thershold
    def predict(self, X):
        number_of_samples= X.shape[0]
        X_column = X[:, self.feautre_index]
        
        #predictions array is default equal to 1, with the size of the sample
        predictions = np.ones(number_of_samples)
        if self.polarity == 1:
            #if feature is smaller then threshold prediciton is -1
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
            return predictions
        

        
        
        
        
        
        
        
        
        
        
        
        
        