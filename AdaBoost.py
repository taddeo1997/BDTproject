#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:58:05 2022

@author: lucatoscano
"""
import numpy as np

#Create the class of the decision stump used in the AdaBoost algorithm as weak classifier 
class DecisionStump:
    
    def __init__(self):
        #says if the sample is classified as +1 or -1 for the given threshold if I want to flip the error I must flip the polarity
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
            #if sample of column X is smaller then threshold, prediciton is -1
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
            return predictions
        
class Adaboost_algorithm:
    
    def __init__(self, n_of_classifier = 5):
        self.n_of_classifier_ = n_of_classifier
        

    #the method fit need has arguments the training sample X, and the labels y
    def fit(self, X, y):
        #first thing is get the shape of the sample
        n_of_samples, n_of_features= X.shape
        
        #initialize the weight using method full. The size of the array is equal
        #to the size of the training sample
        #the initial value is equal to 1/number of samples
        weight = np.full(n_of_samples, (1/n_of_samples))
        
        #create a list to store all the classifier (decision stumps)
        self.classifiers = []
        for _ in range(self.n_of_calssifier):
            classifier = DecisionStump()
            
            #I have to find the feature and the thersold 
            #for which this error is miminum
            min_error = float('inf')
            for feature_i in range(n_of_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                
                for threshold_i in thresholds:
                    pol = 1
                    predictions = np.ones(n_of_samples)
                    predictions[X_column < threshold_i] = -1
                    
                    #error is equal to the sum of the weight of 
                    #misclassified sample
                    weight_misclassified= weight[y != predictions]
                    error= sum(weight_misclassified)
                    
                    count_array_y=np.bincount(y)
                    count_array_predictions=np.bincount(predictions)
                    
                    #Compute gini coeff
                    Gini_neg = 1 - (count_array_y[-1]/count_array_predictions[-1])**2 (count_array_y[+1]/count_array_predictions[-1])**2
                    Gini_pos = 1 - (count_array_y[-1]/count_array_predictions[+1])**2 (count_array_y[+1]/count_array_predictions[+1])**2
                    classifier.Gini=(count_array_predictions[-1]/n_of_samples)*Gini_neg + (count_array_predictions[+1]/n_of_samples)*Gini_pos
            
                    #flip the error if it's larger then 0.5
                    if error > 0.5:
                        error= 1-error
                        pol = -1
                    
                    #we want to chek if the error is smaller then the min_error
                    if error < min_error:
                        min_error = error
                        classifier.polarity = pol
                        classifier.threshold = threshold_i
                        classifier.feature_index = feature_i
                        
            #initialize the amount of say alpha
            EPS = 1e-10
            classifier.alpha = 0.5 * np.log( (1-min_error) / (min_error+EPS) )
            
            #
            predictions = classifier.predict(X)
            #formula for the update of the weights
            weight *= np.exp(classifier.alpha * y * predictions)
            #normalize the weight
            weight /= np.sum(weight)
            
            #save the classifier
            self.classifiers.append(classifier)
            
        def predict(self, X):
            classifier_pred = [classifier.alhpa * classifier.predict(X) for classifier in self.classifiers]
            y_pred = np.sum(classifier_pred, axis = 0)
            y_pred = np.sign(y_pred)
            return y_pred
            
        
        
        
        
        
        
        
        
        
        
        
        
        