#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 10:58:05 2022

@author: lucatoscano
"""
import numpy as np

#The class of the decision stump used in the AdaBoost algorithm as weak classifier
#The class has 4 attributes in __init__ and one method called predict
class DecisionStump:
    
    def __init__(self):
        #FIRST ATTRIBUTE
        #Depending on the polarity, the stump classifies the event as -1 or 1.
        #With a positive polarity the class of the event is -1 if the examinated feature is lower then the threshold
        #With a negative  polarity the class of the event is -1 if the examinated feature is higher then the threshold
        self.polarity = 1
        
        #SECOND ATTRIBUTE
        #The index of the feature  used for the classification
        self.feature_index = None
        
        #THIRD ATTRIBUTE
        # The threshold of the feature used for the classification
        self.threshold = None
        
        #FOURTH ATTRIBUTE
        #The amount of say of a stump indicates how well the stamp classifies
        self.alpha = None
    
    #The function predict create the array of the classification
    #The operation is explained in details in the documentation
    
    def predict(self, X):
        number_of_samples= X.shape[0]
        
        X_column = X [:, self.feature_index]
        
        #predictions array is default equal to 1, with the size of the sample
        predictions = np.ones(number_of_samples)
        
        if self.polarity == 1:
            
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        
        return predictions
 
#The class Adaboost_Algorithm is the core of the training. It contain the algorithm
#The class take as argument the number of  decision stumps componing the forest
#The class has two attributes in __init__ and two methods called fit and predict
#The operation is explained in details in the documentation  
class Adaboost_Algorithm:
    
    def __init__(self, n_of_classifier = 5):
        #FIRST ATTRIBUTE
        self.n_of_classifier = n_of_classifier
        
        #SECOND ATTRIBUTE
        self.classifiers = []

    #the method fit need has arguments the features and the labels of the training sample X and  y
    def fit(self, X, y):
        
        n_of_samples, n_of_features= X.shape
        
        #the size of the weight array must be equal to the size of the n_of_sample
        # At the beginning of the algorithm all the samples must have the same initial weight equal to 1/number_of_samples 
        weight = np.full(n_of_samples, (1/n_of_samples))
        
        
        self.classifiers = []
        
        for classifier_i in range(self.n_of_classifier):
            print('CLASSIFIER N  {}'.format(classifier_i))
            classifier = DecisionStump()
            
            
            min_error = float("inf")
            
            for feature_i in range(n_of_features):
                print('training of feature {}'.format(feature_i))
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                
                for threshold_i in thresholds:
                    #print('threshold {}'.format(threshold_i))
                    pol = 1
                    predictions = np.ones(n_of_samples)
                    predictions[X_column < threshold_i] = -1
                    
                    #error is equal to the sum of the weight of misclassified sample
                    weight_misclassified= weight[y != predictions]
                    error= sum(weight_misclassified)
                    
                    #If there are more wrong classification events than right classification events:
                    #the error is flipped and the polarity change
                    # this correspond to assign the -1 prediction to the events higher than the threshold and +1 to the events lower
                    if error > 0.5:
                        error= 1 - error
                        pol = -1
                    
                    #if the error of the attual feature and threshold is smaller then the min_error, the current feature and threshold are better classifier then all the previous combinations
                    #the current feature becomes the new classifier and the min_error is uptaded to the current one
                    if error < min_error:
                        classifier.polarity = pol
                        classifier.threshold = threshold_i
                        classifier.feature_index = feature_i
                        min_error = error
                    
            EPS = 1e-10
            
            #initialize the amount of say alpha
            classifier.alpha = 0.5 * np.log( (1.0 - min_error + EPS) / (min_error + EPS) )
    
            predictions = classifier.predict(X)
            
            #formula for the update of the weights
            weight *= np.exp(-classifier.alpha * y * predictions)
            
            #normalize the weight
            weight /= np.sum(weight)
            
            #if the weight are well normalized, the sum of weight must close to 1
            def test_weight_normalization():
                
                assert np.sum(weight)>0.99
            
            #update the classifier
            self.classifiers.append(classifier)
            
    def predict(self, X):
        classifier_pred = [classifier.alpha * classifier.predict(X) for classifier in self.classifiers]
        y_pred = np.sum(classifier_pred, axis = 0)
        y_pred = np.sign(y_pred)
        
        return y_pred
            
        
        
        
        
        
        
        
        
        
        
        
        
        