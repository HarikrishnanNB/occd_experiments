# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 22:46:02 2020

@author: harik
"""

import numpy as np
import pandas as pd
import logging
def get_data(data_name):
    """
    Parameters
    ----------
    data_name : TYPE - string
        data_name can take two input as follows.
        data_name == "concentric_circle" -- will load the data set corresponding to concentric circle data (CCD)
        data_name == "cocentric_circle_noise" -- will load the data set corresponding to overlapping concentic circle data (OCCD).

    Returns
    -------
    X_train_norm (normalized train data)
    trainlabel
    X_test_norm (normalized test data)
    testlabel
    """    
    if data_name == "concentric_circle":
        folder_path = "Data/" + data_name + "/" 
          
        # Load Train data
        X_train = np.array( pd.read_csv(folder_path+"X_train.csv", header = None) ) 
        # Load Train label
        trainlabel =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )
        # Load Test data
        X_test = np.array( pd.read_csv(folder_path+"X_test.csv", header = None) )
        # Load Test label
        testlabel = np.array( pd.read_csv(folder_path+"y_test.csv", header = None) )
        
        
        
        ## Data_normalization - A Compulsory step
        # Normalization is done along each column
        
        X_train_norm = (X_train - np.min(X_train, 0))/(np.max(X_train, 0) - np.min(X_train, 0))
        X_test_norm = (X_test - np.min(X_test, 0))/(np.max(X_test, 0) - np.min(X_test, 0))
        
        
        
        try:
            assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
         
        return X_train_norm, trainlabel, X_test_norm, testlabel
    
    
    elif data_name == "concentric_circle_noise":
        folder_path = "Data/" + data_name + "/" 
          
        # Load Train data
        X_train = np.array( pd.read_csv(folder_path+"X_train.csv", header = None) ) 
        # Load Train label
        trainlabel =  np.array( pd.read_csv(folder_path+"y_train.csv", header = None) )
        # Load Test data
        X_test = np.array( pd.read_csv(folder_path+"X_test.csv", header = None) )
        # Load Test label
        testlabel = np.array( pd.read_csv(folder_path+"y_test.csv", header = None) )
        
        
        
        ## Data_normalization - A Compulsory step
        # Normalization is done along each column
        
        X_train_norm = (X_train - np.min(X_train, 0))/(np.max(X_train, 0) - np.min(X_train, 0))
        X_test_norm = (X_test - np.min(X_test, 0))/(np.max(X_test, 0) - np.min(X_test, 0))
        
        
        
        try:
            assert np.min(X_train_norm) >= 0.0 and np.max(X_train_norm <= 1.0)
        except AssertionError:
            logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
            
        try:
            assert np.min(X_test_norm) >= 0.0 and np.max(X_test_norm <= 1.0)
        except AssertionError:
            logging.error("Test Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)
         
        return X_train_norm, trainlabel, X_test_norm, testlabel
     