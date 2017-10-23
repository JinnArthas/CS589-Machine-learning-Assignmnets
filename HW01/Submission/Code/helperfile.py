# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 00:03:18 2017

@author: ravi
"""

# Importing Library
from sklearn.tree import DecisionTreeRegressor # library to perform DecisionTree Regression
from sklearn.neighbors import KNeighborsRegressor # Kneighbourneighbor regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import KFold # cross Validation 
import time # 
import matplotlib.pyplot as plt


# Mean Absolute error 
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

# Implementing Grid Search function to return the best parameter for the modal using k fold
# cross validation, X in the input training examples and Y is target
# Model will return best parameter and list of error with all parameter and 
#time it took to run each modal in miliseconds if true 

def GridSearch(parameters, classifier , n_splits, X, Y, Time):
    
    # Variable to store the parameter value along with MAE error in case of decision
    # tree this will contain tree depth and its MAE error value
    bestparameter = []
    
    # Varialble to stroe the time it took to run model for each parameter
    modal_time = [[], []]
    
    # Looping over all the parameters
    for parameter in parameters:
        
        # Kfold cross validation method 
        k_fold = KFold(n_splits= n_splits, shuffle = True)
        
        # Logic for classifier (if classifier is equals to DT the clf will be initilized as DecisionTree..)
        if classifier == "DT":
            clf = DecisionTreeRegressor(max_depth= parameter)
            
        elif classifier == "KNN":
            clf = KNeighborsRegressor(n_neighbors= parameter)
            
        elif classifier == "Ridge":
            clf = Ridge(alpha = parameter)
            
        elif classifier == "Lasso":
            clf = Lasso(alpha = parameter)
            
        else:
            raise ValueError("Classifier not defined")
        
        # Varialble which stores MAE for each iteration of kfold 
        score = 0
        
        # Storing current time
        if Time == True:
            x = time.time()
        
        # Classifing kfold training set and predicting kfold test data and reporting MAE to
        # the score variable 
        for k, (train_index, test_index) in enumerate(k_fold.split(X, Y)):      
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, y_train)
            score += compute_error(clf.predict(X_test), y_test)
        
        # Time to run the fold for one parameter value 
        if Time == True:
            y  = time.time()
            modal_time[0].append(parameter)
            modal_time[1].append((y - x)*1000)
        
        # Appending the MAE score of one iteration along with the parameter value 
        bestparameter.append([float(score)/float(n_splits), parameter])
    
    # sorting the MAE to get the best parameter value      
    bestparameter.sort()
    
    # returning all parameters along with time it took to run the parameter if true
    if Time == True:
        return bestparameter[0][1], bestparameter, modal_time
    
    # if time is not true return just parameters
    return bestparameter[0][1], bestparameter

# Function to to run the Decision Tree classifier 
def DecisionTree(k, X, y, x_test):
    clf = DecisionTreeRegressor(max_depth= k)
    clf.fit(X, y)
    return clf.predict(x_test)

# Helper function to train knn with defalut parameters
def knn(k, X, y, x_test):
    clf = KNeighborsRegressor(n_neighbors = k)
    clf.fit(X, y)
    return clf.predict(x_test)

# Helper function to train lasso regressor with default parameters
def lasso(k, X, y, x_test):
    clf = Lasso(alpha = k)
    clf.fit(X, y)
    return clf.predict(x_test)

# helper function to train ridge regressor with default parameters
def ridge(k, X, y, x_test):
    clf = Ridge(alpha = k)
    clf.fit(X, y)
    return clf.predict(x_test)

# plots function takes the list of parameter and time and plot a histogram
# X_axis_label and y_axis_label takes the label for respective axis
# you can also provide title to the figure
def plot(data, x_axis_label, y_axis_label = "Time (In milliseconds)", title = ""):
    plt.bar(data[0][:], height = data[1][:])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.ylim([min(data[1]) - 50, max(data[1]) + 50])
    plt.title(title)
