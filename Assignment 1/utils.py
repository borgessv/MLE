"""
Course: Machine Learning in Engineering
Assignment 1: Support Vector Machines (SVM)
Author: Vitor Borges Santos
Date: 16-02-2023

Description: creation of a auxiliary functions to be used by the class LinearSVM
             and by the main.py file.  
"""
import numpy as np

def create_2Ddataset(n=500, c1_ratio=0.5, mean=1, cov_factor=0.1, **kwargs):
    if 'seed' in kwargs:
        np.random.seed(kwargs.get('seed'))
        
    c1_mean = [-mean, -mean]
    c1_cov = [[cov_factor*np.abs(mean), 0], [0, cov_factor*np.abs(mean)]]       # diagonal covariance
    x1_c1, x2_c1 = np.random.multivariate_normal(c1_mean, c1_cov, 
                                                 int(np.ceil(c1_ratio*n))).T
    c1_data = np.column_stack((x1_c1, x2_c1, np.ones(len(x1_c1))))

    c2_mean = [mean, mean]
    c2_cov = [[cov_factor*np.abs(mean), 0], [0, cov_factor*np.abs(mean)]]       # diagonal covariance
    x1_c2, x2_c2 = np.random.multivariate_normal(c2_mean, c2_cov, 
                                                 int(n-np.ceil(c1_ratio*n))).T
    c2_data = np.column_stack((x1_c2, x2_c2, -1.*np.ones(len(x1_c2))))
    return np.row_stack((c1_data, c2_data))

def normalization(data, x):
    x_raw = data[:,:-1]
    x_new = np.zeros(np.shape(x))
    for i in range(np.shape(x)[1]):
        x_new[:,i] = (x[:,i] - np.min(x_raw[:,i]))/np.ptp(x_raw[:,i])
    return x_new

def denormalization(data, x):
    x_raw = data[:,:-1]
    x_new = np.zeros(np.shape(x))
    for i in range(np.shape(x)[1]):
        x_new[:,i] = x[:,i]*np.ptp(x_raw[:,i]) + np.min(x_raw[:,i])
    return x_new
    
    
    
