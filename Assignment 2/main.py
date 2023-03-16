"""
Course: Machine Learning in Engineering
Assignment 2 - Problem 2
Author: Vitor Borges Santos
Date: 16-03-2023

Description: Nonlinear SVM algorithm for regression and classification.
Limitations: classification can handle only 2 classes 
"""
# %% Load libraries and functions
import numpy as np
from svm import SVM
from utils import plot_fun, plot_cm

# %% User inputs
# Data Handling:
dataset = "NonReg"                                                              # options: .txt file or "random" for a random dataset created from a multivariate normal distribution
pred_data = "test"                                                              # options: "test" or array(number of points x number of features) 
normalize = False                                                               # options: True or False    
train_split = 0.75                                                              # options: any value in (0, 1]    
shuffle = False                                                                 # options: True or False
seed = None                                                                     # options: None or any integer number

# SVM Settings:
optimizer = "quadprog"                                                          # options: "cvxopt" or "quadprog"
disturbance = 1e-2                                                              # disturbance to reduce noise in data or force quadratic term of optimization problem to be positive-definite
problem = "expansion"                                                           # options: "classification", "regression" or "expansion"
eps_r = 0                                                                     
margin = "hard"                                                                 # options: "hard" or "soft"
C = 50                                                                          # scalar parameter defining the contribution of slack variables to the weight vector 
kernel = "rbf"                                                                  # options: "linear", "rbf" (requires sigma)
sigma = np.sqrt(625)                                                            # options: real number (required for "rbf" kernel)

# Plot Settings:
plot_filename = dataset                                                         # filename used in plots  
image_format = "svg"                                                            # image format (svg, jpeg, pdf, ...), note that if eps format is used any transparency will be lost
feature_name = [r"$x$", r"$f(x)$"]                                              # names of the features to be used in plot
class_name = ["class +1", "class \N{MINUS SIGN}1"]                              # names of the classes (for classification only)
confusion_matrix = False                                                        # plots the confusion matrix    
# %% SVM implementation                       
model = SVM(dataset, train_split, normalize, shuffle, problem, margin, kernel,  # creates the model and pre-process the dataset
            seed, C, sigma, eps_r)         
lm, b, sp = model.solve_svm(solver=optimizer, disturbance=disturbance)          # calculates the lagrange multipliers, bias and the indexes of support vectors

# %% Prediction
y_pred = model.svm_predict(pred_data)                                           # predicts the class of new examples
plot_fun(model, plot_filename, image_format, feature_name, class_name)          # plots result
if confusion_matrix:
    plot_cm(model, plot_filename, image_format, class_name)                     # plots confusion matrix
