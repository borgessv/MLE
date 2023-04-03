"""
Course: Machine Learning in Engineering
Assignment 3
Author: Vitor Borges Santos
Date: 27-03-2023

Description: Neural networks and Support Vector Machines for classification and
             regression.
Limitations: Binary classification and regression for single input-output 
"""
# %% Load libraries and functions
import numpy as np
from models import SVM, NN
from utils import plot_fun, plot_cm

# %% User inputs
models = ["svm","nn"]                                                           # options: "nn" or/and "svm"

# Data Handling:
dataset = "NonReg"                                                              # options: .txt file or "random" for a random dataset created from a multivariate normal distribution
normalize = False                                                               # options: True or False    
shuffle = False                                                                 # options: True or False
train_split = 0.4                                                               # options: any value in (0, 1]    
validation_split = 0.5                                                          # options: any value in [0, 1] with respect to the test data
pred_data = "test"                                                              # options: "test" or array(number of points x number of features)
seed = 0                                                                        # options: None or any integer number

# Neural Network Settings:
nn_problem = "regression"
time_steps = 4
hidden_dim = [5]
activation = ["tanh"]
nn_optimizer = "adam"
learning_rate = 0.001
batch_size = 4
epochs = 100

# SVM Settings:
svm_problem = "regression"                                                      # options: "classification", "regression" or "expansion"
svm_optimizer = "quadprog"                                                      # options: "cvxopt" or "quadprog"
dist = 1e-2                                                                     # disturbance to reduce noise in data or force quadratic term of optimization problem to be positive-definite
eps_r = 0                                                                       # maximum absolute deviation from the training points (regression only)                                                            
margin = "hard"                                                                 # options: "hard" or "soft"
C = 50                                                                          # scalar parameter defining the contribution of slack variables to the weight vector 
kernel = "rbf"                                                                  # options: "linear", "rbf" (requires sigma)
sigma = np.sqrt(625)                                                            # options: real number (required for "rbf" kernel)

# Plot Settings:
plot_filename = dataset                                                         # filename used in plots  
image_format = "svg"                                                            # image format (svg, jpeg, pdf, ...), note that if eps format is used any transparency will be lost
feature_name = [r"$x$", r"$f(x)$"]                                              # names of the features to be used in plot
class_name = [r"Al", r"Cu"]                                                     # names of the classes (for classification only)
confusion_matrix = False                                                        # plots the confusion matrix    

#%% Model implementation:
ml_obj = []                       
for model in [item for item in models]:
    match model:
        case "svm":                                                             # SVM implementation 
            SVMmodel = SVM(dataset, train_split, pred_data, normalize, shuffle, # creates the SVM model and pre-process the dataset
                           svm_problem, margin, kernel, seed, C, sigma, eps_r, 
                           validation_split)         
            lm,b,sp = SVMmodel.solve_svm(solver=svm_optimizer,disturbance=dist) # calculates the lagrange multipliers, bias and the indexes of support vectors
            y_pred_svm = SVMmodel.predict(pred_data)                            # predicts the value/class for a new point
            ml_obj.append(SVMmodel)
            if confusion_matrix:                                                # plots confusion matrix
                plot_cm(SVMmodel, plot_filename, image_format, class_name)          
        case "nn" :                                                             # NN implementation
            NNmodel = NN(dataset,train_split, pred_data, normalize, shuffle,    # creates the NN model and pre-process the dataset
                         nn_problem, hidden_dim, activation, nn_optimizer, 
                         learning_rate, batch_size, epochs, seed, 
                         validation_split, time_steps)
            NNmodel.create()                                                    # builds the neural network
            train_loss, train_acc, val_loss, val_acc = NNmodel.train()          # trains the neural network
            y_pred_nn = NNmodel.predict(pred_data)                              # predicts the value/class for a new point
            ml_obj.append(NNmodel)

plot_fun(ml_obj,plot_filename,image_format,feature_name,class_name)             # plots the solution 
