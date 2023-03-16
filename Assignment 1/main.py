"""
Course: Machine Learning in Engineering
Assignment 1: Support Vector Machines (SVM)
Author: Vitor Borges Santos
Date: 16-02-2023

Description: Linear classifier SVM algorithm.
Limitations: limited to 2-dimensional (2 features) datasets and 2 classes 
"""
# %% Load libraries and functions
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics
from svm import LinearSVM
from utils import create_2Ddataset, normalization, denormalization

# %% User inputs
# Data Handling:
dataset = "DataERC1.txt"                                                        # options: .txt file or "random" for a random dataset created from a multivariate normal distribution
pred_data = "test"                                                              # options: "test" or 2-column array 
normalize = True                                                                # options: True or False    
train_split = 0.25                                                               # options: any value in [0, 1]    
shuffle = True                                                                  # options: True or False
seed = 1                                                                        # options: False or any integer number
# Optimization Settings:
margin = "hard"                                                                 # options: "hard" or "soft"
C = 50                                                                         # scalar parameter defining the contribution of slack variables to the weight vector 
optimizer = "quadprog"                                                          # options: "cvxopt" or "quadprog"
# Plot Settings:
plot_filename = dataset + "_" + str(int(train_split*100))                                # filename used in plots  
image_format = "svg"                                                            # image format (svg, jpeg, pdf, ...), note that if eps format is used any transparency will be lost
feature_name = ["relative conductivity", "relative permeability"]               # names of the features to be used in plot
class_name = ["Al", "Cu"]                                                       # names of the classes

# %% SVM implementation
match dataset:                                                                  # loads or creates dataset
    case "random":
        data = create_2Ddataset()                                    
    case other:
        data = np.loadtxt(dataset, skiprows=1)                                
        
model = LinearSVM(data, train_split, normalize, shuffle, margin, seed)          # creates the model and pre-process the dataset        

lm, sp = model.solve_qp(solver=optimizer, C=C)
w, b = model.svm_sol(lm, sp)                                                    # calculates the weight vector and bias (mean)

# %% Class prediction
match pred_data:
    case "test":
        x_test, y_test_true = model.test_data[:,:-1], model.test_data[:,-1]
    case other:
        x_test = pred_data
if model.normalize:                                                             # normalizes the data to be predicted (if applicable)
    x_test = normalization(model.raw_data, x_test)

y_test = model.svm_predict(x_test)                                              # predicts the class of new examples


# %% Hyperplane, margins and mispredicted points
xhp = np.linspace(-0.05, 1.05, 2)                                               # x-axis points defining the hyperplane 
yhp = (-w[0]/w[1])*xhp -b/w[1]                                                  # y-axis points defining the hyperplane                                               
hp = np.array(list(zip(xhp, yhp))) 
hp_margin = 1/np.linalg.norm(w)                                                 # margin distance
w_hat = w/np.linalg.norm(w)                                                     # unit vector relative to the weight vector
margin1 = hp + w_hat*hp_margin                                                  # margin above hyperplane
margin2 = hp - w_hat*hp_margin                                                  # margin below hyperplane

val_y = y_test_true - y_test                                                    # compares predicted and true classes
false_c1 = np.where(val_y == 2)                                                 # indexes of false class 1 prediction
false_c2 = np.where(val_y == -2)                                                # indexes of false class 2 prediction
x_test_fc1 = x_test[false_c1]                                                   # class 2 - misclassified examples
x_test_fc2 = x_test[false_c2]                                                   # class 1 - misclassified examples
c1_data_train = model.train_data[model.train_data[:, 2] == 1, :-1]              # class 1 - train examples  
c2_data_train =  model.train_data[model.train_data[:, 2] == -1, :-1]            # class 2 - train examples 
c1_data_test =  x_test[y_test_true == 1]                                        # class 1 - test examples
c2_data_test = x_test[y_test_true == -1]                                        # class 2 - test examples

if model.normalize:                                                             # denormalizes points to be plotted (if applicable)
    hp = denormalization(model.raw_data, hp)
    margin1 = denormalization(model.raw_data, margin1)
    margin2 = denormalization(model.raw_data, margin2)
    c1_data_train = denormalization(model.raw_data, c1_data_train)                                        
    c2_data_train = denormalization(model.raw_data, c2_data_train)                                   
    c1_data_test = denormalization(model.raw_data, c1_data_test)                                                   
    c2_data_test = denormalization(model.raw_data, c2_data_test) 
    x_test_fc1 = denormalization(model.raw_data, x_test_fc1)                    
    x_test_fc2 = denormalization(model.raw_data, x_test_fc2)                    


# %% Plots
from IPython import get_ipython                                                 # forces plots to be shown in plot pane
ipython = get_ipython()
ipython.magic('matplotlib inline')

fig = plt.figure()
plt.plot(c1_data_train[:,0], c1_data_train[:, 1],'o', color='tab:blue', 
         alpha=0.5, label=class_name[0] + " - train")
plt.plot(c2_data_train[:,0], c2_data_train[:,1],'o', color='tab:orange', 
         alpha=0.5, label=class_name[1] + " - train")
plt.plot(c1_data_test[:,0], c1_data_test[:, 1],'o', color='tab:blue', 
         label=class_name[0] + " - test")
plt.plot(c2_data_test[:,0], c2_data_test[:,1],'o', color='tab:orange', 
         label=class_name[1] + " - test")
if x_test_fc2.size != 0:
    plt.plot(x_test_fc2[:,0], x_test_fc2[:,1],'ob', 
             label=class_name[0] + " - false")
if x_test_fc1.size != 0:
    plt.plot(x_test_fc1[:,0], x_test_fc1[:,1], 'or', 
             label=class_name[1] + " - false") 
plt.plot(hp[:,0], hp[:,1], '-k', label="hyperplane")
plt.plot(margin1[:,0], margin1[:,1], '--g', label="margin")
plt.plot(margin2[:,0], margin2[:,1], '--g', label="_nolegend_")

x_lim = [min(model.raw_data[:,0])*(1 - 0.1*np.sign(min(model.raw_data[:,0]))),  
         max(model.raw_data[:,0])*(1 + 0.1*np.sign(max(model.raw_data[:,0])))]
y_lim = [min(model.raw_data[:,1])*(1 - 0.1*np.sign(min(model.raw_data[:,1]))),
         max(model.raw_data[:,1])*(1 + 0.1*np.sign(max(model.raw_data[:,1])))]
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.xlabel(feature_name[0])
plt.ylabel(feature_name[1])
plt.legend()
plt.savefig(plot_filename + "." + image_format, transparent=True, dpi=1200)
plt.show()


#%% Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test_true, y_test)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix,
                                            display_labels = ["Al", "Cu"])
cm_display.plot()
plt.savefig(plot_filename + "_CM." + image_format, transparent=True, dpi=1200)
plt.show() 
