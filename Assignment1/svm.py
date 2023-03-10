"""
Course: Machine Learning in Engineering
Assignment 1: Support Vector Machines (SVM)
Author: Vitor Borges Santos
Date: 16-02-2023

Description: creation of a class for a linear SVM to be used by main.py. 
"""
import numpy as np
import qpsolvers
from utils import normalization

class LinearSVM:
    def __init__(self, data, train_split, normalize, shuffle, margin, seed):    
        self.raw_data = data
        self.train_split = train_split
        self.normalize = normalize
        self.shuffle = shuffle
        self.margin = margin
        self.seed = seed
        self.data_handler()
        
    def data_handler(self):                                                     # function to pre-process the dataset
        if self.seed is False:
            np.random.seed()
        elif self.seed == int(self.seed):
            np.random.seed(self.seed)
        else:
            raise Exception('"seed" must be "False" or an integer!')
    
        if self.shuffle:                                                        # shuffles dataset (if applicable)
            np.random.shuffle(self.raw_data)  
            
        x_raw = self.raw_data[:,:-1]                                            
        y_raw = self.raw_data[:,-1]                                             
        raw_classes = np.unique(y_raw)                                          # gets the values assigned for the classes
        if len(raw_classes) != 2:                                               # makes sure number of classes is 2
            raise Exception('Number of classes is not equal 2!')
       
        y = y_raw                                                               # initialization of pre-processed classes vector           
        if raw_classes[0] != 1. or raw_classes[0] != -1.:                       # makes sure the classes are +1 and -1
            if raw_classes[1] != 1. :
                y[np.isin(y, raw_classes[0])] = 1.
            else:
                y[np.isin(y, raw_classes[0])] = -1.
        if raw_classes[1] != 1. or raw_classes[1] != -1.:
            if raw_classes[0] != 1. :
                y[np.isin(y, raw_classes[1])] = 1.
            else:
                y[np.isin(y, raw_classes[1])] = -1.
        
        self.raw_data = np.column_stack((x_raw, y)) 
        self.train_data = self.raw_data[:int(np.ceil(self.train_split*self.raw_data.shape[0])), :]
        self.test_data = self.raw_data[int(np.ceil(self.train_split*self.raw_data.shape[0])):, :]
        
        if self.normalize:                                                      # normalizes features (if applicable)
            x_train = normalization(self.raw_data, self.train_data[:,:-1])    
        else:
            x_train = self.train_data[:,:-1]
        y_train = self.train_data[:,-1]
        self.train_data = np.column_stack((x_train, y_train))
        return

    def solve_qp(self, solver="quadprog", eps_lm=1e-3, **kwargs): 
        if solver not in qpsolvers.available_solvers:
            raise Exception("The specified QP solver is not available!")              
        x, y = self.train_data[:,:-1], self.train_data[:,-1]
        M = len(y)
        y = np.reshape(y, (-1, 1))
        
        # Dual cost function for optimization: min 1/2*lambda.T*P*lambda + q.T*lambda:
        P = (y@y.T)*(x@x.T)                                                     # matrix in the quadratic term of cost function
        q = -np.ones(M)                                                         # vector in the linear term of cost function
        A = y.T                                                                 # equality constraint y.T*lambda = 0  --> A*x = b
        b = np.zeros(1)       
        P = P + 1e-10*np.eye(P.shape[0])                                        # Applies a small disturbance to guarantee P is positive definite  
        
        if self.margin == "hard":
            G = -np.eye(y.shape[0])                                             # inequality constraint -lambda <= 0  --> G <= h
            h = np.zeros(y.shape[0])
        else:
            C = kwargs.get('C')
            G = np.vstack((-np.eye(y.shape[0]), np.eye(y.shape[0])))            # inequality constraint -lambda <= 0 & lambda <= C  --> Gx <= h
            h = np.hstack((np.zeros(y.shape[0]), C*np.ones(y.shape[0])))
        
        lm = qpsolvers.solve_qp(P, q, G, h, A, b, solver=solver)
        sp = np.where(lm > eps_lm)
        if lm[sp] is None:
            raise Exception("The criteria 'eps_lm' is probably too high!")
        return lm, sp
    
    def svm_sol(self, lm, sp):                                                  # calculates the weight vector and the mean bias
        x_train, y_train = self.train_data[:,:-1], self.train_data[:,-1]    
        n_sp = len(sp[0])
        self.w = (lm*y_train.T)@x_train
        self.b = 1/n_sp*sum(y_train[sp] - self.w@x_train[sp].T)        
        return self.w, self.b
    
    def svm_predict(self, x_new):                                               # predicts the class of new examples 
        y_new = np.sign(self.w@x_new.T + self.b)
        return y_new
        
