"""
Course: Machine Learning in Engineering
Assignment 2: Problem 2
Author: Vitor Borges Santos
Date: 02-03-2023

Description: creation of a class for a linear SVM to be used by main.py. 
"""
import numpy as np
import scipy.optimize as opt
import qpsolvers
from utils import create_2Ddataset, normalization

class SVM:
    def __init__(self, dataset, train_split, normalize, shuffle, problem, margin, 
                 kernel, seed=None, C=None, sigma=None, eps_r=None):
        match dataset:                                                          # loads or creates dataset
            case "random":
                self.raw_data = create_2Ddataset(seed=seed)                                    
            case _:
                self.raw_data = np.loadtxt(dataset+".txt", skiprows=1) 
        self.train_split = train_split
        self.normalize = normalize
        self.shuffle = shuffle
        self.problem = problem
        self.margin = margin
        self.kernel = kernel
        self.seed = seed
        self.C = C
        self.sigma = sigma
        self.eps_r = eps_r
        self.data_handler()       
                   
    def data_handler(self):                                                     # function to pre-process the dataset
        np.random.seed(self.seed)   
        if self.shuffle:                                                        # shuffles dataset (if applicable)
            np.random.shuffle(self.raw_data)  
        x_raw, y_raw = self.raw_data[:,:-1], self.raw_data[:,-1]                                            
        if self.problem == "classification":                                             
            raw_classes = np.unique(y_raw)                                      # gets the values assigned for the classes
            if len(raw_classes) != 2:                                           # makes sure number of classes is 2
                raise Exception('Number of classes is not equal 2!')
            y = y_raw                                                                      
            if raw_classes[0] != 1. or raw_classes[0] != -1.:                   # makes sure the classes are +1 and -1
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
            x_train = normalization(x_raw, self.train_data[:,:-1])
            if self.problem == "regression":
                y_train = normalization(y_raw, self.train_data[:,-1])
            else:
                y_train = self.train_data[:,-1]
        else:
            x_train = self.train_data[:,:-1]
            y_train = self.train_data[:,-1]
        self.train_data = np.column_stack((x_train, y_train))
        return

    def solve_svm(self, solver="quadprog", eps_lm=1e-3, disturbance=1e-10): 
        if solver not in qpsolvers.available_solvers:
            raise Exception("The specified QP solver is not available!")
        if disturbance is None: disturbance = 0.   
           
        x, y = self.train_data[:,:-1], self.train_data[:,-1]
        K = self.kernel_fun(x, x)                                               # calculates the kernel function    
        
        match self.problem:
            case "classification": 
                P = (y[:,None]@y[:,None].T)*K                                   # matrix in the quadratic term of cost function
                P += disturbance*np.eye(K.shape[0])                             # Applies a small disturbance to guarantee P is positive definite  
                q = -np.ones(K.shape[0])                                        # vector in the linear term of cost function
                A = y.T                                                         # equality constraint y.T*lambda = 0  --> A*x = b
                b = np.zeros(1)
                match self.margin:
                    case "hard":
                        G = -np.eye(K.shape[0])                                 # inequality constraint -lambda <= 0  --> G <= h
                        h = np.zeros(K.shape[0])
                    case "soft":
                        G = np.vstack((-np.eye(K.shape[0]), np.eye(K.shape[0])))# inequality constraint -lambda <= 0 & lambda <= C  --> Gx <= h
                        h = np.hstack((np.zeros(K.shape[0]), self.C*np.ones(K.shape[0])))
                self.lm = qpsolvers.solve_qp(P, q, G, h, A, b, solver=solver)   # solves the dual problem
                self.sp = np.squeeze(np.where(self.lm > eps_lm))                # gets the indexes of the support vectors
                match self.margin:
                    case "hard":
                        K = self.kernel_fun(x[self.sp], x[self.sp])
                        self.b = np.mean(y[self.sp] - (self.lm[self.sp]*y[self.sp])@K) # calculates the mean bias
                    case "soft":
                        sp1 = np.squeeze(np.where((self.lm > eps_lm) & (self.lm < self.C)))
                        K1 = self.kernel_fun(x[self.sp], x[sp1])
                        self.b = np.mean(y[sp1] - (self.lm[self.sp]*y[self.sp])@K1)
                return self.lm, self.b, self.sp
            
            case "regression":
                Paux = np.hstack((np.eye(K.shape[0]), -np.eye(K.shape[0])))
                P = Paux.T@K@Paux
                P += disturbance*np.eye(2*K.shape[0])
                q = self.eps_r*np.hstack((np.ones(K.shape[0]),np.ones(K.shape[0]))) - np.hstack((y,-y))
                A = np.hstack((np.ones(K.shape[0]), -np.ones(K.shape[0])))
                b = np.zeros(1)
                match self.margin:
                    case "hard":
                        G = -np.eye(2*K.shape[0])                                
                        h = np.zeros(2*K.shape[0])
                    case "soft":
                        G = np.vstack((-np.eye(2*K.shape[0]), np.eye(2*K.shape[0])))        
                        h = np.hstack((np.zeros(2*K.shape[0]), self.C*np.ones(2*K.shape[0])))
                lm = qpsolvers.solve_qp(P, q, G, h, A, b, solver=solver)
                alpha, alpha_star = np.split(lm, 2)                             # collects alpha and alpha_star from the solution
                sp1 = np.squeeze(np.where(alpha > eps_lm))
                sp2 = np.squeeze(np.where(alpha_star > eps_lm))
                self.sp = np.unique(np.hstack((sp1,sp2)))
                self.lm = alpha - alpha_star
                match self.margin:
                    case "hard":
                        K1 = self.kernel_fun(x[self.sp], x[sp1])
                        K2 = self.kernel_fun(x[self.sp], x[sp2])
                        self.b = np.mean(np.hstack(((y[sp1] - self.lm[self.sp]@K1 - self.eps_r), (y[sp2] - self.lm[self.sp]@K2 + self.eps_r))))
                    case "soft":
                        sp1 = np.squeeze(np.where((alpha > eps_lm) & (alpha < self.C)))
                        sp2 = np.squeeze(np.where((alpha_star > eps_lm) & (alpha_star < self.C)))
                        K1 = self.kernel_fun(x[self.sp], x[sp1])
                        K2 = self.kernel_fun(x[self.sp], x[sp2])
                        self.b = np.mean(np.hstack(((y[sp1] - self.lm[self.sp]@K1 - self.eps_r), (y[sp2] - self.lm[self.sp]@K2 + self.eps_r))))
                return self.lm, self.b, self.sp
            
            case "expansion":
                def lin_system(var, *args):
                    beta = var
                    y_aug = args[0]
                    K_aug = args[1]
                    eq = beta@K_aug - y_aug 
                    return eq
                K += disturbance*np.eye(x.shape[0])
                K_aux = np.hstack((K, np.ones(K.shape[0]).reshape(-1,1)))
                K_aug = np.vstack((K_aux, np.ones(K_aux.shape[1])))
                K_aug[-1,-1] = 0.
                y_aug = np.hstack((y, 0.))
                beta = opt.fsolve(lin_system, np.zeros(len(y_aug)), args=(y_aug, K_aug))
                self.lm, self.b, self.sp = beta[:-1], beta[-1], None
                return self.lm, self.b, self.sp
            
            case _:
                raise Exception("Problem type is not defined correctly!")
            

    def svm_predict(self, x_pred):                                              # predicts the class or value of new examples
        x, y = self.train_data[:,:-1], self.train_data[:,-1]
        match x_pred:
            case np.ndarray() as x_pred:
                x_new = x_pred
            case "test":
                x_new = self.test_data[:,:-1]
        if self.normalize:
            x_new = normalization(self.raw_data, x_new)
        match self.problem:
            case "classification":
                K = self.kernel_fun(x[self.sp], x_new)
                y_new = np.sign((self.lm[self.sp]*y[self.sp])@K + self.b)                                     
            case "regression":
                K = self.kernel_fun(x[self.sp], x_new)
                y_new = self.lm[self.sp]@K + self.b
            case "expansion":
                K = self.kernel_fun(x, x_new)
                y_new = self.lm@K + self.b
        if str(x_pred) == "test":
            self.pred_data = np.column_stack((x_new, y_new))                            
        return y_new
        
    def kernel_fun(self, xi, xj):                                               # defines the kernel function
        match self.kernel:
            case "linear":                                                      # linear kernel
                K = xi@xj.T
            case "rbf":                                                         # radial basis function kernel
                K = np.empty((xi.shape[0], xj.shape[0]))
                for i in range(xi.shape[0]):
                    for j in range(xj.shape[0]):
                        K[i,j] = np.exp(-1/(2*self.sigma**2)*
                                        ((xi[i,:]-xj[j,:])@(xi[i,:]-xj[j,:]).T))
            case _:
                raise Exception("Kernel not defined or not available!")
        return K
    
    # r = np.reshape(np.tile(xi,xj.shape[0]) - np.reshape(xj, (1,-1)), (-1,xi.shape[1]))
    # a = np.reshape(np.diag(r@r.T), (xi.shape[0],xj.shape[0]))
    # k = np.exp(-1/(2*self.sigma**2)*a)        
    

           
