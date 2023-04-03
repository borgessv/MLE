"""
Course: Machine Learning in Engineering
Assignment 3
Author: Vitor Borges Santos
Date: 27-03-2023

Description: creation of SVM and MLP classes. 
"""
import numpy as np
import scipy.optimize as opt
import qpsolvers
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, SimpleRNN
from utils import create_2Ddataset, normalization


class NN:
    def __init__(self, dataset, train_split, pred_data, normalize, shuffle, problem, 
                 hidden_dim, activation, optimizer, learning_rate, batch_size, 
                 epochs, seed=None, validation_split=0.5, time_steps=None):
        match dataset:                                                          # loads or creates dataset
            case "random":
                self.raw_data = create_2Ddataset(seed=seed)                                    
            case _:
                self.raw_data = np.loadtxt(dataset+".txt", skiprows=1) 
        self.train_split = train_split
        self.pred_data = pred_data
        self.validation_split = validation_split
        self.normalize = normalize
        self.shuffle = shuffle
        self.problem = problem
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
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
            if raw_classes[0] != 1. or raw_classes[0] != 0.:                   # makes sure the classes are +1 and -1
                if raw_classes[1] != 1. :
                    y[np.isin(y, raw_classes[0])] = 1.
                else:
                    y[np.isin(y, raw_classes[0])] = 0.
            if raw_classes[1] != 1. or raw_classes[1] != 0.:
                if raw_classes[0] != 1. :
                    y[np.isin(y, raw_classes[1])] = 1.
                else:
                    y[np.isin(y, raw_classes[1])] = 0.
            self.raw_data = np.column_stack((x_raw, y))
            
        if self.normalize:                                                      # normalizes features (if applicable)
            x_data = normalization(x_raw, self.raw_data[:,:-1])
            if self.problem != "classification":
                y_data = normalization(y_raw, self.raw_data[:,-1])
            else:
                y_data = self.raw_data[:,-1]
        else:
            x_data = self.raw_data[:,:-1]
            y_data = self.raw_data[:,-1]
        self.data = np.column_stack((x_data, y_data))
        self.train_data = self.data[:int(np.ceil(self.train_split*self.data.shape[0])), :]
        if str(self.pred_data) == "test":
            rem_data = self.data[int(np.ceil(self.train_split*self.data.shape[0])):, :]
            self.validation_data = rem_data[:int(np.ceil(self.validation_split*rem_data.shape[0])), :]
            self.test_data = rem_data[int(np.ceil(self.validation_split*rem_data.shape[0])):, :]
        else:
            self.test_data = self.data[int(np.ceil(self.train_split*self.data.shape[0])):, :]
            
        if self.problem != "classification":
            self.trainX, self.trainY = self.reshape_data(self.train_data[:,-1], self.time_steps)
            self.testX, self.testY = self.reshape_data(self.test_data[:,-1], self.time_steps)
            if str(self.pred_data) == "test":
                self.valX, self.valY = self.reshape_data(self.validation_data[:,-1], self.time_steps)
        return

    def create(self):
        keras.backend.clear_session()
        match self.problem:
            case "classification":
                self.model = keras.Sequential()
                self.model.add(Input(shape=self.train_data[0,:-1].shape))
                for i in range(len(self.hidden_dim)):
                    self.model.add(Dense(self.hidden_dim[i], activation=self.activation[i]))
                self.model.add(Dense(1))
                self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                                   optimizer=self.optimizer, metrics=['accuracy'])
                self.model.optimizer.lr.assign(self.learning_rate)
            case _:
                self.model = keras.Sequential()
                self.model.add(Input(shape=(self.time_steps,1)))
                for i in range(len(self.hidden_dim)):
                    if i != len(self.hidden_dim)-1:
                        self.model.add(SimpleRNN(self.hidden_dim[i], 
                                activation=self.activation[i], return_sequences=True))
                    else:
                        self.model.add(SimpleRNN(self.hidden_dim[i], 
                                activation=self.activation[i]))
                self.model.add(Dense(1))
                self.model.compile(loss='mse', optimizer=self.optimizer)
                self.model.optimizer.lr.assign(self.learning_rate)
        return 
       
    def train(self):
        match self.problem:
            case "classification":
                hist = self.model.fit(self.train_data[:,:-1], self.train_data[:,-1].astype(int), 
                               validation_data=(self.validation_data[:,:-1], 
                                                self.validation_data[:,-1].astype(int)), 
                               batch_size=self.batch_size, epochs=self.epochs)
                return hist.history['loss'][-1], hist.history['accuracy'][-1], 
                hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]
            case _:  
                hist = self.model.fit(self.trainX, self.trainY, validation_data=(self.valX, self.valY), 
                               batch_size=self.batch_size, epochs=self.epochs)
        return hist.history['loss'][-1], [], hist.history['val_loss'][-1], []
       
    def predict(self, x_pred):
        if not str(x_pred) == "test" and self.normalize:
            if self.problem == "regression":
                x_pred = normalization(self.raw_data[:,-1], x_pred)
            else:
                x_pred = normalization(self.raw_data[:,:-1], x_pred)
        elif not str(x_pred) == "test" and not self.normalize:
            x_pred = x_pred
        else:
            if self.test_data.size == 0:
                return []
            else:
                x_pred = self.test_data[:,:-1]
        match self.problem:
            case "classification":
                y_pred = self.model.predict(x_pred,verbose=0)
                y_pred[y_pred <= 0.5] = -1.
                y_pred[y_pred > 0.5] = 1.
            case _:
                if not str(x_pred) == "test":
                    predX = x_pred.tolist()
                    for i in range(0, len(predX)-self.time_steps):
                        if i==0:
                            x_pred = predX[i:i+self.time_steps]
                        else: 
                            x_pred = np.append(x_pred, predX[i:i+self.time_steps])
                    x_pred = np.reshape(x_pred, (int(np.ceil(len(x_pred)/self.time_steps)), 
                                                 self.time_steps, 1))
                else:
                    x_pred = self.testY
                y_pred = self.model.predict(x_pred, verbose=0)
                x_pred = np.array(range(0, len(y_pred)))
        self.pred_data = np.column_stack((x_pred, np.float64(y_pred.flatten()))) 
        return np.float64(y_pred.flatten())
     
    def reshape_data(self, data, step):
        y_idx = np.arange(step, len(data), step)                                # array with indices for output elements based on the time-step 
        y_tmp = data[y_idx]
        x_tmp = data[range(step*len(y_tmp))]                                    # enforces the input array to stop with the last output
        x_tmp = np.reshape(x_tmp, (len(y_tmp), step, 1))                        # reshapes the input to the shape (samples x step x 1)
        return x_tmp, y_tmp      


class SVM:
    def __init__(self, dataset, train_split, pred_data, normalize, shuffle, problem, margin, 
                 kernel, seed=None, C=None, sigma=None, eps_r=None, validation_split=0.5):
        match dataset:                                                          # loads or creates dataset
            case "random":
                self.raw_data = create_2Ddataset(seed=seed)                                    
            case _:
                self.raw_data = np.loadtxt(dataset+".txt", skiprows=1) 
        self.train_split = train_split
        self.pred_data = pred_data
        self.validation_split = validation_split
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
            
        if self.normalize:                                                      # normalizes features (if applicable)
            x_data = normalization(x_raw, self.raw_data[:,:-1])
            if self.problem != "classification":
                y_data = normalization(y_raw, self.raw_data[:,-1])
            else:
                y_data = self.raw_data[:,-1]
        else:
            x_data = self.raw_data[:,:-1]
            y_data = self.raw_data[:,-1]
        self.data = np.column_stack((x_data, y_data))
        self.train_data = self.data[:int(np.ceil(self.train_split*self.data.shape[0])), :]
        if str(self.pred_data) == "test":
            rem_data = self.data[int(np.ceil(self.train_split*self.data.shape[0])):, :]
            self.validation_data = rem_data[:int(np.ceil(self.validation_split*rem_data.shape[0])), :]
            self.test_data = rem_data[int(np.ceil(self.validation_split*rem_data.shape[0])):, :]
        else:
            self.test_data = self.data[int(np.ceil(self.train_split*self.data.shape[0])):, :]                
        return
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
            

    def predict(self, x_pred):                                              # predicts the class or value of new examples
        x, y = self.train_data[:,:-1], self.train_data[:,-1]
        if not str(x_pred) == "test" and self.normalize:
            x_new = normalization(self.raw_data[:,:-1], x_pred)
        elif not str(x_pred) == "test" and not self.normalize:
            x_new = x_pred
        else:
            if self.test_data.size == 0:
                return []
            x_new = np.vstack((self.validation_data[:,:-1],self.test_data[:,:-1]))
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
    

    # predX = reshape_data(x_pred, self.time_steps)[0]
    # x_input = predX[0:1]
    # y_pred = []
    # for i in range(len(x_pred)-self.time_steps):   
    #     y_pred.append(list(self.model.predict(x_input,batch_size=self.batch_size,verbose=0)[0])) # Generate prediction and add it to the list
    #     x_input = np.append(x_input[:,1:,:],[[y_pred[i]]],axis=1) # Drop oldest and append latest prediction
    # y_pred = np.asarray(y_pred)        
