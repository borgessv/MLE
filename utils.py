"""
Course: Machine Learning in Engineering
Assignment 1: Support Vector Machines (SVM)
Author: Vitor Borges Santos
Date: 16-02-2023

Description: creation of a auxiliary functions to be used by the class LinearSVM
             and by the main.py file.  
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from IPython import get_ipython                                                 # forces plots to be shown in plot pane
ipython = get_ipython()
ipython.magic('matplotlib inline')
plt.rcParams['text.usetex'] = True                                              # uses latex style in plots

def create_2Ddataset(n=500, c1_ratio=0.5, mean=1, cov_factor=0.1, seed=None):
    np.random.seed(seed)
        
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

def normalization(data, x_denorm):
    if data.ndim < 2:                                                           # convert 1d array to 2d 
        data = data[:,None] 
    if x_denorm.ndim < 2:                                                       
        x_denorm = x_denorm[:,None]                                           
    x_norm = np.zeros(np.shape(x_denorm))
    for i in range(np.shape(x_denorm)[1]):
        x_norm[:,i] = (x_denorm[:,i] - np.min(data[:,i]))/np.ptp(data[:,i])
    return x_norm

def denormalization(data, x_norm):
    if data.ndim < 2:                                                           # convert 1d array to 2d 
        data = data[:,None] 
    if x_norm.ndim < 2:                                                          
        x_norm = x_norm[:,None] 
    x_denorm = np.zeros(np.shape(x_norm))
    for i in range(np.shape(x_norm)[1]):
        x_denorm[:,i] = x_norm[:,i]*np.ptp(data[:,i]) + np.min(data[:,i])
    return x_denorm
    
def plot_fun(model, plot_filename, image_format, feature_name, class_name, n_grid=500):
    x_pred, y_pred = model.pred_data[:,:-1], model.pred_data[:,-1]
    if model.kernel == "linear":
        n_grid = 4*n_grid
    if model.problem == "classification":
        if model.train_data[:,:-1].shape[1] == 2 or model.train_data[:,:-1].shape[1] == 1:
            c1_data_train = model.train_data[model.train_data[:, -1] == 1, :-1]     # class 1 - train examples  
            c2_data_train =  model.train_data[model.train_data[:, -1] == -1, :-1]   # class 2 - train examples 
            if model.test_data.size != 0:
                y_true = model.test_data[:,-1]
                c1_data_test =  x_pred[y_true == 1]                                 # class 1 - test examples
                c2_data_test = x_pred[y_true == -1]                                 # class 2 - test examples
                val_y = y_true - y_pred                                             # compares predicted and true classes
                false_c1 = np.where(val_y == 2)                                     # indexes of false class 1 prediction
                false_c2 = np.where(val_y == -2)                                    # indexes of false class 2 prediction
                x_test_fc1 = x_pred[false_c1]                                       # class 2 - misclassified examples
                x_test_fc2 = x_pred[false_c2]                                       # class 1 - misclassified examples
            else:
                c1_data_test =  x_pred[y_pred == 1]
                c2_data_test =  x_pred[y_pred == -1]
                x_test_fc1 = x_pred[[]]                                                     
                x_test_fc2 = x_pred[[]]
            if model.train_data[:,:-1].shape[1] == 2:
                x1 = np.linspace(min(model.raw_data[:,0])*(1 - 0.1*np.sign(min(model.raw_data[:,0]))),  
                          max(model.raw_data[:,0])*(1 + 0.1*np.sign(max(model.raw_data[:,0]))), n_grid)
                x2 = np.linspace(min(model.raw_data[:,1])*(1 - 0.1*np.sign(min(model.raw_data[:,1]))),
                          max(model.raw_data[:,1])*(1 + 0.1*np.sign(max(model.raw_data[:,1]))), n_grid)
                x1_mesh, x2_mesh = np.meshgrid(x1, x2)
                x_mesh = np.column_stack((x1_mesh.T.ravel(), x2_mesh.T.ravel()))
                y_mesh = model.svm_predict(x_mesh)
                y_mesh = np.reshape(y_mesh, x1_mesh.shape).T
            elif model.train_data[:,:-1].shape[1] == 1:
                x_mesh = np.reshape(np.linspace(min(model.raw_data[:,0]),max(model.raw_data[:,0]),n_grid), (-1,1)) 
                y_mesh = model.svm_predict(x_mesh)
                y_mesh = np.reshape(y_mesh, x_mesh.shape).T
                
            if model.normalize:                                                 # denormalizes points to be plotted (if applicable)
                c1_data_train = denormalization(model.raw_data, c1_data_train)                                        
                c2_data_train = denormalization(model.raw_data, c2_data_train)                                   
                c1_data_test = denormalization(model.raw_data, c1_data_test)                                                   
                c2_data_test = denormalization(model.raw_data, c2_data_test)
                if x_test_fc1.size != 0 | x_test_fc2.size != 0:
                    x_test_fc1 = denormalization(model.raw_data, x_test_fc1)                    
                    x_test_fc2 = denormalization(model.raw_data, x_test_fc2)                    
    
            
            if model.train_data[:,:-1].shape[1] == 2:
                fig, ax = plt.subplots()
                cntrf = ax.contourf(x1_mesh, x2_mesh, y_mesh, levels=0, colors=('r', 'b'),
                            alpha=0.15)
                ax.plot(c1_data_train[:,0], c1_data_train[:,1],'x', color='tab:blue', 
                          alpha=1, label=class_name[0] + " - train")
                ax.plot(c2_data_train[:,0], c2_data_train[:,1],'x', color='tab:orange', 
                          alpha=1, label=class_name[1] + " - train")
                if c1_data_test.size != 0:
                    ax.plot(c1_data_test[:,0], c1_data_test[:, 1],'o', color='tab:blue', 
                            label=class_name[0] + " - test")
                if c2_data_test.size != 0:
                    ax.plot(c2_data_test[:,0], c2_data_test[:,1],'o', color='tab:orange', 
                            label=class_name[1] + " - test")
                if x_test_fc2.size > 0:
                    ax.plot(x_test_fc2[:,0], x_test_fc2[:,1],'ob', 
                              label=class_name[0] + " - false")
                if x_test_fc1.size > 0:
                    ax.plot(x_test_fc1[:,0], x_test_fc1[:,1], 'or', 
                              label=class_name[1] + " - false")
                cntr = ax.contour(x1_mesh, x2_mesh, y_mesh, levels=1, colors="w", alpha=0)
                db = cntr.allsegs[0][:]
                for i in range(len(db)):
                    if i == 0:
                        ax.plot(db[i][:,0],db[i][:,1],'-k',label="decision boundary")
                    else:
                        ax.plot(db[i][:,0],db[i][:,1],'-k',label='_nolegend_')
                x_lim = [min(model.raw_data[:,0])*(1 - 0.1*np.sign(min(model.raw_data[:,0]))),  
                          max(model.raw_data[:,0])*(1 + 0.1*np.sign(max(model.raw_data[:,0])))]
                y_lim = [min(model.raw_data[:,1])*(1 - 0.1*np.sign(min(model.raw_data[:,1]))),
                          max(model.raw_data[:,1])*(1 + 0.1*np.sign(max(model.raw_data[:,1])))]
                plt.xlabel(feature_name[0],fontsize=15)
                plt.ylabel(feature_name[1],fontsize=15)
                cbar = fig.colorbar(cntrf, ticks=[1, -1], aspect=5, 
                                    shrink=0.33, anchor=(-0.3,0.))
                cbar.ax.set_yticklabels([class_name[0], class_name[1]], 
                                        fontsize=13)
                
            elif model.train_data[:,:-1].shape[1] == 1:
                fig, ax = plt.subplots(figsize=(6.4,2.5))
                y_aux1 = np.ones(x_mesh.shape)
                y_aux2 = np.ones(x_mesh.shape)
                y_aux1[y_mesh[0]==-1] = np.nan
                y_aux2[y_mesh[0]==1] = np.nan
                ax.fill_between(np.squeeze(x_mesh),np.squeeze(y_aux1), np.squeeze(-y_aux1),
                                color='tab:blue', alpha=0.2, label=class_name[0])
                ax.fill_between(np.squeeze(x_mesh),np.squeeze(y_aux2),np.squeeze(-y_aux2),
                                color='tab:orange', alpha=0.2, label=class_name[1])
                ax.plot(c1_data_train[:,0], np.zeros(c1_data_train[:,0].shape),'x', color='tab:blue',
                          alpha=1, label=class_name[0] + " - train")
                ax.plot(c2_data_train[:,0], np.zeros(c2_data_train[:,0].shape),'x', color='tab:orange',
                          alpha=1, label=class_name[1] + " - train")
                if y_pred.size != 0:
                    ax.plot(c1_data_test[:,0], np.zeros(c1_data_test[:,0].shape),'o', color='tab:blue', 
                            label=class_name[0] + " - test")
                    ax.plot(c2_data_test[:,0], np.zeros(c2_data_test[:,0].shape),'o', color='tab:orange', 
                            label=class_name[1] + " - test")
                if x_test_fc2.size > 0:
                    ax.plot(x_test_fc2[:,0], np.zeros(x_test_fc2[:,0].shape),'ob', 
                              label=class_name[0] + " - false")
                if x_test_fc1.size > 0:
                    ax.plot(x_test_fc1[:,0], np.zeros(x_test_fc1[:,0].shape), 'or', 
                              label=class_name[1] + " - false")
                x_lim = [min(model.raw_data[:,0]), max(model.raw_data[:,0])]
                y_lim = [-1, 1]
                plt.xlabel(feature_name[0], fontsize=15)            
                ax.set(yticklabels=[])
                ax.tick_params(left=False)   
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend( loc='upper left', ncol=1, fontsize=13)
            filename = plot_filename + "_" + model.margin + "_" + model.kernel + "_" + str(int(model.train_split*100)) 
            plt.savefig(filename + "." + image_format, transparent=True, dpi=1200, bbox_inches='tight')
            plt.show()
            
    else:
        if model.train_data[:,:-1].shape[1] == 2 or model.train_data[:,:-1].shape[1] == 1:
            x_train, y_train = model.train_data[:, :-1], model.train_data[:,-1] 
            if model.test_data.size != 0:
                if model.normalize:
                    x_test = normalization(model.raw_data, model.test_data[:,:-1])
                    y_test = normalization(model.raw_data[:,-1], model.test_data[:,-1])
                else:
                    x_test, y_test = model.test_data[:,:-1], model.test_data[:,-1]
            else:
                x_test, y_test = x_pred, y_pred
            if model.train_data[:,:-1].shape[1] == 1:
                x_mesh = np.vstack((x_train, x_test))
                if model.normalize:
                    x_mesh = denormalization(model.raw_data, x_mesh)
                y_mesh = model.svm_predict(x_mesh)
                y_mesh = y_mesh.reshape(-1,1)
            if model.train_data[:,:-1].shape[1] == 2:
                pass

            if model.normalize:                                                
                x_train = denormalization(model.raw_data, x_train)                                        
                y_train = denormalization(model.raw_data[:,-1], y_train) 
                x_test = denormalization(model.raw_data, x_test)                                        
                y_test = denormalization(model.raw_data[:,-1], y_test)                                                                                   
                y_mesh = denormalization(model.raw_data[:,-1], y_mesh)

            if model.train_data[:,:-1].shape[1] == 1:
                fig, ax = plt.subplots()
                ax.plot(x_train, y_train,'x', color='tab:blue', 
                          label=r"train")
                if y_pred.size != 0:
                    ax.plot(x_test, y_test,'x', color='tab:orange', 
                            label=r"test")
                ax.plot(x_mesh,y_mesh, '-k', label=r"SVM regression")
                x_lim = [min(model.raw_data[:,0]), max(model.raw_data[:,0])]
                y_lim = [min(model.raw_data[:,-1]), max(model.raw_data[:,-1])]
                plt.xlabel(feature_name[0], fontsize=13)
                plt.ylabel(feature_name[1], fontsize=13)               
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize=13)
            filename = plot_filename + "_" + model.margin + "_" + model.kernel + "_" + str(int(model.train_split*100)) + "."
            plt.savefig(filename + image_format, transparent=True, dpi=1200, bbox_inches='tight')
            plt.show()
        return

def plot_cm(model, plot_filename, image_format, class_name):
    if model.test_data.size != 0:
        y_pred, y_true = model.pred_data[:,-1], model.test_data[:,-1]
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=class_name)
        cm_display.plot()
        filename = plot_filename + "_CM_" + model.margin + "_" + model.kernel + "_" + str(int(model.train_split*100)) + "."
        plt.savefig(filename + image_format, transparent=True, dpi=1200)
        plt.show() 
    