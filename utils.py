"""
Course: Machine Learning in Engineering
Assignment 3
Author: Vitor Borges Santos
Date: 27-02-2023

Description: creation of a auxiliary functions to be used by models.py
             and main.py files.  
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
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
    

def plot_fun(models, plot_filename, image_format, feature_name, class_name, n_grid=500):
    model = models[0]
    if model.problem == "classification":
        model.train_data[model.train_data[:,-1] == 0, -1] = -1
        model.test_data[model.test_data[:,-1] == 0, -1] = -1
        model.validation_data[model.validation_data[:,-1] == 0, -1] = -1
        c1_data_train = model.train_data[model.train_data[:,-1] == 1, :-1]     # class 1 - train examples  
        c2_data_train =  model.train_data[model.train_data[:, -1] == -1, :-1]
        c1_data_test = model.test_data[model.test_data[:,-1] == 1, :-1]     # class 1 - train examples  
        c2_data_test =  model.test_data[model.test_data[:, -1] == -1, :-1]
        c1_data_validation = model.validation_data[model.validation_data[:,-1] == 1, :-1]     # class 1 - train examples  
        c2_data_validation =  model.validation_data[model.validation_data[:, -1] == -1, :-1]    
        if model.normalize:                                                 # denormalizes points to be plotted (if applicable)
            c1_data_train = denormalization(model.raw_data, c1_data_train)                                        
            c2_data_train = denormalization(model.raw_data, c2_data_train)                                   
            c1_data_test = denormalization(model.raw_data, c1_data_test)                                                   
            c2_data_test = denormalization(model.raw_data, c2_data_test)
            c1_data_validation = denormalization(model.raw_data, c1_data_validation)                                                   
            c2_data_validation = denormalization(model.raw_data, c2_data_validation)
        if model.train_data[:,:-1].shape[1] == 2:
            fig, ax = plt.subplots()
            ax.plot(c1_data_train[:,0], c1_data_train[:,1],'o', color='tab:blue', 
                      alpha=0.3, label=class_name[0] + " - train")
            ax.plot(c2_data_train[:,0], c2_data_train[:,1],'o', color='tab:orange', 
                      alpha=0.3, label=class_name[1] + " - train")
            if c1_data_validation.size != 0:
                ax.plot(c1_data_validation[:,0], c1_data_validation[:, 1],'o', color='tab:blue', 
                        label=class_name[0] + " - validation")
            if c2_data_validation.size != 0:
                ax.plot(c2_data_validation[:,0], c2_data_validation[:,1],'o', color='tab:orange', 
                        label=class_name[1] + " - validation")
            if c1_data_test.size != 0:
                ax.plot(c1_data_test[:,0], c1_data_test[:, 1],'x', color='c', 
                        label=class_name[0] + " - test")
            if c2_data_test.size != 0:
                ax.plot(c2_data_test[:,0], c2_data_test[:,1],'x', color='m', 
                        label=class_name[1] + " - test")
            cmap = pl.cm.Dark2(np.linspace(1,0,len(models)))  
            for idx, model in enumerate(models):
                x1 = np.linspace(min(model.raw_data[:,0])*(1 - 0.1*np.sign(min(model.raw_data[:,0]))),  
                          max(model.raw_data[:,0])*(1 + 0.1*np.sign(max(model.raw_data[:,0]))), n_grid)
                x2 = np.linspace(min(model.raw_data[:,1])*(1 - 0.1*np.sign(min(model.raw_data[:,1]))),
                          max(model.raw_data[:,1])*(1 + 0.1*np.sign(max(model.raw_data[:,1]))), n_grid)
                x1_mesh, x2_mesh = np.meshgrid(x1, x2)
                x_mesh = np.column_stack((x1_mesh.T.ravel(), x2_mesh.T.ravel()))
                y_mesh = model.predict(x_mesh)
                y_mesh = np.reshape(y_mesh, x1_mesh.shape).T
                cntr = ax.contour(x1_mesh, x2_mesh, y_mesh, levels=1, colors="w", alpha=0)
                db = cntr.allsegs[0][:]
                for i in range(len(db)):
                    if i == 0:
                        ax.plot(db[i][:,0],db[i][:,1],'-', color=cmap[idx],label=str(model)[8:11]+" - hyperplane")
                    else:
                        ax.plot(db[i][:,0],db[i][:,1],'-k',label='_nolegend_')    
            x_lim = [min(model.raw_data[:,0])*(1 - 0.05*np.sign(min(model.raw_data[:,0]))),  
                      max(model.raw_data[:,0])*(1 + 0.05*np.sign(max(model.raw_data[:,0])))]
            y_lim = [min(model.raw_data[:,1])*(1 - 0.05*np.sign(min(model.raw_data[:,1]))),
                      max(model.raw_data[:,1])*(1 + 0.05*np.sign(max(model.raw_data[:,1])))]
            plt.xlabel(feature_name[0],fontsize=15)
            plt.ylabel(feature_name[1],fontsize=15)
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize=13)
            filename = plot_filename + "_" + str(int(model.train_split*100)) 
            plt.savefig(filename + "." + image_format, transparent=True, dpi=1200, bbox_inches='tight')
            plt.show()
            
    else:
        if model.train_data[:,:-1].shape[1] == 2 or model.train_data[:,:-1].shape[1] == 1:
            x_train = model.train_data[:, :-1]     # class 1 - train examples  
            y_train =  model.train_data[:, -1]
            x_test = model.test_data[:, :-1]     # class 1 - train examples  
            y_test =  model.test_data[:, -1]
            x_validation = model.validation_data[:, :-1]     # class 1 - train examples  
            y_validation =  model.validation_data[:, -1]    
            if model.normalize:                                                 # denormalizes points to be plotted (if applicable)
                x_train = denormalization(model.raw_data, x_train)                                        
                y_train = denormalization(model.raw_data[:,-1], y_train)                                   
                x_test = denormalization(model.raw_data, x_test)                                                   
                y_test = denormalization(model.raw_data[:,-1], y_test)
                x_validation = denormalization(model.raw_data, x_validation)                                                   
                y_validation = denormalization(model.raw_data[:,-1], y_validation)
            if model.train_data[:,:-1].shape[1] == 1:
                fig, ax = plt.subplots()
                ax.plot(x_train, y_train,'x', color='tab:blue', alpha=0.5, label=r"train")
                ax.plot(x_validation, y_validation,'x', color='tab:orange', alpha=0.5, label=r"validation")
                ax.plot(x_test, y_test,'x', color='m', alpha=0.5, label=r"test")
                cmap = pl.cm.turbo(np.linspace(1,0,len(models))) 
                for idx, model in enumerate(models):    
                    x_mesh = np.vstack((x_train, x_validation, x_test))
                    if str(model)[8:11] == "NN ":
                        y_mesh = model.predict(np.concatenate((y_train, y_validation, y_test)))
                    else:
                        y_mesh = model.predict(x_mesh)
                    y_mesh = y_mesh.reshape(-1,1)
                    if model.normalize:
                        y_mesh = denormalization(model.raw_data[:,-1], y_mesh)
                    if str(model)[8:11] == "NN ":
                        y_mesh = np.append(np.zeros(model.time_steps),y_mesh)
                    ax.plot(x_mesh,y_mesh, '.', markersize=2, color=cmap[idx], label=str(model)[8:11]+" - regression")
                x_lim = [min(model.raw_data[:,0]), max(model.raw_data[:,0])]
                y_lim = [min(model.raw_data[:,-1]), max(model.raw_data[:,-1])]
                plt.xlabel(feature_name[0], fontsize=13)
                plt.ylabel(feature_name[1], fontsize=13)               
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize=13)
                filename = plot_filename + "_" + str(int(model.train_split*100)) + "."
                plt.savefig(filename + image_format, transparent=True, dpi=1200, bbox_inches='tight')
                plt.show()
            else:
                pass
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
    