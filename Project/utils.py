"""
Machine Learning in Engineering, University of Twente, 2023

Project: Prediction of von Mises stress distribution on 2D structures under a 
         specific load and boundary conditions. 

Description: Auxiliary plot and callback functions 
"""
import os, cv2, keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from IPython import get_ipython                                                 
from IPython.display import clear_output
ipython = get_ipython()
ipython.magic('matplotlib inline')

def plot_loss(history):                                                         # Plot train and validation losses
    fig, ax = plt.subplots()
    ax.plot(range(1,len(history.history['loss'])+1),history.history['loss'], 
            'b', label='train')
    ax.plot(range(1,len(history.history['loss'])+1),history.history['val_loss'], 
            'r', label='validation')
    ax.set_yscale('log')
    ax.grid(which='both')
    plt.xlabel('epoch')
    plt.ylabel('loss (MSE)')
    plt.legend()
    plt.savefig('loss_plot_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')+'.png',
                dpi=300, bbox_inches='tight')
    return


class PlotCallback(keras.callbacks.Callback):                                   # Callback function to store losses at each epoch and create gif of the train evolution including a test example
    def __init__(self, test_img, epochs, model):
        self.test = test_img
        self.model = model
        self.epochs = epochs
        
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        self.train_img_folder = 'train_img_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
        self.img2gif_folder = 'img2gif_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
        os.mkdir(self.train_img_folder)
        os.mkdir(self.img2gif_folder)
        self.frames = []
            
    def on_epoch_end(self, epoch, logs={}):
        pred_img = self.model.predict(np.expand_dims(self.test,axis=0), verbose=0)
        pred_img = pred_img.squeeze()
        pred_img = pred_img*255
        cv2.imwrite(os.path.join(self.train_img_folder, str(epoch)+'.png'), pred_img.astype(np.uint8))
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        metrics = ['loss']
        fig, ax = plt.subplots()
        clear_output(wait=True)
        for i, metric in enumerate(metrics):
            ax.plot(range(1, epoch + 2), self.metrics[metric], 'b', label='train')
            if logs['val_' + metric]:
                ax.plot(range(1, epoch + 2), self.metrics['val_' + metric], 'r', label='validation')
            ax.set_yscale('log')    
            ax.legend(loc='upper left')
            ax.grid(which='both')
        plt.title('epoch: %i' %(epoch+1), loc='right', weight='bold')
        plt.xlabel('epoch')
        plt.ylabel('loss (MSE)')
        plt.xlim([0, self.epochs])
        plt.tight_layout()
        attach_img = plt.imread(os.path.join(self.train_img_folder, str(epoch)+'.png'))
        ax2 = fig.add_axes([0.55,0.5,0.4,0.4], anchor='NE', zorder=1)
        ax2.imshow(attach_img)
        ax2.axis('off')
        plt.savefig(os.path.join(self.img2gif_folder, 'epoch'+str(epoch)), dpi=300, bbox_inches='tight') 
        plt.close()
        
        new_frame = Image.open(os.path.join(self.img2gif_folder, 'epoch'+str(epoch))+'.png')
        self.frames.append(new_frame)
        
    def on_train_end(self, logs={}):
        self.frames[0].save('train_evolution_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')+'gif', 
                            format='GIF', append_images=self.frames[1:], save_all=True, 
                            duration=100, loop=0)
        
        