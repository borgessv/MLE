"""
Machine Learning in Engineering, University of Twente, 2023

Project: Prediction of von Mises stress distribution on 2D structures under a 
         specific load and boundary conditions. 
Dataset source: https://data.mendeley.com/datasets/wzbzznk8z3/2

Description: Employment of a U-NET model to predict von Mises stress 
             distribution of 2D structures of arbitrary shape subjected to 
             gravity and clamped at the bottom end. 
"""
import os, cv2, h5py, time, random, keras
import numpy as np
from datetime import datetime
import keras.optimizers as opt
from preprocess import preprocess_fun
from unet import UNET
from utils import plot_loss, PlotCallback

################################# USER INPUTS #################################                                                   
dataset = '100k_Max300MPa'                                                      # Dataset directory  
max_stress = np.infty                                                           # Maximum stress of the samples (set to np.infty to get all samples from the dataset)
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15                            # Sets' ratios
get_model = 'model_100k_Max300MPa_exp3_train0.7_lr0.0001_ep100_batch32.h5'      # 'create' or '.h5' file  
exp0 = 3                                                                        # Initial power of two defining the number of filters of the convolutional layers  
lr = 1e-4                                                                       # Learning rate
loss = 'mse'                                                                    # Loss function
epochs = 100                                                                    # Number of epochs
batch_size = 32                                                                 # Batch size
callback = True                                                                 # To call the callback funciton during training
###############################################################################

#%% Preprocess:
if not os.path.exists(dataset+'_preprocessed.hdf5'):
    preprocess_fun(dataset, max_stress)                                         # Preprocess dataset
        
with h5py.File(dataset+'_preprocessed.hdf5', 'r') as f:                         # Load preprocessed dataset 
    inputs = f['inputs'][()]
    outputs = f['outputs'][()]
    inputs_filename = f['inputs_filename'][()].astype('str')
    outputs_filename = f['outputs_filename'][()].astype('str')
                                 
num_train = int(inputs.shape[0]*train_ratio)                                    # Split dataset into training, validation, and test sets
num_val = int(inputs.shape[0]*val_ratio)
num_test = inputs.shape[0] - num_train - num_val
train_inputs, train_outputs = inputs[:num_train], outputs[:num_train]
val_inputs, val_outputs = inputs[num_train:num_train+num_val], outputs[num_train:num_train+num_val]
test_inputs, test_outputs = inputs[-num_test:], outputs[-num_test:]
test_inp_filename, test_out_filename = inputs_filename[-num_test:], outputs_filename[-num_test:]

#%% Create or load the NN model:
if get_model == 'create':        
    model = UNET(exp0, train_inputs.shape[1:]).create()                         # Create model
    model.compile(optimizer=opt.Adam(learning_rate=lr), loss=loss,              # Compile model
                  metrics=[])          
    callback_fun = PlotCallback(test_inputs[2], epochs, model) if callback else []
    tic = time.time()
    history = model.fit(train_inputs, train_outputs,                            # Train model
                        validation_data=(val_inputs, val_outputs), 
                        epochs=epochs, batch_size=batch_size, 
                        callbacks=[callback_fun])
    toc = time.time() - tic
    print(f'Training time: {toc} s')
    plot_loss(history)
    test_loss = model.evaluate(test_inputs, test_outputs, verbose=0)            # Evaluate model on test set
    print(f'Test loss: {test_loss}')
    model.save('model_'+dataset+'_exp'+str(exp0)+'_train'+str(train_ratio)+     # Save trained model
               '_lr'+str(lr)+'_ep'+str(epochs)+'_batch'+str(batch_size)+'.h5')
else:
    model = keras.models.load_model(get_model)                                  # Load trained model

#%% Prediction:
path_sol = 'test_solutions_'+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')       # Create directory to save the solutions
os.mkdir(path_sol)
test_samples = random.sample(range(len(test_inputs)), 100)                      # Randomly select samples from the test set
for sample in test_samples:
    input_img = np.expand_dims(test_inputs[sample],axis=0)                      # Input image
    output_img = test_outputs[sample]                                           # True output image    
    pred_img = model.predict(input_img, verbose=0)                              # Predicted image 

    output_img = (output_img.squeeze()*255).astype(np.uint8)                    # Convert pixel values back to integer range [0, 255]
    pred_img = (pred_img.squeeze()*255).astype(np.uint8)                     
    cv2.imwrite(os.path.join(path_sol, test_out_filename[sample]), output_img)  # Save images as .png file
    cv2.imwrite(os.path.join(path_sol, test_out_filename[sample].replace('OUT', 'PRD')), pred_img)

    # input_img = input_img.squeeze()                                           # Save input image
    # input_img = (input_img*255).astype(np.uint8)
    # cv2.imwrite(os.path.join(path_sol, inputs_filename[sample]), input_img)
    
