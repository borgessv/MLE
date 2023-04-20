"""
Machine Learning in Engineering, University of Twente, 2023

Project: Prediction of von Mises stress distribution on 2D structures under a 
         specific load and boundary conditions. 

Description: This code preprocess the FEM images dataset to be used by the 
             U-NET model. 
"""
import cv2, os, h5py
import numpy as np

def preprocess_fun(dataset, max_stress=np.infty):
    img_dir = os.path.join(os.getcwd(), dataset)                                # Path to the dataset directory
    files = sorted(filter(lambda x: os.path.isfile(os.path.join(img_dir, x)),   # Sort the files in alphabetical order 
                          os.listdir(img_dir)))
    inputs, outputs = [], []                                                    # Initialize list variables
    inputs_filename, outputs_filename = [], []
    
    for idx, filename in enumerate(files):                                      # Prepocess each image of the dataset   
        print('\rPreprocessing dataset: '+str(idx+1)+'/'+str(len(files)), end='')
        if filename.endswith('INP.png') and filename.replace('INP','OUT') in files:
            if float(filename[filename.index('MAX_')+4:filename.index('_INP')]) <= max_stress:
                input_path = os.path.join(img_dir, filename)                    # Get the full path of the input image file
                input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)        # Load the input image and convert to grayscale
                input_preprocessed = preprocess_input(input_img)                # Call the preprocess function
                inputs.append(input_preprocessed)                               # Append the prepocessed input image
                inputs_filename.append(filename)                                # Append the input filename
    
                output_path = os.path.join(img_dir, filename.replace('INP','OUT')) # Get the full path of the output image file
                output_img = cv2.imread(output_path)                            # Load the output image
                output_preprocessed = preprocess_output(output_img)             # Call the preprocess function
                outputs.append(output_preprocessed)                             # Append the prepocessed output image
                outputs_filename.append(filename.replace('INP','OUT'))          # Append the output filename
        elif filename.replace('INP','OUT') not in files:
            pass
        
    data = {'inputs': np.array(inputs), 'outputs': np.array(outputs),           # Collect the dataset
            'inputs_filename': inputs_filename, 'outputs_filename': outputs_filename}

    with h5py.File(dataset+'_preprocessed.hdf5', 'w') as f:                    # Save dataset to the file       
        for key, value in data.items():
            f.create_dataset(key, data=value, compression='gzip')
    return
    
# scale_img = 2
def preprocess_input(img):                                                      # Preprocess the input images
    inp_img = np.array(img)/255.                                                # Normalize pixel values [0,1]
    # inp_img = cv2.resize(inp_img, tuple(scale_img*x for x in (img.shape[1],   # Scale image (if needed)
    #                                                           img.shape[0])))  
    inp_img = np.expand_dims(inp_img, axis=-1)                                  # Add extra dimension (grayscale channel)    
    return inp_img

def preprocess_output(img):                                                     # Preprocess the output images
    out_img = np.array(img)/255.                                                # Normalize pixel values [0,1]
    # out_img = cv2.resize(out_img, tuple(scale_img*x for x in (img.shape[1],   # Scale image (if needed)
    #                                                           img.shape[0]))) 
    return out_img