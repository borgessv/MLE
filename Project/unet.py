"""
Machine Learning in Engineering, University of Twente, 2023

Project: Prediction of von Mises stress distribution on 2D structures under a 
         specific load and boundary conditions. 

Description: Implementation of a simple U-NET architecture. Inspired by:
https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
"""
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate

class UNET:
    def __init__(self, exp0, input_shape):
        self.exp0 = exp0
        self.input_shape = input_shape
        
    def create(self):
        # Input layer
        inputs = Input(self.input_shape)
    
        # Encoder
        conv1 = Conv2D(2**(self.exp0), (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(2**(self.exp0), (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
    
        conv2 = Conv2D(2**(self.exp0+1), (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(2**(self.exp0+1), (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
    
        conv3 = Conv2D(2**(self.exp0+2), (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(2**(self.exp0+2), (3, 3), activation='relu', padding='same')(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D((2, 2))(drop3)
        
        # Bridge
        conv4 = Conv2D(2**(self.exp0+3), (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(2**(self.exp0+3), (3, 3), activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
    
        # Decoder
        up5 = Conv2D(2**(self.exp0+2), (2, 2), activation='relu', padding='same')(UpSampling2D((2, 2))(drop4))
        merge5 = Concatenate(axis=3)([conv3, up5])
        conv5 = Conv2D(2**(self.exp0+2), (3, 3), activation='relu', padding='same')(merge5)
        conv5 = Conv2D(2**(self.exp0+2), (3, 3), activation='relu', padding='same')(conv5)
    
        up6 = Conv2D(2**(self.exp0+1), (2, 2), activation='relu', padding='same')(UpSampling2D((2, 2))(conv5))
        merge6 = Concatenate(axis=3)([conv2, up6])
        conv6 = Conv2D(2**(self.exp0+1), (3, 3), activation='relu', padding='same')(merge6)
        conv6 = Conv2D(2**(self.exp0+1), (3, 3), activation='relu', padding='same')(conv6)
        
        up7 = Conv2D(2**(self.exp0), (2, 2), activation='relu', padding='same')(UpSampling2D((2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv1, up7])
        conv7 = Conv2D(2**(self.exp0), (3, 3), activation='relu', padding='same')(merge7)
        conv7 = Conv2D(2**(self.exp0), (3, 3), activation='relu', padding='same')(conv7)
        
        # Output layer
        outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv7)

        return keras.Model(inputs=inputs, outputs=outputs)