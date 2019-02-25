# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:09:06 2019

@author: Sohail Khan
"""

from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        #apply the keras utility functions that correctly rearange the dimensions
        # of the image
        return img_to_array(image, data_format=self.dataFormat)