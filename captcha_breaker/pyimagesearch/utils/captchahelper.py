# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 05:08:15 2019

@author: Sohail Khan
"""

import imutils
import cv2

def preprocess(image, width, height):
    # grab the dimensions of the image, then initailze the padding values
    (h, w) = image.shape[:2]
    
    # if the width is greater than the height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)
        
    # otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)
    
    # determine the padding values for the width and height to obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    
    # pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    # return the pre-processed image
    return image
    