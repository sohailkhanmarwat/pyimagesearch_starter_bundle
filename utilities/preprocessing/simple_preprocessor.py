# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 03:25:00 2019

@author: Sohail Khan
"""

import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height= height
        self.inter = inter
    
    
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)