# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 00:39:47 2019

@author: Sohail Khan
@email: mrsohailkhan@gmail.com

"""

import numpy as np
import cv2

labels = ["dog", "cat", "panda"]
np.random.seed(1)

W = np.random.randn(3, 3072)
b = np.random.rand(3)

#load example image
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b

for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))
    
cv2.putText(orig,"Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Image", orig)
cv2.waitKey(0)