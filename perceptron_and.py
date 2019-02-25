# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 02:20:18 2019

@author: Sohail Khan
@email: mrsohailkhan@gmail.com

"""

from nn import Perceptron
import numpy as np

# construct the AND dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4x2
y = np.array([[0], [0], [0], [1]])

# define our perceptron and train it
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that our perceptron is train ed we can evaluate it
print("[INFO] testing perceptron ...")

# now that our network is trained, loop over the datapoints
for (x, target) in zip(X, y):
# make a prediction on the data point and disply the result to our console
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
    