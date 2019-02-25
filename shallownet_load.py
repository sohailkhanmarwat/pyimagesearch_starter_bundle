# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:46:05 2019

@author: Sohail Khan
@email: mrsohailkhan@gmail.com

"""

from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default="../datasets/animals", help="path to the input dataset")
ap.add_argument("-o", "--output", required=False, default="output/shallownet_load.png", help="path to save the plot")
ap.add_argument("-m", "--model", required=False, default="output/shallownet_weights.hdf5", help="path to save the model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images in the dataset then randomly sample indexes into the image paths list
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# make predictions on the images
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)    




