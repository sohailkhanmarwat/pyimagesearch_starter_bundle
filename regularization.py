# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 03:18:17 2019

@author: Sohail Khan
@email: mrsohailkhan@gmail.com

"""

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the dataset")
args = vars(ap.parse_args())
#print(args)

imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])

(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

for r in (None, "l1", "l2"):
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=100, learning_rate="constant", tol=0.001, eta0=0.01, random_state=42)
    model.fit(testX, testY)
    
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy: {:.2f}%".format(r, acc * 100))
    

# >python regularization.py -d ../datasets/animals
    