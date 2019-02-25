# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 01:35:28 2019

@author: Sohail Khan
@email: mrsohailkhan@gmail.com

"""

from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False, default="output/lenet_mnist.png", help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time using this dataset then the 55MB download
# may take a minute)
print("[INFO] accessing MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data

# if we are using "channel first" ordering, then reshape the design
# matrix such that the matrix is:
# num_samples x depth x rows x columns

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
    
# otherwise, we are using "channels last" ordering, so the design matrix sahpe
# should be: num_sample x rows x columns
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

# now the data matrix is properly reshaped, we can perofrm training and testing split 
# scale the input data to the range [0, 1] and perform a train/test split
(trainX, testX, trainY, testY) = train_test_split(data/255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY  = lb.transform(testY)

# intialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure(figsize=(10, 10))
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig(args["output"], dpi=1000)









