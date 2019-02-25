# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 02:28:23 2019

@author: Sohail Khan
@email: mrsohailkhan@gmail.com

"""

from imutils import paths
import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt

# construct the argument parse and aprse the agruments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, default="./downloads", help="path to input directory of images" )
ap.add_argument("-a", "--annot", required=False, default="./dataset", help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image then initialize the dictionary of character counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # display an update ot the user
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    print(imagePath)
    try:
        # load the image and convert it to grayscale, then pad the image to the
        # ensure digits caught on the border of the image are retained
        image = cv2.imread(imagePath)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray  = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        
        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #find contours in the image, keeping only the four largest onces
        im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        print(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        # loop over the contours 
        for c in cnts:
            # computer the bounding box for the contour then extract the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + h + 5]
            
            # display the character, mamking it larger enough for us to see, then wait for a keypress
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            
            # if the ,', key is pressed, the ignore the character
            if key == ord("'"):
                print("[INFO] ignoring character")
                continue
            
            # grab the key that was pressed and construct the path the output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])
            
            # if the output directory does not exist, create it 
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            
            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)
            
            # increment the count for the current key
            counts[key] = count + 1
            
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
#    except:
#        print("[INFO] skipping image...")