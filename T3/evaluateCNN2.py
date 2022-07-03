#Authors:
#Marianna Karenina - 10821144
#Rodrigo Bragato - 10684573
#Vinicius Ribeiro da Silva - 10828141

#Description:
#Load the Dense and Knn models and show the evaluations of each one

import tensorflow as tf
import numpy as np
import pickle
import keras
import os
import cv2
from enum import Enum 
import pandas as pd
from keras import backend as K


def load_images_from_folder(folder):
    images = []
    files = os.listdir(folder)
    img = cv2.imread(os.path.join(folder,files[0]))
    width = int(img.shape[1])# * 0.25)
    height = int(img.shape[0])# * 0.25)
    dim = (width, height)
    for filename in files:
        # print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        images.append(img)
        #if img is not None:
        #   images.append(cv2.resize(img, dim, interpolation = cv2.INTER_AREA))
    return images

#Load images as a numpy array
def readFolder(basePath):
    Healthy = load_images_from_folder(basePath + '/Healthy')
    Early = load_images_from_folder(basePath + '/EarlyBlight')
    Late = load_images_from_folder(basePath + '/LateBlight')

    return np.array(Healthy), np.array(Early), np.array(Late)

basePath = '../Imgs/RGB'
Healthy, Early, Late = readFolder(basePath)
Sick = np.vstack((Early, Late))

#------------------EVALUATE CNN2------------------#
CNN2 = keras.models.load_model('../Models/CNN2Classes')
Y_Healthy = np.ones((Healthy.shape[0],), dtype=np.uint8)
Y_Sick = np.zeros((Sick.shape[0],))

#Join the inputs and outputs
X = np.concatenate((Healthy, Sick))
Y = np.concatenate((Y_Healthy, Y_Sick))

#Change the pixel value range from 0-255 to 0-1
X = X / 255

#Evaluate the model over all saved features
pred = CNN2.predict(X)
CNN2.evaluate(X, Y)

T_P = 0
T_N = 0
F_P = 0
F_N = 0

for i in range(len(pred)):
	if(pred[i] > 0.5):
		if(Y[i]==0):
			F_P+=1
		else:
			T_P+=1
	else:
		if(Y[i]==0):
			T_N+=1
		else:
			F_N+=1


print("===========CNN2===========")
print("T_N: {:5d}      F_P: {:5d}".format(T_N, F_P))
print("T_P: {:5d}      F_N: {:5d}".format(T_P, F_N))
print("Pr0: {:.3f}      Pr1: {:.3f}".format((T_N/(T_N+F_N)),(T_P/(T_P+F_P))))
print("Rv0: {:.3f}      Rv1: {:.3f}".format((T_N/(T_N+F_P)),(T_P/(T_P+F_N))))
print("Acc: {:.3f}                ".format((T_N+T_P)/(T_P+F_N+T_N+F_P)))
print("==========================")