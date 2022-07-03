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

#------------------EVALUATE CNN3------------------#
K.clear_session()
CNN3 = keras.models.load_model('../Models/CNN3Classes')
Y_Healthy = np.full((Healthy.shape[0],3), np.array([1,0,0]))
Y_Early = np.full((Early.shape[0],3), np.array([0,1,0]))
Y_Late = np.full((Late.shape[0],3), np.array([0,0,1]))

print('Y_Healthy:', Y_Healthy.shape)
print('Y_Early:', Y_Early.shape)
print('Y_Late:', Y_Late.shape)

#Join the inputs and outputs
X = np.concatenate((Healthy, Early, Late))
Y = np.concatenate((Y_Healthy, Y_Early, Y_Late))

#Change the pixel value range from 0-255 to 0-1
X = X / 255

#Evaluate the model over all saved features
pred = CNN3.predict(X)
CNN3.evaluate(X, Y)

HealthyIdx = 0
EarlyIdx = 1
LateIdx = 2

#Confusion Matrix
CM = pd.DataFrame(data={"prediction": ['H', 'E', 'L'], "H": [0,0,0], "E": [0,0,0], "L": [0,0,0]})
CM.set_index('prediction', inplace = True)

# for i in range(20):
for i in range(len(pred)):
    predResult = np.argmax(pred[i])
    trueResult = np.argmax(Y[i])

    if(trueResult == HealthyIdx):
        if(predResult == HealthyIdx):
            CM.loc['H', 'H'] += 1
        elif(predResult == EarlyIdx):
            CM.loc['E', 'H'] += 1
        elif(predResult == LateIdx):
            CM.loc['L', 'H'] += 1

    elif(trueResult == EarlyIdx):
        if(predResult == HealthyIdx):
            CM.loc['H', 'E'] += 1
        elif(predResult == EarlyIdx):
            CM.loc['E', 'E'] += 1
        elif(predResult == LateIdx):
            CM.loc['L', 'E'] += 1

    elif(trueResult == LateIdx):
        if(predResult == HealthyIdx):
            CM.loc['H', 'L'] += 1
        elif(predResult == EarlyIdx):
            CM.loc['E', 'L'] += 1
        elif(predResult == LateIdx):
            CM.loc['L', 'L'] += 1

print('matrix de confusao')
print(CM, '\n')

print("===========CNN3===========")
print("Pr_Healthy: {:.3f}".format(CM.loc['H', 'H']/(CM.loc['H', 'H'] + CM.loc['H', 'E'] + CM.loc['H', 'L'])))
print("Pr_Early:   {:.3f}".format(CM.loc['E', 'E']/(CM.loc['E', 'H'] + CM.loc['E', 'E'] + CM.loc['E', 'L'])))
print("Pr_Late:    {:.3f}".format(CM.loc['L', 'L']/(CM.loc['L', 'H'] + CM.loc['L', 'E'] + CM.loc['L', 'L'])))

print("Rv_Healthy: {:.3f}".format(CM.loc['H', 'H']/(CM.loc['H', 'H'] + CM.loc['E', 'H'] + CM.loc['L', 'H'])))
print("Rv_Early:   {:.3f}".format(CM.loc['E', 'E']/(CM.loc['H', 'E'] + CM.loc['E', 'E'] + CM.loc['L', 'E'])))
print("Rv_Late:    {:.3f}".format(CM.loc['L', 'L']/(CM.loc['H', 'L'] + CM.loc['E', 'L'] + CM.loc['L', 'L'])))

print("Acc:        {:.3f}".format((CM.loc['H', 'H'] + CM.loc['E', 'E'] + CM.loc['L', 'L']) / (
                                    CM.loc['H', 'H'] + CM.loc['H', 'E'] + CM.loc['H', 'L'] +
                                    CM.loc['E', 'H'] + CM.loc['E', 'E'] + CM.loc['E', 'L'] +
                                    CM.loc['L', 'H'] + CM.loc['L', 'E'] + CM.loc['L', 'L']
                                    )))
print("==========================")
