#Authors:
#Marianna Karenina - 10821144
#Rodrigo Bragato - 10684573
#Vinicius Ribeiro da Silva - 10828141

#Description:
#Extract Features from images using VGG16 network trained on imagenet

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2

from glob import glob
from tensorflow.keras.applications import densenet
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vggpreprocess_input


#Load Images
def load_images_from_folder(folder):
    images = []
    files = os.listdir(folder)
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        images.append(img)
    return np.array(images)

def readFolder(basePath):
	Healthy = load_images_from_folder(basePath + '/Healthy')
	Early = load_images_from_folder(basePath + '/EarlyBlight')
	Late = load_images_from_folder(basePath + '/LateBlight')

	return Healthy, Early, Late

#Load the VGGModel
VGGModel = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3), pooling='avg')

#Load Images
basePath = './Imgs/RGB'
Healthy, Early, Late = readFolder(basePath)

print('len(Healthy):', len(Healthy))
print('len(Early):', len(Early))
print('len(Late):', len(Late))

#Extract Features and save the results
#The use of the same variable 'X' in the following lines was made to save memory resources during processing
X = vggpreprocess_input(Healthy)
X = VGGModel.predict(X)
np.save('./Imgs/VGG16/Healthy.npy', X)
X = vggpreprocess_input(Early)
X = VGGModel.predict(X)
np.save('./Imgs/VGG16/Early.npy', X)
X = vggpreprocess_input(Late)
X = VGGModel.predict(X)
np.save('./Imgs/VGG16/Late.npy', X)