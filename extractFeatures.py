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
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input as effpreprocess_input



def load_images_from_folder(folder):
    images = []
    files = os.listdir(folder)
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        # images.append(np.expand_dims(image.img_to_array(img), axis=0))
        images.append(img)
    return np.array(images)

def readFolder(basePath):
	Healthy = load_images_from_folder(basePath + '/Healthy')
	Early = load_images_from_folder(basePath + '/EarlyBlight')
	Late = load_images_from_folder(basePath + '/LateBlight')

	return Healthy, Early, Late



VGGModel = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3), pooling='avg')
EffModel = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(256,256,3), pooling='avg')

basePath = './Imgs/RGB'
Healthy, Early, Late = readFolder(basePath)

print('len(Healthy):', len(Healthy))
print('len(Early):', len(Early))
print('len(Late):', len(Late))

# HealthyVggPreProcess = vggpreprocess_input(Healthy)
# EarlyVggPreProcess = vggpreprocess_input(Early)
# LateVggPreProcess = vggpreprocess_input(Late)

# HealthyEffPreProcess = effpreprocess_input(Healthy)
# EarlyEffPreProcess = effpreprocess_input(Early)
# LateEffPreProcess = effpreprocess_input(Late)


X = vggpreprocess_input(Healthy)
X = VGGModel.predict(X)
np.save('./Imgs/VGG16/Healthy.npy', X)
X = vggpreprocess_input(Early)
X = VGGModel.predict(X)
np.save('./Imgs/VGG16/Early.npy', X)
X = vggpreprocess_input(Late)
X = VGGModel.predict(X)
np.save('./Imgs/VGG16/Late.npy', X)

X = effpreprocess_input(Healthy)
X = EffModel.predict(X)
np.save('./Imgs/EffNetB2/Healthy.npy', X)
X = effpreprocess_input(Early)
X = EffModel.predict(X)
np.save('./Imgs/EffNetB2/Early.npy', X)
X = effpreprocess_input(Late)
X = EffModel.predict(X)
np.save('./Imgs/EffNetB2/Late.npy', X)


# HealthyVggFeatures = VGGModel.predict(HealthyVggPreProcess)
# EarlyVggFeatures = VGGModel.predict(EarlyVggPreProcess)
# LateVggFeatures = VGGModel.predict(LateVggPreProcess)

# HealthyEffFeatures = EffModel.predict(HealthyEffPreProcess)
# EarlyEffFeatures = EffModel.predict(EarlyEffPreProcess)
# LateEffFeatures = EffModel.predict(LateEffPreProcess)

# print('HealthyVggFeatures.shape:', HealthyVggFeatures.shape)
# print('EarlyVggFeatures.shape:', EarlyVggFeatures.shape)
# print('LateVggFeatures.shape:', LateVggFeatures.shape)
# print('HealthyEffFeatures.shape:', HealthyEffFeatures.shape)
# print('EarlyEffFeatures.shape:', EarlyEffFeatures.shape)
# print('LateEffFeatures.shape:', LateEffFeatures.shape)


# np.save('./Imgs/VGG16/Healthy.npy', HealthyVggFeatures)
# np.save('./Imgs/VGG16/Early.npy', EarlyVggFeatures)
# np.save('./Imgs/VGG16/Late.npy', LateVggFeatures)

# np.save('./Imgs/EffNetB2/Healthy.npy', HealthyEffFeatures)
# np.save('./Imgs/EffNetB2/Early.npy', EarlyEffFeatures)
# np.save('./Imgs/EffNetB2/Late.npy', LateEffFeatures)

