import tensorflow as tf
import pandas as pd
import numpy as np

from glob import glob
from tensorflow.keras.applications import densenet
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vggpreprocess_input
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input as effipreprocess_input

model = VGG16(weights='imagenet', include_top=False, pooling='avg')

img_path = './Imgs/RGB/Healthy/Healthy_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = vggpreprocess_input(x)

features = model.predict(x)
# print('features:', features)
print('features.shape:', features.shape)



model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(256,256,3), pooling='avg')

img_path = './Imgs/RGB/Healthy/Healthy_0.jpg'
img = image.load_img(img_path)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = effipreprocess_input(x)

features = model.predict(x)
print('features.shape:', features.shape)


