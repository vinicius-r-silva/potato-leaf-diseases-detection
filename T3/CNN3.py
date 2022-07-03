from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, InputLayer
from keras import losses, Sequential, Model
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np

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
print("Healthy.shape: ", Healthy.shape)
print("Early.shape: ", Early.shape)
print("Late.shape: ", Late.shape)

model = Sequential();
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(256,256,3), padding='same'));
model.add(MaxPooling2D(pool_size = (2,2)));
model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'));
model.add(Conv2D(16, kernel_size=3, activation='relu', padding='same'));
model.add(MaxPooling2D(pool_size = (2,2)));
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'));
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'));
model.add(MaxPooling2D(pool_size = (2,2)));
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'));
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'));
model.add(MaxPooling2D(pool_size = (2,2)));
model.add(Flatten());
model.add(Dense(16, activation='relu'));
model.add(Dense(3, activation='softmax'));

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'])
print(model.summary())


#Create the output of the neural network(Healthy ->[1,0,0] ; Early ->[0,1,0] ; Late ->[0,0,1])
# Y_Healthy = np.ones((Healthy.shape[0],), dtype=np.uint8)
# Y_Sick = np.zeros((Sick.shape[0],))
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

#Split the train, test and validation values
#80% Train, 10% Test and 10% validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
# printClassDistribution(Y_train, Y_test, Y_val)

Y_train = np.asarray(Y_train).astype('float32')
Y_test = np.asarray(Y_test).astype('float32')
Y_val = np.asarray(Y_val).astype('float32')

model.fit(
  X_train,
  Y_train,
  epochs = 40,
  shuffle = True,
  validation_data = (X_val, Y_val),
)

model.evaluate(X_test, Y_test)
model.save("../Models/CNN3Classes")

