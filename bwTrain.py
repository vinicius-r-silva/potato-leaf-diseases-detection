from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import losses, Sequential, Model

import tensorflow as tf
import numpy as np
import cv2
import os


def load_images_from_folder(folder):
	images = []
	files = os.listdir(folder)
	img = cv2.imread(os.path.join(folder,files[0]),cv2.IMREAD_GRAYSCALE)
	width = int(img.shape[1] * 0.25)
	height = int(img.shape[0] * 0.25)
	dim = (width, height)
	for filename in files:
		# print(filename)
		img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
		if img is not None:
			images.append(cv2.resize(img, dim, interpolation = cv2.INTER_AREA))
	return images

def readFolder(basePath):
	Healthy = load_images_from_folder(basePath + '/Healthy')
	Sick = load_images_from_folder(basePath + '/EarlyBlight')
	Sick += load_images_from_folder(basePath + '/LateBlight')

	return np.array(Healthy), np.array(Sick)

#Neural Network
class HazeRemover(Model):
	def __init__(self):

		super(HazeRemover, self).__init__() 

		self.encoder = Sequential([ 
			Input(shape=(4096,1)),#2^10
			MaxPooling1D(8),
			Flatten(),
			# Dense(256, activation='relu'),#2^8
			Dense(16, activation='relu')])#2^5
		self.classifier = Sequential([ 
			Dense(1, activation='sigmoid')])

	def call(self, x): 
		encoded = self.encoder(x)
		decoded = self.classifier(encoded)
		return decoded


basePath = './Imgs/BW'
Healthy, Sick = readFolder(basePath)
print("Healthy.shape: ", Healthy.shape)
print("Sick.shape: ", Sick.shape)

HealthyCount = Healthy.shape[0]
SickCount = Sick.shape[0]

Y_Healthy = np.ones((HealthyCount,), dtype=np.uint8)
Y_Sick = np.zeros((SickCount,))

X = np.concatenate((Healthy, Sick))
Y = np.concatenate((Y_Healthy, Y_Sick))

imgPixelCount = X.shape[1] * X.shape[2]
X = X.reshape(X.shape[0], imgPixelCount)
X = np.expand_dims(X, axis=-1)
X = X / 255

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

(unique_train, counts_train) = np.unique(Y_train, return_counts=True)
frequencies_train = np.asarray((unique_train, counts_train)).T
(unique_test, counts_test) = np.unique(Y_test, return_counts=True)
frequencies_test = np.asarray((unique_test, counts_test)).T


print("classes frequency at train data: ")
print("\t 0 -> Sick, 1 -> Healthy")
print("\t", frequencies_train[0][0], " -> ", frequencies_train[0][1], "(", frequencies_train[0][1] / (frequencies_train[0][1] + frequencies_train[1][1]), "%)")
print("\t", frequencies_train[1][0], " -> ", frequencies_train[1][1], "(", frequencies_train[1][1] / (frequencies_train[0][1] + frequencies_train[1][1]), "%)")
print("classes frequency at test data: ")
print("\t 0 -> Sick, 1 -> Healthy")
print("\t", frequencies_test[0][0], " -> ", frequencies_test[0][1], "(", frequencies_test[0][1] / (frequencies_test[0][1] + frequencies_test[1][1]), "%)")
print("\t", frequencies_test[1][0], " -> ", frequencies_test[1][1], "(", frequencies_test[1][1] / (frequencies_test[0][1] + frequencies_test[1][1]), "%)")

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
print("X_val.shape: ", X_val.shape)

print("X_train.max(): ", X_train.max(), ", X_train.min(): ", X_train.min())
print("X_test.max(): ", X_test.max(), ", X_test.min(): ", X_test.min())
print("X_val.max(): ", X_val.max(), ", X_val.min(): ", X_val.min())

print("X_train.dtype: ", X_train.dtype)
print("Y_train.dtype: ", Y_train.dtype)

# Y_train = to_categorical(Y_train, 1)
# Y_test  = to_categorical(Y_test, 1)
# Y_val  = to_categorical(Y_val, 1)

Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))

print("Y_train.shape: ", Y_train.shape)
print("Y_test.shape: ", Y_test.shape)


modelo = HazeRemover()
modelo.compile(optimizer='adam',
	loss = losses.BinaryCrossentropy(from_logits=True),
	metrics = ['accuracy'])

modelo.build((None, imgPixelCount))

#Prints the summary of the network
# modelo.encoder.summary()
# modelo.classifier.summary()
modelo.summary()


#Trains the model
modelo.fit(
  X_train,
  Y_train,
  epochs = 500,
  shuffle = True,
  validation_data = (X_val, Y_val),
)

modelo.evaluate(X_test, Y_test)

#Saves the network
modelo.save("./network/models")