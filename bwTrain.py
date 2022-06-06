#Authors:
#Marianna Karenina - 10821144
#Rodrigo Bragato - 10684573
#Vinicius Ribeiro da Silva - 10828141

#Description:
#Load the black and White images e create a neural model


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import losses, Sequential, Model

import tensorflow as tf
import numpy as np
import cv2
import os


#Load images as black white and change its lengh and width for 1/4 of the original size
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


#Load images as a numpy array
def readFolder(basePath):
	Healthy = load_images_from_folder(basePath + '/Healthy')
	Sick = load_images_from_folder(basePath + '/EarlyBlight')
	Sick += load_images_from_folder(basePath + '/LateBlight')

	return np.array(Healthy), np.array(Sick)

def printClassDistribution(Y_train, Y_test, Y_val):
	#Count the distribution
	(unique_train, counts_train) = np.unique(Y_train, return_counts=True)
	frequencies_train = np.asarray((unique_train, counts_train)).T
	(unique_test, counts_test) = np.unique(Y_test, return_counts=True)
	frequencies_test = np.asarray((unique_test, counts_test)).T
	(unique_val, counts_val) = np.unique(Y_val, return_counts=True)
	frequencies_val = np.asarray((unique_val, counts_val)).T
 
	print("classes frequency at train data: ")
	print("\t 0 -> Sick, 1 -> Healthy")
	print("\t", frequencies_train[0][0], " -> ", frequencies_train[0][1], "(", frequencies_train[0][1] / (frequencies_train[0][1] + frequencies_train[1][1]), "%)")
	print("\t", frequencies_train[1][0], " -> ", frequencies_train[1][1], "(", frequencies_train[1][1] / (frequencies_train[0][1] + frequencies_train[1][1]), "%)")
	print("classes frequency at test data: ")
	print("\t 0 -> Sick, 1 -> Healthy")
	print("\t", frequencies_test[0][0], " -> ", frequencies_test[0][1], "(", frequencies_test[0][1] / (frequencies_test[0][1] + frequencies_test[1][1]), "%)")
	print("\t", frequencies_test[1][0], " -> ", frequencies_test[1][1], "(", frequencies_test[1][1] / (frequencies_test[0][1] + frequencies_test[1][1]), "%)")
	print("classes frequency at validation data: ")
	print("\t 0 -> Sick, 1 -> Healthy")
	print("\t", frequencies_val[0][0], " -> ", frequencies_val[0][1], "(", frequencies_val[0][1] / (frequencies_val[0][1] + frequencies_val[1][1]), "%)")
	print("\t", frequencies_val[1][0], " -> ", frequencies_val[1][1], "(", frequencies_val[1][1] / (frequencies_val[0][1] + frequencies_val[1][1]), "%)")

	# print("X_train.shape: ", X_train.shape)
	# print("X_test.shape: ", X_test.shape)
	# print("X_val.shape: ", X_val.shape)

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

#Loads the images
basePath = './Imgs/BW'
Healthy, Sick = readFolder(basePath)
print("Healthy.shape: ", Healthy.shape)
print("Sick.shape: ", Sick.shape)


#Create the output of the neural network(1 -> Healthy, 0 -> Sick)
Y_Healthy = np.ones((Healthy.shape[0],), dtype=np.uint8)
Y_Sick = np.zeros((Sick.shape[0],))

#Join the inputs and outputs
X = np.concatenate((Healthy, Sick))
Y = np.concatenate((Y_Healthy, Y_Sick))

#Flatten the image and change its range from 0-255 to 0-1
imgPixelCount = X.shape[1] * X.shape[2]
X = X.reshape(X.shape[0], imgPixelCount)
X = np.expand_dims(X, axis=-1)
X = X / 255

#Split the train, test and validation values
#80% Train, 10% Test and 10% validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
printClassDistribution(Y_train, Y_test, Y_val)

Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))


modelo = HazeRemover()
modelo.compile(optimizer='adam',
	loss = losses.BinaryCrossentropy(from_logits=True),
	metrics = ['accuracy'])

modelo.build((None, imgPixelCount))

#Prints the summary of the network
# modelo.encoder.summary()
# modelo.classifier.summary()
modelo.summary()


# Trains the model
modelo.fit(
  X_train,
  Y_train,
  epochs = 500,
  shuffle = True,
  validation_data = (X_val, Y_val),
)

modelo.evaluate(X_test, Y_test)
modelo.save("./Models/NetworkBW")

pred = modelo.predict(X)
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

print("===========Rede===========")
print("T_N: {:5d}      F_P: {:5d}".format(T_N, F_P))
print("T_P: {:5d}      F_N: {:5d}".format(T_P, F_N))
print("Pr0: {:.3f}      Pr1: {:.3f}".format((T_N/(T_N+F_N)),(T_P/(T_P+F_P))))
print("Rv0: {:.3f}      Rv1: {:.3f}".format((T_N/(T_N+F_P)),(T_P/(T_P+F_N))))
print("Acc: {:.3f}                ".format((T_N+T_P)/(T_P+F_N+T_N+F_P)))
print("==========================")