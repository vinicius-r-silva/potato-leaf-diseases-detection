#Authors:
#Marianna Karenina - 10821144
#Rodrigo Bragato - 10684573
#Vinicius Ribeiro da Silva - 10828141

#Description:
#Learn an dense neural network and knn models with the features of the images

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Input, Dense
from keras import losses, Sequential

import numpy as np

#Load tha image Features
def readFolder(basePath):
	with open(basePath + '/Healthy.npy', 'rb') as f:
		Healthy = np.load(f)
	with open(basePath + '/Early.npy', 'rb') as f:
		Sick = np.load(f)
	with open(basePath + '/Late.npy', 'rb') as f:
		Sick = np.concatenate((Sick,np.load(f)))

	return Healthy, Sick


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

	
#create dense model
denseModel = Sequential([
	Input(shape = (512,)),
	Dense(2, activation='sigmoid'),
	Dense(1, activation='sigmoid')
])

#Load the features
basePath = './Imgs/VGG16'
Healthy, Sick = readFolder(basePath)
print("Healthy.shape: ", Healthy.shape)
print("Sick.shape: ", Sick.shape)

#Create the output of the neural network(1 -> Healthy, 0 -> Sick)
Y_Healthy = np.ones((Healthy.shape[0],), dtype=np.uint8)
Y_Sick = np.zeros((Sick.shape[0],))

#Join the inputs and outputs
X = np.concatenate((Healthy, Sick))
Y = np.concatenate((Y_Healthy, Y_Sick))

#Split the train, test and validation values
#80% Train, 10% Test and 10% validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
printClassDistribution(Y_train, Y_test, Y_val)

Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))

print("Y_train.shape: ", Y_train.shape)
print("Y_test.shape: ", Y_test.shape)


denseModel.compile(optimizer='adam',
	loss = losses.BinaryCrossentropy(from_logits=True),
	metrics = ['accuracy'])

imgPixelCount = X.shape[1]
denseModel.build((None, imgPixelCount))

#Prints the summary of the network
denseModel.summary()


# Trains the model
denseModel.fit(
  X_train,
  Y_train,
  epochs = 100,
  shuffle = True,
  validation_data = (X_val, Y_val),
)

denseModel.save("./Models/Network")

p_test = denseModel.predict(X_test)

T_P = 0
T_N = 0
F_P = 0
F_N = 0

for i in range(len(p_test)):
	# print(p_test[i], Y_test[i])
	if(p_test[i] > 0.5):
		if(Y_test[i]==0):
			F_P+=1
		else:
			T_P+=1
	else:
		if(Y_test[i]==0):
			T_N+=1
		else:
			F_N+=1

print("T_N: ", T_N)
print("T_P: ", T_P)
print("F_N: ", F_N)
print("F_P: ", F_P)

denseModel.evaluate(X_test, Y_test)

KNN3 = KNeighborsClassifier(n_neighbors=3)
KNN5 = KNeighborsClassifier(n_neighbors=5)
KNN7 = KNeighborsClassifier(n_neighbors=7)


KNN3.fit(X_train,Y_train.flatten())
KNN5.fit(X_train,Y_train.flatten())
KNN7.fit(X_train,Y_train.flatten())

print("KNN3: ", KNN3.score(X_test,Y_test.flatten()))
print("KNN5: ", KNN5.score(X_test,Y_test.flatten()))
print("KNN7: ", KNN7.score(X_test,Y_test.flatten()))

import pickle 

# Its important to use binary mode
knnPickle3 = open('./Models/KNN/knn3', 'wb')
knnPickle5 = open('./Models/KNN/knn5', 'wb')
knnPickle7 = open('./Models/KNN/knn7', 'wb')
pickle.dump(KNN3, knnPickle3)
pickle.dump(KNN5, knnPickle5)
pickle.dump(KNN7, knnPickle7)
knnPickle3.close()
knnPickle5.close()
knnPickle7.close()
