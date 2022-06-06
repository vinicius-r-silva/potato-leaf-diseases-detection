from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Input, Dense
from keras import losses, Sequential

import numpy as np

def readFolder(basePath):
	with open(basePath + '/Healthy.npy', 'rb') as f:
		Healthy = np.load(f)
	with open(basePath + '/Early.npy', 'rb') as f:
		Sick = np.load(f)
	with open(basePath + '/Late.npy', 'rb') as f:
		Sick = np.concatenate((Sick,np.load(f)))

	return Healthy, Sick

modelVGG = Sequential([
	Input(shape = (512,)),
	Dense(2, activation='sigmoid'),
	Dense(1, activation='sigmoid')
])

basePath = './Imgs/VGG16'
Healthy, Sick = readFolder(basePath)
print("Healthy.shape: ", Healthy.shape)
print("Sick.shape: ", Sick.shape)

HealthyCount = Healthy.shape[0]
SickCount = Sick.shape[0]

Y_Healthy = np.ones((HealthyCount,), dtype=np.uint8)
Y_Sick = np.zeros((SickCount,))

X = np.concatenate((Healthy, Sick))
Y = np.concatenate((Y_Healthy, Y_Sick))

imgPixelCount = X.shape[1]
X = X.reshape(X.shape[0], imgPixelCount)
# X = np.expand_dims(X, axis=-1)
# X = X / 255

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

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
# print("X_val.shape: ", X_val.shape)

print("X_train.max(): ", X_train.max(), ", X_train.min(): ", X_train.min())
print("X_test.max(): ", X_test.max(), ", X_test.min(): ", X_test.min())
# print("X_val.max(): ", X_val.max(), ", X_val.min(): ", X_val.min())

print("X_train.dtype: ", X_train.dtype)
print("Y_train.dtype: ", Y_train.dtype)

# Y_train = to_categorical(Y_train, 1)
# Y_test  = to_categorical(Y_test, 1)
# Y_val  = to_categorical(Y_val, 1)

Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
# Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))

print("Y_train.shape: ", Y_train.shape)
print("Y_test.shape: ", Y_test.shape)


modelVGG.compile(optimizer='adam',
	loss = losses.BinaryCrossentropy(from_logits=True),
	metrics = ['accuracy'])

modelVGG.build((None, imgPixelCount))

#Prints the summary of the network
# modelVGG.encoder.summary()
# modelVGG.classifier.summary()
modelVGG.summary()


# Trains the model
# modelVGG.fit(
#   X_train,
#   Y_train,
#   epochs = 100,
#   shuffle = True,
#   validation_split = 0.9,
# )

# modelVGG.save("./Models/Network")

p_test = modelVGG.predict(X_test)

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

modelVGG.evaluate(X_test, Y_test)

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
