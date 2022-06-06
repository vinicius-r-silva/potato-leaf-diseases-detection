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

#Load the features
def readFolder(basePath):
	with open(basePath + '/Healthy.npy', 'rb') as f:
		Healthy = np.load(f)
	with open(basePath + '/Early.npy', 'rb') as f:
		Sick = np.load(f)
	with open(basePath + '/Late.npy', 'rb') as f:
		Sick = np.concatenate((Sick,np.load(f)))

	return Healthy, Sick


#Load features
basePath = './Imgs/VGG16'
Healthy, Sick = readFolder(basePath)

#Create the output of the neural network(1 -> Healthy, 0 -> Sick)
Y_Healthy = np.ones((Healthy.shape[0],), dtype=np.uint8)
Y_Sick = np.zeros((Sick.shape[0],))

#Join the inputs and outputs
X = np.concatenate((Healthy, Sick))
Y = np.concatenate((Y_Healthy, Y_Sick))

#Loads the dense model
denseModel = keras.models.load_model('./Models/Network')

#Evaluate the model over all saved features
pred = denseModel.predict(X)

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

denseModel.evaluate(X, Y)

print("===========Rede===========")
print("T_N: {:5d}      F_P: {:5d}".format(T_N, F_P))
print("T_P: {:5d}      F_N: {:5d}".format(T_P, F_N))
print("Pr0: {:.3f}      Pr1: {:.3f}".format((T_N/(T_N+F_N)),(T_P/(T_P+F_P))))
print("Rv0: {:.3f}      Rv1: {:.3f}".format((T_N/(T_N+F_P)),(T_P/(T_P+F_N))))
print("Acc: {:.3f}                ".format((T_N+T_P)/(T_P+F_N+T_N+F_P)))
print("==========================")


# load the Knn models
KNN3 = pickle.load(open('./Models/KNN/knn3', 'rb'))
KNN5 = pickle.load(open('./Models/KNN/knn5', 'rb'))
KNN7 = pickle.load(open('./Models/KNN/knn7', 'rb'))

#Evaluate the models over all saved features
res = [KNN3.predict(X), KNN5.predict(X), KNN7.predict(X)]

KnnTP = [0, 0, 0]
KnnTN = [0, 0, 0]
KnnFP = [0, 0, 0]
KnnFN = [0, 0, 0]

for i in range(len(res[0])):
	for j in range(3):
		if(res[j][i] == Y[i]):
			if(Y[i]==0):
				KnnTN[j]+=1
			else:
				KnnTP[j]+=1
		else:
			if(Y[i]==0):
				KnnFN[j]+=1
			else:
				KnnFP[j]+=1

for i in range(3):
	print("===========KNN{}===========".format(1+2*(i+1)))
	print("T_N: {:5d}      F_P: {:5d}".format(KnnTN[i], KnnFP[i]))
	print("T_P: {:5d}      F_N: {:5d}".format(KnnTP[i], KnnFN[i]))
	print("Pr0: {:.3f}      Pr1: {:.3f}".format((KnnTN[i]/(KnnTN[i]+KnnFN[i])),(KnnTP[i]/(KnnTP[i]+KnnFP[i]))))
	print("Rv0: {:.3f}      Rv1: {:.3f}".format((KnnTN[i]/(KnnTN[i]+KnnFP[i])),(KnnTP[i]/(KnnTP[i]+KnnFN[i]))))
	print("Acc: {:.3f}                ".format((KnnTN[i]+KnnTP[i])/(KnnTP[i]+KnnFN[i]+KnnTN[i]+KnnFP[i])))
	print("==========================")
