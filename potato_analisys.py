import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        # print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def readFolder(basePath):
    Healthy = load_images_from_folder(basePath + '/Healthy')
    Early_Blight = load_images_from_folder(basePath + '/Early_Blight')
    Late_Blight = load_images_from_folder(basePath + '/Late_Blight')

    return Healthy, Early_Blight, Late_Blight

def readTestingFolfer():
    basePath = './potato/imgs/PLD_3_Classes_256/Testing'
    return readFolder(basePath)

def readTrainingFolfer():
    basePath = './potato/imgs/PLD_3_Classes_256/Training'
    return readFolder(basePath)

def readValidationFolfer():
    basePath = './potato/imgs/PLD_3_Classes_256/Validation'
    return readFolder(basePath)


TestingImages = readTestingFolfer()
Healthy = TestingImages[0]
Early_Blight = TestingImages[1]
Late_Blight = TestingImages[2]

TrainingImages = readTrainingFolfer()
Healthy += TrainingImages[0]
Early_Blight += TrainingImages[1]
Late_Blight += TrainingImages[2]

ValidationImages = readValidationFolfer()
Healthy += ValidationImages[0]
Early_Blight += ValidationImages[1]
Late_Blight += ValidationImages[2]

print('\nQuantidade de imagens:')
print('Healthy len: ' + str(len(Healthy)))
print('Early_Blight len: ' + str(len(Early_Blight)))
print('Late_Blight len: ' + str(len(Late_Blight)))
print('Total len: ' + str(len(Healthy) + len(Early_Blight) + len(Late_Blight)))

print('\nImagem')
print('Tamanho: ' + str(Healthy[0].shape))

print('\nUniques')
Healthy_Uniques = []
for arr in Healthy:
    if not any(np.array_equal(arr, unique_arr) for unique_arr in Healthy_Uniques):
        Healthy_Uniques.append(arr)
print('Healthy_Uniques len: ' + str(len(Healthy_Uniques)))

Early_Blight_Uniques = []
for arr in Early_Blight:
    if not any(np.array_equal(arr, unique_arr) for unique_arr in Early_Blight_Uniques):
        Early_Blight_Uniques.append(arr)
print('Early_Blight_Uniques len: ' + str(len(Early_Blight_Uniques)))

Late_Blight_Uniques = []
for arr in Late_Blight:
    if not any(np.array_equal(arr, unique_arr) for unique_arr in Late_Blight_Uniques):
        Late_Blight_Uniques.append(arr)
print('Late_Blight_Uniques len: ' + str(len(Late_Blight_Uniques)))

print('Total uniques len: ' + str(len(Healthy_Uniques) + len(Early_Blight_Uniques) + len(Late_Blight_Uniques)))


print('qtd duplicates:')
print('Healthy_Uniques qtf: ' + str(len(Healthy) - len(Healthy_Uniques)))
print('Early_Blight_Uniques qtd: ' + str(len(Early_Blight) - len(Early_Blight_Uniques)))
print('Late_Blight_Uniques qtd: ' + str(len(Late_Blight) - len(Late_Blight_Uniques)))
print('total: ' + str(len(Healthy) + len(Early_Blight) + len(Late_Blight) - len(Healthy_Uniques) - len(Early_Blight_Uniques) - len(Late_Blight_Uniques)))