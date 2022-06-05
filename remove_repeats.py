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

def saveToFolder(imgArray, path, baseName, BW):
	i = 0
	for img in imgArray:
		if(BW):
			cv2.imwrite(path+baseName+"_"+str(i)+'.jpg',cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		else:
			cv2.imwrite(path+baseName+"_"+str(i)+'.jpg',img)
		i+=1

basePath = './potato/imgs/colorful'
ImgsH, ImgsEB, ImgsLB = readFolder(basePath)

ImgsH_Uniques = []
for arr in ImgsH:
    if not any(np.array_equal(arr, unique_arr) for unique_arr in ImgsH_Uniques):
        ImgsH_Uniques.append(arr)
print('Healthy_Uniques len: ' + str(len(ImgsH_Uniques)))

ImgsEB_Uniques = []
for arr in ImgsEB:
    if not any(np.array_equal(arr, unique_arr) for unique_arr in ImgsEB_Uniques):
        ImgsEB_Uniques.append(arr)
print('Early_Blight_Uniques len: ' + str(len(ImgsEB_Uniques)))

ImgsLB_Uniques = []
for arr in ImgsLB:
    if not any(np.array_equal(arr, unique_arr) for unique_arr in ImgsLB_Uniques):
        ImgsLB_Uniques.append(arr)
print('Late_Blight_Uniques len: ' + str(len(ImgsLB_Uniques)))

saveToFolder(ImgsH_Uniques, './Imgs/RGB/Healthy/', 'Healthy', False)
saveToFolder(ImgsEB_Uniques, './Imgs/RGB/EarlyBlight/', 'EarlyB', False)
saveToFolder(ImgsLB_Uniques, './Imgs/RGB/LateBlight/', 'LateB', False)

saveToFolder(ImgsH_Uniques, './Imgs/BW/Healthy/', 'Healthy', True)
saveToFolder(ImgsEB_Uniques, './Imgs/BW/EarlyBlight/', 'EarlyB', True)
saveToFolder(ImgsLB_Uniques, './Imgs/BW/LateBlight/', 'LateB', True)