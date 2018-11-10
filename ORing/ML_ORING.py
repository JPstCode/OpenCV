import numpy as np
import cv2 as cv
from sklearn.neural_network import MLPClassifier
import glob

#Define layers for MLP
layers = [60,40,25,10]

#Labels for training pictures
labels_array = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

#Multiply training samples
multiplier = 1

for j in range(0,multiplier):
    for i in range(0,15):
        labels_array.append(labels_array[i])


train_images = [cv.imread(file,0) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\ORing\Training\*.jpg')]

for i in range(0,multiplier):
    for j in range(0,15):
        train_images.append(train_images[j])


test_images = [cv.imread(file,0) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\ORing\Test\*.jpg')]

for i in range(0,multiplier):
    for j in range(0,12):
        test_images.append(test_images[j])


#convert image and label lists to numpy arrays
labels_array = np.asarray(labels_array).reshape(1,-1)
train_images = np.asarray(train_images).reshape(-1,48400)
test_images = np.asarray(test_images).reshape(-1,48400)


#print(labels_array.shape)
#print(train_images.shape)
#print(test_images.shape)

#Train MLP
mlp = MLPClassifier(hidden_layer_sizes=layers,solver='lbfgs')
mlp.fit(train_images,labels_array[0])


#test classifires
for i in range(0,len(test_images)):
    est = mlp.predict(test_images[i].reshape(1,-1))
    print(est)

