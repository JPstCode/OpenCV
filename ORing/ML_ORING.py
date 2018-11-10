import numpy as np
import cv2 as cv
from sklearn.neural_network import MLPClassifier
import glob

layers = [60,40,25,10]
labels_array = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

for j in range(0,1):
    for i in range(0,15):
        labels_array.append(labels_array[i])

labels_array = np.asarray(labels_array).reshape(1,-1)
print(labels_array.shape)

train_images = [cv.imread(file,0) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\ORing\Training\*.jpg')]

for j in range(0,1):
    for i in range(0,15):
        train_images.append(train_images[i])


train_images = np.asarray(train_images).reshape(-1,48400)
print(train_images.shape)
input("jas")


test_images = [cv.imread(file,0) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\ORing\Test\*.jpg')]

for j in range(0,2):
    for i in range(0,12):
        test_images.append(test_images[i])

test_images = np.asarray(test_images).reshape(-1,48400)

print(test_images.shape)
#print(train_images[2][11000:11220])
#print(labels_array.shape)
input("joh")


#Train MLP
mlp = MLPClassifier(hidden_layer_sizes=layers,solver='lbfgs')
mlp.fit(train_images,labels_array[0])


#test classifires
for i in range(0,len(test_images)):
    est = mlp.predict(test_images[i].reshape(1,-1))
    print(est)

