import numpy as np
import cv2
import os
import logging

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

import pickle

#############################################
path = 'C:\\Users\\SOUROV\\PycharmProjects\\HandWrittenOCR\\Training\\Training'
test_ratio = 0.2
validation_ratio = 0.2
imageDimensions = (32, 32, 3)

batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 200
#############################################

images = []
classNo = []
myList = os.listdir(path)
print("Total No of Classes Detected :", len(myList))
noOfClasses = len(myList)
print("Importing classes ................")

logging.getLogger("tensorflow").setLevel(logging.ERROR)

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
    x += 1
print(" ")
print("Total Images in Images List = ", len(images))
print("Total IDS in ClassNo List= ", len(classNo))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_ratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numofSamples = []
for x in range(0, noOfClasses):
    # print(len(np.where(Y_train == x)[0]))
    numofSamples.append(len(np.where(Y_train == x)[0]))
print(numofSamples)


plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numofSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


# img = preProcessing(X_train[67])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)
# print(X_train[30].shape)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

Y_train = to_categorical(Y_train, noOfClasses)
Y_test = to_categorical(Y_test, noOfClasses)
Y_validation = to_categorical(Y_validation, noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    onOfNode = 500

    models = Sequential()
    models.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                                imageDimensions[1],
                                                                1), activation='relu'
                       )))
    models.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    models.add(MaxPooling2D(pool_size=sizeOfPool))
    models.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    models.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    models.add(MaxPooling2D(pool_size=sizeOfPool))
    models.add(Dropout(0.5))

    models.add(Flatten())
    models.add(Dense(onOfNode, activation='relu'))
    models.add(Dropout(0.5))
    models.add(Dense(noOfClasses, activation='softmax'))
    models.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return models


model = myModel()
print(model.summary())

history = model.fit(dataGen.flow(X_train, Y_train, batch_size=batchSizeVal),
                    steps_per_epoch=stepsPerEpochVal,
                    validation_data=(X_validation, Y_validation),
                    epochs=epochsVal)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

pickle_out = open("Testing/model_trained_A.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
