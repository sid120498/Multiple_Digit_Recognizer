# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:45:27 2018

@author: Siddharth
"""
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json

train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
 

x_train, x_val, y_train, y_val = train_test_split(
    train_df.iloc[:,1:].values, train_df.iloc[:,0].values, test_size=0.1)

fig, ax = plt.subplots(2, 1, figsize=(12,6))
ax[0].plot(x_train[0])
ax[0].set_title('784x1 data')
ax[1].imshow(x_train[0].reshape(28,28), cmap='gray')
ax[1].set_title('28x28 data')

x_train = x_train.reshape(-1, 28, 28, 1)
x_val = x_val.reshape(-1, 28, 28, 1)

x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
#example:
print(y_train[0])


model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=100, #Increase this when not on Kaggle kernel
                           verbose=1,  #1 for ETA, 0 for silent
                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed
                           callbacks=[annealer])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()

plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()

y_hat = model.predict(x_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)

x_test = test_df.values.astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1)/255

y_hat = model.predict(x_test, batch_size=64)

y_pred = np.argmax(y_hat,axis=1)

sample_submission = pd.read_csv('datasets/sample_submission.csv')
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':np.array(y_pred)})
result.to_csv('submission2.csv',index=False) 

import cv2

from skimage.feature import hog

im = cv2.imread("datasets/test_img3.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)
im_gray.shape



"""fig, ax = plt.subplots(1, 1, figsize=(100,100))
ax.imshow(im_gray, cmap = "gray")"""

# Threshold the image
#ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 23, 23)

"""fig, ax = plt.subplots(1, 1, figsize=(100,100))
ax.imshow(im_th, cmap = "gray")"""


# Find contours in the image
img, cntr, heir = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(con) for con in cntr]


counter = 0
for rect in rects:
    counter = counter + 1
    try:
        #im = cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        leng = int(rect[3] * 1.2)
        pt1 = max(int(rect[1] + rect[3] // 2 - leng // 2), 0)
        pt2 = max(int(rect[0] + rect[2] // 2 - leng // 2), 0)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        if(counter == 5):
            print(pt2)
            
            fig, ax = plt.subplots(1, 1, figsize=(100,100))
            ax.imshow(roi, cmap = "gray")
            im = cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        roi = roi.reshape(1,28,28, 1)
        roi = roi.astype("float32")/255
        nbr = model.predict(roi)
        text = np.argmax(nbr, axis=1)
        print(text)
        cv2.putText(im, str(int(text)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    except Exception as e:
        print(e)
        pass
print(counter)
cv2.imshow("Resulting Image with Rectangular ROIs", im)