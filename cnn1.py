# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:50:18 2018

@author: Siddharth
"""

# importing required libraries. 
import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_df = pd.read_csv('datasets/train.csv')
test_df = pd.read_csv('datasets/test.csv')

train_label = train_df["label"]

# dropping 'Label' column 
train = train_df.drop("label", axis = 1).values


train = train/255.0
test_df= test_df/255.0


fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i in range(15):
    ax1[i].imshow(train[i].reshape((28,28)))
    ax1[i].axis('off')
    ax1[i].set_title(train_label[i])
    
    
train_image =np.array(train).reshape(-1,28,28,1)
test_image =np.array(test_df).reshape(-1,28,28,1) 

# label encoding of train_label dataset which has category of 0 to 9 values using one hot encoding
train_label = to_categorical(train_label)


classifier = Sequential()
# step 2 - Convolution
classifier.add(Conv2D(32, (3, 3), 
                      padding = 'Same', activation="relu", input_shape=(28, 28, 1))) 

# step 3-  Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding another convulationary layer to make it deep neural net model 
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

# step 5- Full connection (hidden layers)
classifier.add(Dense(output_dim = 256, activation = 'relu'))

# adding output layer
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs=30
batch_size=90  

classifier.fit(train_image, train_label, batch_size=batch_size, epochs=epochs) 


results = classifier.predict(test_image)

# submitting the prediction
pred = []
numTest = results.shape[0]
for i in range(numTest):
    pred.append(np.argmax(results[i])) 
predictions = np.array(pred) 

sample_submission = pd.read_csv('datasets/sample_submission.csv')
result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':predictions})
result.to_csv('submission.csv',index=False) 
    
    