# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:10:34 2018

@author: Siddharth
"""

import cv2
from keras.models import model_from_json

# load json and create model
json_file = open('models/cnn2model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/cnn2model.h5")
print("Loaded model from disk")
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

#finding bounding rectangle
rects = [cv2.boundingRect(con) for con in cntr]


counter = 0
for rect in rects:
    counter = counter + 1
    try:
        leng = int(rect[3] * 1.2)
        pt1 = max(int(rect[1] + rect[3] // 2 - leng // 2), 0)
        pt2 = max(int(rect[0] + rect[2] // 2 - leng // 2), 0)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        if(roi.shape[0]*roi.shape[1]<100):
            continue
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        im = cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        """if(counter == 5):
            print(pt2)
            
            fig, ax = plt.subplots(1, 1, figsize=(100,100))
            ax.imshow(roi, cmap = "gray")
            im = cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)"""
        roi = roi.reshape(1,28,28, 1)
        roi = roi.astype("float32")/255
        nbr = model.predict(roi)
        text = np.argmax(nbr, axis=1)
        cv2.putText(im, str(int(text)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    except Exception as e:
        print(e)
        pass
print(counter)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
#cv2.imwrite('results/result_img3.jpg', im)