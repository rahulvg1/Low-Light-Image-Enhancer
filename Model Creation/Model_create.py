# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:02:19 2019

@author: Rahul
"""

def L1(yhat, y):
    loss = np.sum(np.abs(yhat-y),axis = 0)
    return loss

import numpy as np
from keras.models import Model
from keras.layers import Input,Dropout, Conv2D, MaxPool2D, concatenate, Conv2DTranspose
import cv2, glob
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage import io


imagep = glob.glob("new/low/*")
labelp = glob.glob("new/high/*")


X = []
Y = []


for img in imagep:
    image = io.imread(img)
    image = image/255.
    X.append(image)
    
for lbl in labelp:
    image = io.imread(lbl)
    image = image/255.
    Y.append(image)
    
X = np.asarray(X)
Y = np.asarray(Y)



(X_train, X_test, Y_train, Y_test) = train_test_split( X, Y, test_size=0.10, random_state=200)


inputs = Input(shape= (256, 256, 3))

  
c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPool2D((2, 2)) (c1)

c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPool2D((2, 2)) (c2)

c3 = Conv2D(128, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(128, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPool2D((2, 2)) (c3)

c4 = Conv2D(256, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(256, (3, 3), activation='relu', padding='same') (c4)
d1 = Dropout(0.5)(c4)
p4 = MaxPool2D(pool_size=(2, 2)) (d1)


c5 = Conv2D(512, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(512, (3, 3), activation='relu', padding='same') (c5)
d2 = Dropout(0.5)(c5)

u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (d2)
u6 = concatenate([u6, c4])
c6 = Conv2D(256, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(256, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (c9)


outputs = Conv2D(3, (1, 1), activation= None) (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer= "adam" , loss= L1, metrics=['accuracy'])


model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
          epochs= 50, batch_size=7, verbose =1)




model.save('Model/3.2.h5')
model.save_weights("Model/Weights/3.2._wt.h5")

